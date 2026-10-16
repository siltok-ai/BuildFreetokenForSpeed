"""Actual mapped-host Q2 payload copy, with a working set larger than GPU L2."""
import hashlib
import json
from pathlib import Path

import torch
import triton
from freetoken.kernel.pinned import alloc_pinned_tensor, device_ptr
from freetoken.moe.q2_residency import _owner_gather
from gather_aligned_variant import _owner_gather_aligned

ROOT = Path(__file__).parent
OUTPUT = ROOT / 'gather-aligned-components-v1.json'
if OUTPUT.exists():raise FileExistsError(OUTPUT)
torch.manual_seed(771)
torch.cuda.set_device(0)
ROWS = 512
FEATS = (921600, 460800)
host = [alloc_pinned_tensor(ROWS, width, dtype=torch.uint8) for width in FEATS]
for tensor in host:
    tensor.copy_(torch.randint(0,256,tensor.shape,dtype=torch.uint8))
gpu = [tensor.to('cuda') for tensor in host]
destinations = [[torch.empty_like(tensor) for tensor in gpu] for _ in range(2)]
ptrs = [torch.tensor([t.data_ptr() for t in bank],device='cuda',dtype=torch.int64) for bank in destinations]
features = torch.tensor(FEATS,device='cuda',dtype=torch.int64)
result = dict(status='running',scope='mapped host and GPU owner copies, actual Q2 payload geometry; not model TPS',cases=[],
              variant_sha256=hashlib.sha256((ROOT/'gather_aligned_variant.py').read_bytes()).hexdigest())


def save():
    """Keep completed checks even if a later workload fails."""
    OUTPUT.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')


for host_fraction in (1.0, 0.55, 0.0):
    table = []
    for bank, width in enumerate(FEATS):
        host_base = device_ptr(host[bank])
        gpu_base = gpu[bank].data_ptr()
        assert host_base % 16 == gpu_base % 16 == width % 16 == 0
        table.append([host_base+row*width if row < int(ROWS*host_fraction) else gpu_base+row*width for row in range(ROWS)])
    sources = torch.tensor(table,device='cuda',dtype=torch.int64)
    for count in (1, 4, 16, 40):
        indices = []
        for step in range(64 if count==1 else 32):
            ids = ((torch.arange(count,dtype=torch.int32)+step*count)%ROWS).to('cuda')
            indices.append(ids)
        destination_ids = torch.arange(count,device='cuda',dtype=torch.int32)
        n = torch.tensor([count],device='cuda',dtype=torch.int64)

        def launch(which, ids):
            kernel = (_owner_gather,_owner_gather_aligned)[which]
            kernel[(2,32)](ptrs[which],sources,features,destination_ids,ids,n,ROWS,32,1024,num_warps=8)

        for tensor in [*destinations[0],*destinations[1]]:tensor.zero_()
        launch(0,indices[0]);launch(1,indices[0])
        torch.cuda.synchronize()
        expected = indices[0].cpu().long()
        exact = all(torch.equal(destinations[0][bank][:count],host[bank].index_select(0,expected).to('cuda'))
                    and torch.equal(destinations[0][bank],destinations[1][bank]) for bank in range(2))
        if not exact:raise RuntimeError('Mapped gather bytes differ')
        def workload(which):
            for ids in indices:launch(which,ids)
        old_us = triton.testing.do_bench_cudagraph(lambda:workload(0),rep=100)*1000/len(indices)
        new_us = triton.testing.do_bench_cudagraph(lambda:workload(1),rep=100)*1000/len(indices)
        row = dict(host_fraction=host_fraction,count=count,calls_per_graph=len(indices),exact=exact,
                   baseline_us=old_us,aligned_us=new_us,gain_percent=(old_us/new_us-1)*100)
        result['cases'].append(row);save();print(json.dumps(row),flush=True)
result['status']='passed';save()
