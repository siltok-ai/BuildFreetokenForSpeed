"""Verify CUDA alias identity, real Q3_K embedding rows and changed-input graph replay."""
from pathlib import Path
import json
import gc

import torch
from freetoken.kernel.pinned import alloc_pinned_tensor
from freetoken.kernel.gguf_quant import dequantize_rows
from mapped_host_bytes import MappedHostBytes

ROOT=Path(__file__).parent
OUT=ROOT/'mapped-host-bytes-components-v1.json'
if OUT.exists():raise FileExistsError(OUT)
torch.manual_seed(827)
torch.cuda.set_device(0)
vocab,width=257,2560
packed=torch.randint(0,256,(vocab,width//256,110),dtype=torch.uint8)
scales=(torch.rand((vocab,width//256))*.01+.001).half().contiguous()
packed[:,:,108:110]=scales.view(torch.uint8).reshape(vocab,width//256,2)
host=alloc_pinned_tensor(packed.numel(),dtype=torch.uint8)
host.copy_(packed.flatten())
interface=MappedHostBytes(host)
alias=interface.alias(torch.device('cuda',0))
gpu=packed.flatten().cuda()
checks=[]
for count in (1,4,64,2048):
    ids=torch.randint(0,vocab,(count,),device='cuda',dtype=torch.int32)
    ids[0]=0
    if count>1:ids[-1]=vocab-1
    normal=dequantize_rows(gpu,ids,width,11)
    mapped=dequantize_rows(alias,ids,width,11)
    if not torch.equal(normal,mapped):raise RuntimeError('Mapped Q3_K embedding rows differ')
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):replayed=dequantize_rows(alias,ids,width,11)
    ids.add_(17).remainder_(vocab)
    graph.replay()
    expected=dequantize_rows(gpu,ids,width,11)
    if not torch.equal(replayed,expected):raise RuntimeError('Mapped graph read stale input ids')
    graph.reset()
    checks.append(dict(tokens=count,exact=True,changed_graph=True))
# Keep the interface as the explicit owner, then prove it survives unrelated Python GC.
del host
gc.collect()
torch.cuda.synchronize()
if not torch.equal(alias.cpu(),gpu.cpu()):raise RuntimeError('Mapped owner lifetime was lost')
report=dict(status='passed',checks=checks,bytes=alias.numel(),cuda_alias_ptr=alias.data_ptr(),
    device_visible_ptr=interface.ptr,host_ptr=interface.owner.data_ptr(),zero_copy=True,
    ownership='interface owns CPU storage; no alias mutation; drain graph readers before release')
OUT.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
print(json.dumps(report,indent=2))
