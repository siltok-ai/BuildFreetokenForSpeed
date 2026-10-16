"""Native indexed Q5_K correctness and graph cost versus full/copied draft heads."""
from pathlib import Path
import json
import hashlib

import torch
import triton
from freetoken.kernel.gguf_native import mmvq,quantize_q8_1
from draft_mapped_subset_variant import module
from draft_subset_variant import VOCAB,VOCAB_SHA

ROOT = Path(__file__).parent
OUTPUT = ROOT/'mapped-head-components-v1.json'
if OUTPUT.exists():raise FileExistsError(OUTPUT)
native = module()
torch.manual_seed(802)
packed = torch.randint(0,256,(248320,10,176),device='cuda',dtype=torch.uint8)
# Finite, bounded block scales; keep the quant codes independently randomized.
packed[:,:,0]=0;packed[:,:,1]=60;packed[:,:,2]=0;packed[:,:,3]=48
packed = packed.flatten()
raw = VOCAB.read_bytes()
if hashlib.sha256(raw).hexdigest()!=VOCAB_SHA:raise ValueError('Draft vocabulary changed')
ids = torch.frombuffer(bytearray(raw),dtype=torch.int32).long().sort().values.to('cuda')
copied = packed.view(248320,1760).index_select(0,ids).flatten()
checks=[]
for seed in (41,42,43):
    generator = torch.Generator(device='cuda').manual_seed(seed)
    x = torch.randn((1,2560),device='cuda',generator=generator)
    q8 = quantize_q8_1(x)
    full = mmvq(packed,q8,248320,13)
    for row_ids in (ids[:1],ids[:17],ids,ids.flip(0),torch.tensor([0,248319,0],device='cuda')):
        output = torch.empty((1,len(row_ids)),device='cuda',dtype=torch.float32)
        native.project(packed,q8,row_ids,output,torch.cuda.current_stream().cuda_stream)
        exact = torch.equal(full.index_select(1,row_ids),output)
        checks.append(dict(seed=seed,rows=len(row_ids),exact=exact))
        if not exact:raise RuntimeError('Indexed native logits differ')
out_full=torch.empty((1,248320),device='cuda');out_sub=torch.empty((1,len(ids)),device='cuda')
def full_head():mmvq(packed,q8,248320,13,out=out_full)
def copied_head():mmvq(copied,q8,len(ids),13,out=out_sub)
def mapped_head():native.project(packed,q8,ids,out_sub,torch.cuda.current_stream().cuda_stream)
times={name:triton.testing.do_bench_cudagraph(fn,rep=200)*1000 for name,fn in
       [('full',full_head),('copied',copied_head),('mapped',mapped_head)]}
result=dict(status='passed',scope='synthetic finite Q5 blocks with original native arithmetic; GPU kernel graph cost, not model TPS',
    checks=checks,microseconds=times,avoided_packed_bytes=copied.numel(),
    native_header_sha256=hashlib.sha256((ROOT/'mapped_q5_proto4_r1.cuh').read_bytes()).hexdigest(),
    vocab_sha256=VOCAB_SHA)
OUTPUT.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
print(json.dumps(result,indent=2))
