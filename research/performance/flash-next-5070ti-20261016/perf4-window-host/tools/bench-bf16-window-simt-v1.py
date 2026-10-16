"""Test actual GR geometries, FP32 reference error, stride updates and graph timings."""
from pathlib import Path
import hashlib
import json

import torch
import triton
from freetoken.kernel.gguf_native import bf16_gemv
from bf16_window_simt_variant import project

ROOT=Path(__file__).parent
OUTPUT=ROOT/'bf16-window-simt-components-v1.json'
if OUTPUT.exists():raise FileExistsError(OUTPUT)
torch.manual_seed(814)
torch.backends.cuda.matmul.allow_tf32=False
torch.set_float32_matmul_precision('highest')
report=dict(status='running',scope='actual GR shapes, synthetic BF16 weights and FP32 inputs; microbench not model TPS',
    variant_sha256=hashlib.sha256((ROOT/'bf16_window_simt_variant.py').read_bytes()).hexdigest(),cases=[])
for rows,width,n0 in ((324,10240,320),(320,10240,320),(10240,320,10240)):
    w=(torch.randn((rows,width),device='cuda')*.02).bfloat16()
    packed=w[:n0].contiguous().view(torch.uint8).flatten()
    p1=None if n0==rows else w[n0:].contiguous().view(torch.uint8).flatten()
    for tokens in (2,3,4):
        backing=torch.randn((tokens,width*2),device='cuda')*.5
        x=backing[:,::2]
        seed=x.clone()
        baseline=bf16_gemv(packed,x,rows,packed1=p1,first_rows=n0)
        reference=x@w.float().T
        old_us=triton.testing.do_bench_cudagraph(lambda:bf16_gemv(packed,x,rows,packed1=p1,first_rows=n0),rep=80)*1000
        choices=[]
        for bn,bk,split in (((1,256,4),(2,256,4),(4,256,4),(2,256,8),(4,256,8),(4,512,4)) if width>512 else ((4,128,1),(8,128,1),(4,256,1),(8,256,1))):
            y=torch.empty((tokens,rows*2),device='cuda')[:,::2]
            fn=lambda:project(packed,x,rows,out=y,packed1=p1,first_rows=n0,bn=bn,bk=bk,split=split)
            fn()
            torch.testing.assert_close(y,reference,rtol=2e-5,atol=2e-5)
            torch.testing.assert_close(y,baseline,rtol=3e-5,atol=3e-5)
            error=float((y-reference).abs().max())
            graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):fn()
            x.mul_(.9).add_(.01)
            graph.replay()
            expected=x@w.float().T
            torch.testing.assert_close(y,expected,rtol=2e-5,atol=2e-5)
            x.copy_(seed)
            graph.reset()
            us=triton.testing.do_bench_cudagraph(fn,rep=80)*1000
            choices.append(dict(bn=bn,bk=bk,split=split,us=us,max_abs_fp32=error,changed_graph_and_strides=True))
        row=dict(tokens=tokens,rows=rows,width=width,first_rows=n0,baseline_us=old_us,
                 baseline_max_abs_fp32=float((baseline-reference).abs().max()),choices=choices)
        report['cases'].append(row)
        OUTPUT.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
        print(json.dumps(row),flush=True)
report['status']='passed'
OUTPUT.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
