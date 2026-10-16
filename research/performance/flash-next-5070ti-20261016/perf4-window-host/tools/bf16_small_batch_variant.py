"""Process-only small-window BF16 weights with FP32 activations and tf32x3 dots."""
from functools import partial

import torch
import triton

GEOMETRIES={(324,10240,320),(320,10240,320),(10240,320,10240)}


def project(packed,x,rows,*,out=None,packed1=None,first_rows=None,bn=64,bk=64):
    """Use the existing validated batched FP32-input kernel at M=2..4."""
    from freetoken.kernel.gguf_native import _bf16_batch
    if out is None:out=torch.empty((len(x),rows),device=x.device,dtype=torch.float32)
    n0=rows if first_rows is None else first_rows
    p1=packed if packed1 is None else packed1
    _bf16_batch[(1,triton.cdiv(rows,bn))](packed.view(torch.bfloat16),p1.view(torch.bfloat16),x,out,
        n0,len(x),rows,x.shape[1],*x.stride(),*out.stride(),16,bn,bk,num_warps=4)
    return out


def enable(*,down_bn=32,down_bk=64,up_bn=64,up_bk=64,mode='both'):
    """Change only measured GR shapes; draft single-token and other models keep the old dispatch."""
    import freetoken.kernel.gguf_native as native
    original=native.bf16_gemv
    counts={}

    def dispatch(packed,x,rows,*,out=None,packed1=None,first_rows=None):
        """Apply a small-batch tile without converting FP32 activations to BF16."""
        n0=rows if first_rows is None else first_rows
        shape=(rows,x.shape[-1],n0)
        use=2<=len(x)<=4 and shape in GEOMETRIES
        use=use and (mode=='both' or (mode=='down')==(x.shape[-1]>512))
        if not use:return original(packed,x,rows,out=out,packed1=packed1,first_rows=first_rows)
        counts[str((len(x),*shape))]=counts.get(str((len(x),*shape)),0)+1
        bn,bk=(down_bn,down_bk) if x.shape[-1]>512 else (up_bn,up_bk)
        return project(packed,x,rows,out=out,packed1=packed1,first_rows=first_rows,bn=bn,bk=bk)

    native.bf16_gemv=dispatch
    return counts

