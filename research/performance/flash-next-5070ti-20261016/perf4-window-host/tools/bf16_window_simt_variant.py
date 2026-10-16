"""Small GR windows reuse BF16 weights across FP32 token lanes, with optional split K."""
import torch
import triton
import triton.language as tl


@triton.jit
def _window_rows(W0,W1,X,Y,N0:tl.constexpr,M:tl.constexpr,N:tl.constexpr,K:tl.constexpr,
    XS0:tl.constexpr,XS1:tl.constexpr,YS0:tl.constexpr,YS1:tl.constexpr,
    BM:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr,SPLIT:tl.constexpr):
    rows=tl.program_id(0)*BN+tl.arange(0,BN)
    part=tl.program_id(1)
    tokens=tl.arange(0,BM)
    cols=tl.arange(0,BK)
    base=tl.where(rows<N0,W0+rows*K,W1+(rows-N0)*K)
    acc=tl.full((BM,BN,BK),0.,tl.float32)
    steps=triton.cdiv(K,BK*SPLIT)
    for step in range(steps):
        ks=(part*steps+step)*BK+cols
        weight=tl.load(base[:,None]+ks[None,:],(rows[:,None]<N)&(ks[None,:]<K),0).to(tl.float32)
        value=tl.load(X+tokens[:,None]*XS0+ks[None,:]*XS1,(tokens[:,None]<M)&(ks[None,:]<K),0)
        acc+=value[:,None,:]*weight[None,:,:]
    result=tl.sum(acc,2)
    tl.store(Y+part*M*N+tokens[:,None]*YS0+rows[None,:]*YS1,result,
             (tokens[:,None]<M)&(rows[None,:]<N))


@triton.jit
def _sum_parts(P,Y,M:tl.constexpr,N:tl.constexpr,SPLIT:tl.constexpr,
    YS0:tl.constexpr,YS1:tl.constexpr,B:tl.constexpr):
    cell=tl.program_id(0)*B+tl.arange(0,B)
    parts=tl.arange(0,triton.next_power_of_2(SPLIT))
    values=tl.load(P+parts[:,None]*M*N+cell[None,:],(parts[:,None]<SPLIT)&(cell[None,:]<M*N),0)
    tl.store(Y+(cell//N)*YS0+(cell%N)*YS1,tl.sum(values,0),cell<M*N)


def project(packed,x,rows,*,out=None,packed1=None,first_rows=None,bn=4,bk=256,split=1):
    """Keep FP32 products/reductions; separate K slices supply enough blocks for small N."""
    if out is None:out=torch.empty((len(x),rows),device=x.device,dtype=torch.float32)
    n0=rows if first_rows is None else first_rows
    p1=packed if packed1 is None else packed1
    temporary=out if split==1 else torch.empty((split,len(x),rows),device=x.device,dtype=torch.float32)
    ys=out.stride() if split==1 else temporary.stride()[1:]
    _window_rows[(triton.cdiv(rows,bn),split)](packed.view(torch.bfloat16),p1.view(torch.bfloat16),x,temporary,
        n0,len(x),rows,x.shape[1],*x.stride(),*ys,triton.next_power_of_2(len(x)),bn,bk,split,
        num_warps=4,enable_fp_fusion=False)
    if split>1:
        _sum_parts[(triton.cdiv(len(x)*rows,128),)](temporary,out,len(x),rows,split,*out.stride(),128,num_warps=4)
    return out


def enable(*,down_bn=2,down_bk=256,down_split=4,up_bn=8,up_bk=128,mode='both'):
    """Use explicit tested geometries; one-token decode keeps the accepted implementation."""
    import freetoken.kernel.gguf_native as native
    original=native.bf16_gemv
    shapes={(324,10240,320),(320,10240,320),(10240,320,10240)}
    counts={}

    def dispatch(packed,x,rows,*,out=None,packed1=None,first_rows=None):
        """Reuse weights within a small verify/catch-up window, preserving source segments."""
        n0=rows if first_rows is None else first_rows
        use=2<=len(x)<=4 and (rows,x.shape[-1],n0) in shapes
        use=use and (mode=='both' or (mode=='down')==(x.shape[-1]>512))
        if not use:return original(packed,x,rows,out=out,packed1=packed1,first_rows=first_rows)
        key=str((len(x),rows,x.shape[-1],n0));counts[key]=counts.get(key,0)+1
        bn,bk,split=(down_bn,down_bk,down_split) if x.shape[-1]>512 else (up_bn,up_bk,1)
        return project(packed,x,rows,out=out,packed1=packed1,first_rows=first_rows,bn=bn,bk=bk,split=split)

    native.bf16_gemv=dispatch
    return counts

