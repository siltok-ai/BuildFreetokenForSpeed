"""Q2 GPU原路径，新增CPU route不读payload且写零的分支。"""
import triton
import triton.language as tl

@triton.jit
def _hybrid_routed_vector(Q32,S,W,IDS,Y,K:tl.constexpr,N:tl.constexpr,
                 INPUT_TOPK:tl.constexpr,BN:tl.constexpr,BG:tl.constexpr):
    route=tl.program_id(0)
    token=route//INPUT_TOPK
    expert=tl.load(IDS+route).to(tl.int64)
    if expert<0:
        zero_rows=tl.program_id(1)*BN+tl.arange(0,BN)
        tl.store(Y+route*N+zero_rows,0.,zero_rows<N)
        return
    rows=tl.program_id(1)*BN+tl.arange(0,BN)
    groups=tl.arange(0,BG)
    words=tl.arange(0,8)
    base=expert*N*(K//64*18)+rows[:,None]*(K//64*18)+(groups[None,:]//2)*18
    valid=(rows[:,None]<N)&(groups[None,:]<K//32)
    # Q2 blocks are 18 bytes: load headers as bytes, never assume int32 alignment.
    low=tl.load(W+base,valid,0).to(tl.uint16)
    high=tl.load(W+base+1,valid,0).to(tl.uint16)
    dw=(low|(high<<8)).to(tl.float16,bitcast=True).to(tl.float32)
    codes=tl.load(W+base[:,:,None]+2+(groups[None,:,None]%2)*8+words[None,None,:],valid[:,:,None],0).to(tl.uint32)
    packed=((codes&3)|(((codes>>2)&3)<<8)|(((codes>>4)&3)<<16)|(((codes>>6)&3)<<24)).to(tl.int32)
    x=tl.load(Q32+token*(K//4)+groups[:,None]*8+words[None,:],groups[:,None]<K//32,0).to(tl.int32)
    # The unsigned two-bit codes need the exact integer sum(x) correction.
    products=tl.inline_asm_elementwise('dp4a.s32.s32 $0, $1, $2, 0;',constraints='=r,r,r',
        args=[packed,x[None,:,:]],dtype=tl.int32,is_pure=True,pack=1)
    correction=tl.inline_asm_elementwise('dp4a.s32.s32 $0, 0x01010101, $1, 0;',constraints='=r,r',
        args=[x],dtype=tl.int32,is_pure=True,pack=1)
    dot=tl.sum(products-correction[None,:,:],2).to(tl.float32)
    dx=tl.load(S+token*(K//32)+groups,groups<K//32,0)
    value=tl.sum(dot*dw*dx[None,:],1)
    tl.store(Y+route*N+rows,value,rows<N)
