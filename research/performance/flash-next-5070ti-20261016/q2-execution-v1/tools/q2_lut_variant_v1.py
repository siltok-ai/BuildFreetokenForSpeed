"""借鉴Strata的Q2 byte-permute表；保持原始权重、FP32 scale及Q8编码。"""
import torch
import triton
import triton.language as tl


@triton.jit
def signed_codes(code16,second):
    # The duplicated table maps every two-bit value to -1,0,1,2. Mask the prmt
    # sign selectors: CUDA __byte_perm uses the three-bit index of each nibble.
    selector=tl.where(second,0x7362,0x5140)
    return tl.inline_asm_elementwise(
        '{ .reg .b32 even,odd; prmt.b32 even,0x020100FF,0x020100FF,$1; '
        'prmt.b32 odd,0x020100FF,0x020100FF,$2; prmt.b32 $0,even,odd,$3; }',
        constraints='=r,r,r,r',args=[code16&0x7777,(code16>>2)&0x7777,selector],
        dtype=tl.int32,is_pure=True,pack=1)


@triton.jit
def _routed_lut(Q32,S,W16,IDS,Y,K:tl.constexpr,N:tl.constexpr,
                 INPUT_TOPK:tl.constexpr,BN:tl.constexpr,BG:tl.constexpr):
    route=tl.program_id(0)
    token=route//INPUT_TOPK
    expert=tl.load(IDS+route).to(tl.int64)
    rows=tl.program_id(1)*BN+tl.arange(0,BN)
    groups=tl.arange(0,BG)
    words=tl.arange(0,8)
    base=expert*N*(K//64*9)+rows[:,None]*(K//64*9)+(groups[None,:]//2)*9
    valid=(rows[:,None]<N)&(groups[None,:]<K//32)
    scale=tl.load(W16+base,valid,0).to(tl.uint16)
    dw=scale.to(tl.float16,bitcast=True).to(tl.float32)
    codes=tl.load(W16+base[:,:,None]+1+(groups[None,:,None]%2)*4+words[None,None,:]//2,
        valid[:,:,None],0).to(tl.uint32)
    packed=signed_codes(codes,words[None,None,:]%2!=0)
    x=tl.load(Q32+token*(K//4)+groups[:,None]*8+words[None,:],groups[:,None]<K//32,0).to(tl.int32)
    products=tl.inline_asm_elementwise('dp4a.s32.s32 $0, $1, $2, 0;',constraints='=r,r,r',
        args=[packed,x[None,:,:]],dtype=tl.int32,is_pure=True,pack=1)
    dot=tl.sum(products,2).to(tl.float32)
    dx=tl.load(S+token*(K//32)+groups,groups<K//32,0)
    value=tl.sum(dot*dw*dx[None,:],1)
    tl.store(Y+route*N+rows,value,rows<N)


def project(value,bank,result,ids,input_topk,*,bn=8,warps=4):
    """仅建立两字节对齐的零复制别名，不重打包或展开专家。"""
    from freetoken.moe.fused_q2_q8 import quantize_expert_activation
    q,scales=quantize_expert_activation(value)
    k,n=value.shape[1],result.shape[1]
    _routed_lut[(ids.numel(),triton.cdiv(n,bn))](q.view(torch.int32),scales,bank.view(torch.int16),
        ids,result,k,n,input_topk,bn,triton.next_power_of_2(k//32),
        num_warps=warps,enable_fp_fusion=False)


def enable():
    """只替换实验进程的向量投影；大批prefill与所有权机制保持原有实现。"""
    from freetoken.moe import fused_q2_q8 as module
    original=module._routed_project
    module._routed_project=project
    return original
