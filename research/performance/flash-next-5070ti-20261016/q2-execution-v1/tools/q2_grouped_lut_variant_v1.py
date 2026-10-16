"""独立小窗口原型：一次生成专家到token的路由表，供gate/up和down共同使用。"""
import torch
import triton
import triton.language as tl
from q2_lut_variant_v1 import signed_codes


@triton.jit
def _build_groups(IDS,META,TOKENS:tl.constexpr,TOPK:tl.constexpr,BT:tl.constexpr,BK:tl.constexpr):
    route=tl.program_id(0)
    expert=tl.load(IDS+route)
    tokens=tl.arange(0,BT)
    ranks=tl.arange(0,BK)
    offsets=tokens[:,None]*TOPK+ranks[None,:]
    ids=tl.load(IDS+offsets,(tokens[:,None]<TOKENS)&(ranks[None,:]<TOPK),-1)
    selected=tl.min(tl.where(ids==expert,offsets,TOKENS*TOPK),1)
    leader=tl.min(selected,0)
    tl.store(META+route*(BT+1),leader)
    tl.store(META+route*(BT+1)+1+tokens,selected)


@triton.jit
def _small_precomputed(Q32,S,W,IDS,META,Y,K:tl.constexpr,N:tl.constexpr,
    TOKENS:tl.constexpr,TOPK:tl.constexpr,INPUT_TOPK:tl.constexpr,BT:tl.constexpr,BN:tl.constexpr,BG:tl.constexpr):
    route=tl.program_id(0)
    if route!=tl.load(META+route*(BT+1)):return
    expert=tl.load(IDS+route).to(tl.int64)
    token_rows=tl.arange(0,BT)
    selected=tl.load(META+route*(BT+1)+1+token_rows)
    rows=tl.program_id(1)*BN+tl.arange(0,BN)
    groups=tl.arange(0,BG)
    words=tl.arange(0,8)
    base=expert*N*(K//64*9)+rows[:,None]*(K//64*9)+(groups[None,:]//2)*9
    valid=(rows[:,None]<N)&(groups[None,:]<K//32)
    scale=tl.load(W+base,valid,0).to(tl.uint16)
    dw=scale.to(tl.float16,bitcast=True).to(tl.float32)
    codes=tl.load(W+base[:,:,None]+1+(groups[None,:,None]%2)*4+words[None,None,:]//2,
        valid[:,:,None],0).to(tl.uint32)
    packed=signed_codes(codes,words[None,None,:]%2!=0)
    for token_index in tl.static_range(TOKENS):
        selected_route=tl.sum(tl.where(token_rows==token_index,selected,0),0)
        if selected_route<TOKENS*TOPK:
            input_row=selected_route//INPUT_TOPK
            x=tl.load(Q32+input_row*(K//4)+groups[:,None]*8+words[None,:],groups[:,None]<K//32,0).to(tl.int32)
            products=tl.inline_asm_elementwise('dp4a.s32.s32 $0, $1, $2, 0;',constraints='=r,r,r',
                args=[packed,x[None,:,:]],dtype=tl.int32,is_pure=True,pack=1)
            dot=tl.sum(products,2).to(tl.float32)
            dx=tl.load(S+input_row*(K//32)+groups,groups<K//32,0)
            value=tl.sum(dot*dw*dx[None,:],1)
            tl.store(Y+selected_route*N+rows,value,rows<N)


def make_projector(original):
    """仅测试进程用闭包承接同一专家调用的两次投影，图内顺序生成/消费GPU表。"""
    pending={}

    def project(value,bank,result,ids,input_topk):
        """2..4token同窗口共用压缩权重，其他形状保持既有向量实现。"""
        tokens,topk=ids.shape
        if not 2<=tokens<=4:return original(value,bank,result,ids,input_topk)
        bt=triton.next_power_of_2(tokens)
        if input_topk==topk:
            meta=torch.empty((ids.numel(),bt+1),device=ids.device,dtype=torch.int32)
            _build_groups[(ids.numel(),)](ids,meta,tokens,topk,bt,triton.next_power_of_2(topk),num_warps=4)
            pending.update(ids=ids,meta=meta)
        else:
            assert input_topk==1 and pending['ids'] is ids
            meta=pending.pop('meta')
            pending.pop('ids')
        from freetoken.moe.fused_q2_q8 import quantize_expert_activation
        q,scales=quantize_expert_activation(value)
        k,n=value.shape[1],result.shape[1]
        _small_precomputed[(ids.numel(),triton.cdiv(n,8))](q.view(torch.int32),scales,bank.view(torch.int16),ids,meta,result,
            k,n,tokens,topk,input_topk,bt,8,triton.next_power_of_2(k//32),num_warps=4,enable_fp_fusion=False)

    return project


def enable():
    """仅替换当前进程；交付实现需把表所有权显式放入fused调用。"""
    from freetoken.moe import fused_q2_q8 as module
    from q2_lut_variant_v1 import project as original
    module._routed_project=make_projector(original)
    return original
