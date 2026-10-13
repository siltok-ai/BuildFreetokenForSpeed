"""Routed Q2xQ8 vector projection over existing FreeToken 18-byte blocks.

Follows Strata's route/row DP4A design, retaining FP32 expert activation scales.
This is a standalone prototype, not a new accepted wheel.
"""
import hashlib
import inspect
import json
from pathlib import Path
import sys
import capture_freetoken_reference
import torch
import triton
import triton.language as tl
ROOT=Path(__file__).resolve().parent

@triton.jit
def _dp4a_vector(Q32,S,W,IDS,Y,K:tl.constexpr,N:tl.constexpr,
                 INPUT_TOPK:tl.constexpr,BN:tl.constexpr,BG:tl.constexpr):
    route=tl.program_id(0)
    token=route//INPUT_TOPK
    expert=tl.load(IDS+route).to(tl.int64)
    rows=tl.program_id(1)*BN+tl.arange(0,BN)
    groups=tl.arange(0,BG)
    words=tl.arange(0,8)
    base=expert*N*(K//64*18)+rows[:,None]*(K//64*18)+(groups[None,:]//2)*18
    valid=(rows[:,None]<N)&(groups[None,:]<K//32)
    low=tl.load(W+base,valid,0).to(tl.uint16)
    high=tl.load(W+base+1,valid,0).to(tl.uint16)
    dw=(low|(high<<8)).to(tl.float16,bitcast=True).to(tl.float32)
    codes=tl.load(W+base[:,:,None]+2+(groups[None,:,None]%2)*8+words[None,None,:],valid[:,:,None],0).to(tl.uint32)
    packed=((codes&3)|(((codes>>2)&3)<<8)|(((codes>>4)&3)<<16)|(((codes>>6)&3)<<24)).to(tl.int32)
    x=tl.load(Q32+token*(K//4)+groups[:,None]*8+words[None,:],groups[:,None]<K//32,0).to(tl.int32)
    products=tl.inline_asm_elementwise('dp4a.s32.s32 $0, $1, $2, 0;',constraints='=r,r,r',
        args=[packed,x[None,:,:]],dtype=tl.int32,is_pure=True,pack=1)
    correction=tl.inline_asm_elementwise('dp4a.s32.s32 $0, 0x01010101, $1, 0;',constraints='=r,r',
        args=[x],dtype=tl.int32,is_pure=True,pack=1)
    dot=tl.sum(products-correction[None,:,:],2).to(tl.float32)
    dx=tl.load(S+token*(K//32)+groups,groups<K//32,0)
    value=tl.sum(dot*dw*dx[None,:],1)
    tl.store(Y+route*N+rows,value,rows<N)

def vector_project(value,bank,result,ids,input_topk):
    from freetoken.moe.fused_q2_q8 import quantize_expert_activation
    q,scales=quantize_expert_activation(value)
    k,n=value.shape[1],result.shape[1]
    _dp4a_vector[(ids.numel(),triton.cdiv(n,4))](q.view(torch.int32),scales,bank,ids,result,
        k,n,input_topk,4,triton.next_power_of_2(k//32),num_warps=4,enable_fp_fusion=False)

def candidate():
    import freetoken.moe.fused_q2_q8 as mod
    text=inspect.getsource(mod.fused_experts_q2_q8)
    old='    sorted_ids, experts, padded = moe_align_block_size(topk_ids, 16, gate_up.shape[0])'
    if text.count(old)!=1:raise RuntimeError('Unexpected frozen expert implementation')
    text=text.replace(old,'    vector_decode = not is_prefill and tokens <= 8\n    if not vector_decode:\n        sorted_ids, experts, padded = moe_align_block_size(topk_ids, 16, gate_up.shape[0])')
    old='    def project(value, bank, result, input_topk):\n        q, scales = quantize_expert_activation(value)'
    new='    def project(value, bank, result, input_topk):\n        if vector_decode:\n            return vector_project(value, bank, result, topk_ids, input_topk)\n        q, scales = quantize_expert_activation(value)'
    if text.count(old)!=1:raise RuntimeError('Unexpected project helper')
    ns=dict(mod.__dict__,vector_project=vector_project)
    exec(compile(text.replace(old,new),str(Path(__file__)), 'exec'),ns)
    return ns['fused_experts_q2_q8']

def install():
    import freetoken.moe.fused_q2_q8 as mod
    mod.fused_experts_q2_q8=candidate()

def component():
    from safetensors import safe_open
    from freetoken.models.qwen4_exp.gguf_source import GGUFModelSource
    from freetoken.moe.fused_q2_q8 import fused_experts_q2_q8
    from prototype_bf16_decode_v1 import timing
    target=ROOT/'prototype-expert-dp4a-v13-component.json'
    if target.exists():raise FileExistsError(target)
    source=GGUFModelSource([r['path'] for r in json.loads((ROOT/'complete-gguf-inventory.json').read_text())['shards']])
    sample=json.loads((ROOT/'expanded-correctness-dev3-v3/freetoken.json').read_text())['runs'][-1]
    with safe_open(sample['capture'],framework='pt',device='cpu') as f: x=f.get_tensor('input_0')[-7:,:2560].contiguous().cuda()
    h,ff=2560,640;slots=[7,4095,1027,1901,29,3028,811,2458,90,3800]
    gu=torch.empty((4096,2*ff,h//64*18),device='cuda',dtype=torch.uint8)
    down=torch.empty((4096,h,ff//64*18),device='cuda',dtype=torch.uint8)
    for expert,slot in enumerate(slots):
        g=source.tensors['blk.0.ffn_gate_exps.weight'].read_rows(expert*ff,ff)
        u=source.tensors['blk.0.ffn_up_exps.weight'].read_rows(expert*ff,ff)
        d=source.tensors['blk.0.ffn_down_exps.weight'].read_rows(expert*h,h)
        gu[slot].copy_(torch.frombuffer(bytearray(g+u),dtype=torch.uint8).reshape(2*ff,-1))
        down[slot].copy_(torch.frombuffer(bytearray(d),dtype=torch.uint8).reshape(h,-1))
    ids=torch.tensor(slots,device='cuda',dtype=torch.int32)[None].repeat(7,1)
    weights=torch.full((7,10),.1,device='cuda',dtype=torch.float32)
    proposed=candidate();checks=[]
    for tokens in [1,3,7]:
        a=fused_experts_q2_q8(x[:tokens],gu,down,weights[:tokens],ids[:tokens])
        b=proposed(x[:tokens],gu,down,weights[:tokens],ids[:tokens])
        delta=(a-b).abs();rel=float(delta.norm()/a.norm().clamp_min(1e-10))
        checks.append(dict(tokens=tokens,max_error=float(delta.max()),relative_l2=rel,
            finite=bool(torch.isfinite(a).all() and torch.isfinite(b).all()),passed=rel<=2e-3))
    speeds=dict(baseline=timing(lambda:fused_experts_q2_q8(x[:1],gu,down,weights[:1],ids[:1])),
        vector=timing(lambda:proposed(x[:1],gu,down,weights[:1],ids[:1])))
    for r in speeds.values():r['scope']='GPU event graph replay; whole expert function outputs/scratch allocated during capture; CPU/compilation excluded'
    report=dict(passed=all(r['passed'] and r['finite'] for r in checks),checks=checks,timing=speeds,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        scope='Real layer0 GGUF expert bytes with captured residual slice inputs; changed float reduction tree bounded, full model required',
        process_override=True,full_engine_acceptance=False)
    target.write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report),flush=True)
    if not report['passed']:raise RuntimeError('DP4A component failed')

def engine():
    gate=json.loads((ROOT/'prototype-expert-dp4a-v13-component.json').read_text())
    if not gate['passed']:raise RuntimeError('Component failed')
    install()
    import first_freetoken_performance_dev4_v1 as harness
    out=Path(sys.argv[sys.argv.index('--output')+1])
    try:harness.main()
    finally:
        if out.exists():
            data=json.loads(out.read_text(encoding='utf-8'));data['native_engine_only']=False
            data['status_before_prototype_tag']=data['status'];data['status']='prototype_dp4a_'+data['status']
            data['prototype']=dict(name='expert-dp4a-v13',process_override=True,full_engine_acceptance=False,
                source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
            out.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')

if __name__=='__main__':
    if '--component' in sys.argv:
        with torch.cuda.stream(torch.cuda.Stream()):component()
    else:engine()
