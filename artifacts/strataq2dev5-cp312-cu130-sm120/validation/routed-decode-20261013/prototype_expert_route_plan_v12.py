"""Decode-only routed plans, no full-cache histogram and no arithmetic change.

Process-local diagnosis on installed dev4. Does not modify engine source/wheel.
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
def _route_plan(SORTED, PADDED, ROUTES:tl.constexpr, B:tl.constexpr):
    i=tl.arange(0,B)
    total=ROUTES*16
    value=tl.where(i%16==0,i//16,ROUTES)
    tl.store(SORTED+i,value,i<total)
    tl.store(PADDED+tl.zeros((B,),tl.int32),total,i==0)

def route_plan(ids):
    routes=ids.numel()
    sorted_ids=torch.empty((routes*16,),device=ids.device,dtype=torch.int32)
    padded=torch.empty((1,),device=ids.device,dtype=torch.int32)
    _route_plan[(1,)](sorted_ids,padded,routes,triton.next_power_of_2(routes*16),num_warps=4)
    return sorted_ids,ids.flatten(),padded

def candidate():
    import freetoken.moe.fused_q2_q8 as mod
    original=mod.fused_experts_q2_q8
    text=inspect.getsource(original)
    old='    sorted_ids, experts, padded = moe_align_block_size(topk_ids, 16, gate_up.shape[0])'
    if text.count(old)!=1:raise RuntimeError('Unexpected frozen function')
    new='    if not is_prefill and tokens <= 8:\n        sorted_ids, experts, padded = route_plan(topk_ids)\n    else:\n        sorted_ids, experts, padded = moe_align_block_size(topk_ids, 16, gate_up.shape[0])'
    namespace=dict(mod.__dict__,route_plan=route_plan)
    exec(compile(text.replace(old,new),str(Path(__file__)), 'exec'),namespace)
    return namespace['fused_experts_q2_q8']

def install():
    import freetoken.moe.fused_q2_q8 as mod
    mod.fused_experts_q2_q8=candidate()

def component():
    from safetensors import safe_open
    from freetoken.models.qwen4_exp.gguf_source import GGUFModelSource
    from freetoken.moe.fused_q2_q8 import fused_experts_q2_q8
    from prototype_bf16_decode_v1 import timing
    target=ROOT/'prototype-expert-route-plan-v12-component.json'
    if target.exists():raise FileExistsError(target)
    source=GGUFModelSource([r['path'] for r in json.loads((ROOT/'complete-gguf-inventory.json').read_text())['shards']])
    sample=json.loads((ROOT/'expanded-correctness-dev3-v3/freetoken.json').read_text())['runs'][-1]
    with safe_open(sample['capture'],framework='pt',device='cpu') as f: x=f.get_tensor('input_0')[-7:,:2560].contiguous().cuda()
    # Real canonical expert rows assigned to sparse GPU cache slots; all routes consume only these initialized slots.
    h,ff=2560,640;slots=[7,4095,1027,1901,29,3028,811,2458,90,3800]
    gu=torch.empty((4096,2*ff,h//64*18),device='cuda',dtype=torch.uint8)
    down=torch.empty((4096,h,ff//64*18),device='cuda',dtype=torch.uint8)
    evidence=[]
    for expert,slot in enumerate(slots):
        g=source.tensors['blk.0.ffn_gate_exps.weight'].read_rows(expert*ff,ff)
        u=source.tensors['blk.0.ffn_up_exps.weight'].read_rows(expert*ff,ff)
        d=source.tensors['blk.0.ffn_down_exps.weight'].read_rows(expert*h,h)
        gu[slot].copy_(torch.frombuffer(bytearray(g+u),dtype=torch.uint8).reshape(2*ff,-1))
        down[slot].copy_(torch.frombuffer(bytearray(d),dtype=torch.uint8).reshape(h,-1))
        evidence.append(dict(canonical_expert=expert,cache_slot=slot,gate_up_sha256=hashlib.sha256(g+u).hexdigest(),down_sha256=hashlib.sha256(d).hexdigest()))
    ids=torch.tensor(slots,device='cuda',dtype=torch.int32)[None].repeat(7,1)
    weights=torch.full((7,10),.1,device='cuda',dtype=torch.float32)
    proposed=candidate();checks=[]
    for tokens in [1,3,7]:
        a=fused_experts_q2_q8(x[:tokens],gu,down,weights[:tokens],ids[:tokens])
        b=proposed(x[:tokens],gu,down,weights[:tokens],ids[:tokens])
        delta=(a-b).abs();rel=float(delta.norm()/a.norm().clamp_min(1e-10))
        checks.append(dict(tokens=tokens,max_error=float(delta.max()),relative_l2=rel,
            finite=bool(torch.isfinite(a).all() and torch.isfinite(b).all()),passed=rel<=2e-3))
    # Repeated slot routes from different tokens and duplicate slots within a token are legitimate plan inputs.
    repeated=ids[:3].clone();repeated[:,1]=repeated[:,0]
    a=fused_experts_q2_q8(x[:3],gu,down,weights[:3],repeated)
    b=proposed(x[:3],gu,down,weights[:3],repeated)
    rel=float((a-b).norm()/a.norm().clamp_min(1e-10))
    checks.append(dict(tokens=3,duplicate_route=True,relative_l2=rel,finite=bool(torch.isfinite(b).all()),passed=rel<=2e-3))
    speeds=dict(baseline=timing(lambda:fused_experts_q2_q8(x[:1],gu,down,weights[:1],ids[:1])),
        routed=timing(lambda:proposed(x[:1],gu,down,weights[:1],ids[:1])))
    report=dict(passed=all(r['passed'] and r['finite'] for r in checks),checks=checks,timing=speeds,
        canonical_experts=evidence,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        scope='Real GGUF layer0 expert bytes with captured residual slice inputs; full-model check remains necessary',
        process_override=True,full_engine_acceptance=False)
    target.write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report),flush=True)
    if not report['passed']:raise RuntimeError('Route plan component failed')

def engine():
    gate=json.loads((ROOT/'prototype-expert-route-plan-v12-component.json').read_text())
    if not gate['passed']:raise RuntimeError('Component failed')
    install()
    import first_freetoken_performance_dev4_v1 as harness
    out=Path(sys.argv[sys.argv.index('--output')+1])
    try:harness.main()
    finally:
        if out.exists():
            data=json.loads(out.read_text(encoding='utf-8'))
            data['native_engine_only']=False
            data['status_before_prototype_tag']=data['status']
            data['status']='prototype_route_plan_'+data['status']
            data['prototype']=dict(name='expert-route-plan-v12',process_override=True,full_engine_acceptance=False,
                source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
            out.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')

if __name__=='__main__':
    if '--component' in sys.argv:
        with torch.cuda.stream(torch.cuda.Stream()):component()
    else:engine()
