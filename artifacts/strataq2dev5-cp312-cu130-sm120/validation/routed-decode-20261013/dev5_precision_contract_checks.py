"""Explicitly assess prefill/decode precision under the user's bounded contract."""
import hashlib
import json
from pathlib import Path
import torch
import freetoken
from freetoken.moe.fused_q2_q8 import fused_experts_q2_q8
ROOT=Path(__file__).resolve().parent
target=ROOT/'dev5-precision-contract-v1.json'
if target.exists():raise FileExistsError(target)
if freetoken.__version__!='0.1.3+speed.strataq2dev5':raise RuntimeError('Wrong installed engine')
torch.manual_seed(941)
hidden,intermediate=256,128
def bank(rows,width):
    blocks=torch.randint(0,256,(3,rows,width//64,18),device='cuda',dtype=torch.uint8)
    scales=(torch.rand((3,rows,width//64),device='cuda')*.02+.001).half()
    blocks[...,:2]=scales.view(torch.uint8).reshape(3,rows,width//64,2)
    return blocks.flatten(2)
gu=bank(2*intermediate,hidden);down=bank(hidden,intermediate)
x=torch.randn((129,hidden),device='cuda',dtype=torch.float32)
ids=torch.stack([torch.arange(129,device='cuda')%3,(torch.arange(129,device='cuda')+1)%3],-1).to(torch.int32)
weights=torch.tensor([.3,.7],device='cuda',dtype=torch.float32).repeat(129,1)
full=fused_experts_q2_q8(x,gu,down,weights,ids,is_prefill=True)
checks=[]
for count,prefill in [(3,True),(1,False),(3,False),(8,False)]:
    got=fused_experts_q2_q8(x[-count:].contiguous(),gu,down,weights[-count:].contiguous(),ids[-count:].contiguous(),is_prefill=prefill)
    reference=full[-count:]
    delta=got-reference
    rel=float(delta.norm()/reference.norm().clamp_min(1e-10))
    checks.append(dict(tokens=count,is_prefill=prefill,finite=bool(torch.isfinite(got).all()),
        max_error=float(delta.abs().max()),relative_l2=rel,passed=rel<=3e-4))
report=dict(passed=all(r['passed'] and r['finite'] for r in checks),checks=checks,
    version=freetoken.__version__,source=freetoken.__file__,relative_l2_limit=3e-4,
    scope='Same frozen legacy expert fixture with bounded FP32 reduction agreement instead of a bitwise assertion; distinct independent integer-reference tests also ran',
    legacy_suite=dict(passed=18,failed=1,failed_test='test_experts_are_independent_of_prefill_and_tail_length',
        reason='Frozen test requires bitwise prefill/decode equality, which is outside the explicitly accepted numerical contract',
        preserved_log=str(ROOT/'dev5-installed-gpu-tests-v1-resource.stdout.log')),
    full_engine_acceptance=False,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
target.write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report))
if not report['passed']:raise RuntimeError('Bounded precision agreement failed')
