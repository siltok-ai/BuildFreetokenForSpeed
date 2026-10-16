"""从当前source5工作快照生成独立可复核的cached-adapt实验，不编辑冻结材料。"""
from pathlib import Path
import hashlib
import json
import shutil

ROOT=Path(__file__).parent
SOURCE=Path(r'C:\workspace\build\freetoken-mats\flash-perf-candidate5\build-source')
BASE=Path(r'C:\workspace\build\freetoken-mats\flash-perf-materials-04\candidate-source\build-source')
SNAPSHOT=ROOT/'candidate5-working-source-v1'
if SNAPSHOT.exists():raise FileExistsError(SNAPSHOT)
SNAPSHOT.mkdir()
files=[]
for name in ('python/freetoken/moe/q2_adaptive.py','python/freetoken/moe/q2_residency.py',
    'python/freetoken/engine/config.py','tests/moe/test_q2_adaptive.py','tests/moe/test_q2_residency.py'):
    target=SNAPSHOT/name;target.parent.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(SOURCE/name,target)
    files.append(dict(path=name,sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
        base_sha256=hashlib.sha256((BASE/name).read_bytes()).hexdigest()))
(SNAPSHOT/'index.json').write_text(json.dumps(dict(scope='Unversioned source5 GPU metadata publication and zero dynamic quota; not perf4 frozen artifacts',
    parent_profile='strataq2perf4',files=files),indent=2)+'\n')
source=(SOURCE/'python/freetoken/moe/q2_adaptive.py').read_text(encoding='utf-8')
begin=source.index('    def adapt(self):')
end=source.index('    def step(self):',begin)
method=source[begin:end]
old='''        _exchange[(len(c.bank_schema),32)](c._copy_dst_ptrs,self.source_ptrs,
            c._copy_feat_bytes,self.gpu_slots,self.count,self.max_swaps,32,1024,num_warps=8)'''
new='''        exchange[(len(c.bank_schema),32)](c._copy_dst_ptrs,self.source_ptrs,
            c._copy_feat_bytes,self.gpu_slots,self.count,self.commands,c.slot_for_id.view(-1),
            self.cached_copy_counts,self.max_swaps,32,1024,p.experts,p.protected_begin,num_warps=8)'''
assert method.count(old)==1
method=method.replace(old,new)
prefix='''"""复用GPU动态副本的适应性所有者交换，保留source5发布协议与完整原始专家。"""
import time
import numpy as np
import torch
import triton
from freetoken.moe import q2_adaptive as base
from freetoken.moe.q2_adaptive import _publish_exchange
from q2_cached_exchange_kernel_v1 import exchange


class CachedQ2Owners(base.AdaptiveQ2Owners):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.cached_copy_counts=torch.zeros(2,device=self.cache.device,dtype=torch.int64)

'''
suffix='''
def enable(max_swaps=96):
    """显式实验容量；调用者传入的组件测试容量保留，不覆盖它。"""
    class SelectedCachedOwners(CachedQ2Owners):
        def __init__(self,*args,**kwargs):
            kwargs.setdefault('max_swaps',max_swaps)
            super().__init__(*args,**kwargs)
    base.AdaptiveQ2Owners=SelectedCachedOwners
'''
target=ROOT/'q2_cached_adaptive_variant_v1.py'
if target.exists():raise FileExistsError(target)
target.write_text(prefix+method+suffix,encoding='utf-8')
source=(ROOT/'probe33-source5.py').read_text(encoding='utf-8')
source=source.replace("p.add_argument('--mapped-embedding',action='store_true')",
    "p.add_argument('--mapped-embedding',action='store_true')\n    p.add_argument('--cached-adapt',action='store_true')\n    p.add_argument('--adapt-max-swaps',type=int,default=96)\n    p.add_argument('--q2-lut',action='store_true')")
source=source.replace("report['variants']={}","""report['variants']={}
    if args.cached_adapt:
        from q2_cached_adaptive_variant_v1 import enable
        enable(max_swaps=args.adapt_max_swaps)
        report['variants']['cached_adapt']=dict(max_swaps=args.adapt_max_swaps,source_sha256=hashlib.sha256(Path('q2_cached_adaptive_variant_v1.py').read_bytes()).hexdigest(),kernel_sha256=hashlib.sha256(Path('q2_cached_exchange_kernel_v1.py').read_bytes()).hexdigest())
    if args.q2_lut:
        from q2_lut_variant_v1 import enable
        enable()
        report['variants']['q2_lut']=hashlib.sha256(Path('q2_lut_variant_v1.py').read_bytes()).hexdigest()
""")
source=source.replace("        report['runs'].append(row)","        if args.cached_adapt:\n            row['cached_exchange_sources']=llm.engine.q2_adaptive_owners.cached_copy_counts.cpu().tolist()\n        report['runs'].append(row)")
source=source.replace("'unversioned_candidate5_python_source_with_frozen_perf4_native_inputs_not_delivery'",
    "'source5_with_external_cached_exchange_and_optional_Q2_LUT_not_delivery'")
target=ROOT/'probe35-cached-adapt.py'
if target.exists():raise FileExistsError(target)
target.write_text(source,encoding='utf-8')
source=(ROOT/'test-candidate5-source-v1.py').read_text(encoding='utf-8')
source=source.replace('import pytest','from q2_cached_adaptive_variant_v1 import enable\nenable()\nimport pytest')
target=ROOT/'test-cached-adaptation-v1.py'
if target.exists():raise FileExistsError(target)
target.write_text(source,encoding='utf-8')
print('Saved source5 snapshot and prepared cached exchange experiment')
