"""Consolidate only validated candidates into a new tree, preserving each patch stage."""
from pathlib import Path
import hashlib
import json
import ast

RUN=Path(__file__).parent
SOURCE=Path(r'C:\workspace\build\freetoken-mats\flash-perf-candidate6\build-source')
STAGES=RUN/'source6-patch-stages-v1'
if STAGES.exists():raise FileExistsError(STAGES)
STAGES.mkdir()
phases=[]


def apply_phase(name,changes):
    files=[]
    for relative,new in changes.items():
        path=SOURCE/relative
        old=path.read_bytes() if path.exists() else b''
        new=new.encode('utf-8') if isinstance(new,str) else new
        if old==new:raise ValueError('Empty change: '+relative)
        for side,data in [('before',old),('after',new)]:
            target=STAGES/name/side/relative
            target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data)
        path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(new)
        if path.suffix=='.py':ast.parse(new.decode('utf-8'),filename=relative)
        files.append(dict(path=relative,before_sha256=hashlib.sha256(old).hexdigest(),
            after_sha256=hashlib.sha256(new).hexdigest(),created=not old))
    phases.append(dict(name=name,files=files))
    (STAGES/'index.json').write_text(json.dumps(dict(scope='Working source6, version stillperf4, not material acceptance',phases=phases),indent=2)+'\n')


def read(name):return (SOURCE/name).read_text(encoding='utf-8')


snapshot=RUN/'candidate5-working-source-v1'
index=json.loads((snapshot/'index.json').read_text())
changes={}
for row in index['files']:
    assert hashlib.sha256((SOURCE/row['path']).read_bytes()).hexdigest()==row['base_sha256']
    value=(snapshot/row['path']).read_bytes()
    assert hashlib.sha256(value).hexdigest()==row['sha256']
    changes[row['path']]=value
apply_phase('36-owner-metadata-and-dynamic-quota',changes)

name='python/freetoken/moe/fused_q2_q8.py'
s=read(name)
lut=(RUN/'q2_lut_variant_v1.py').read_text(encoding='utf-8')
lut=lut[lut.index('@triton.jit'):lut.index('\ndef project')]
lut=lut.replace('signed_codes','_signed_q2_codes').replace('_routed_lut','_routed_vector')
a=s.index('@triton.jit\ndef _routed_vector');b=s.index('\ndef _routed_project',a)
s=s[:a]+lut+'\n'+s[b:]
s=s.replace('scales,bank,ids,result','scales,bank.view(torch.int16),ids,result',1)
t=(RUN/'test-q2-lut-exhaustive-v1.py').read_text(encoding='utf-8')
t=t.replace('from q2_lut_variant_v1 import signed_codes','import pytest\nfrom freetoken.moe.fused_q2_q8 import _signed_q2_codes as signed_codes')
t=t.replace('def main():',"@pytest.mark.skipif(not torch.cuda.is_available(),reason='requires CUDA')\ndef test_all_q2_codes_decode_exactly():")
a=t.index("    out=Path(__file__)");b=t.index('    values=torch.arange',a);t=t[:a]+t[b:]
t=t[:t.index('    result=dict')]
apply_phase('37-signed-q2-lut',{name:s,'tests/moe/test_q2_lut.py':t})

name='python/freetoken/moe/q2_adaptive.py'
s=read(name)
kernel=(RUN/'q2_cached_exchange_kernel_v1.py').read_text(encoding='utf-8')
kernel=kernel[kernel.index('@triton.jit'):].replace('def exchange(','def _exchange(')
a=s.index('@triton.jit\ndef _exchange');b=s.index('@triton.jit\ndef _publish_exchange',a)
s=s[:a]+kernel+'\n\n'+s[b:]
s=s.replace('        self.staging_bytes=0','        self.cached_copy_counts=torch.zeros(2,device=cache.device,dtype=torch.int64)\n        self.staging_bytes=0',1)
s=s.replace('c._copy_feat_bytes,self.gpu_slots,self.count,self.max_swaps,32,1024,num_warps=8)',
    'c._copy_feat_bytes,self.gpu_slots,self.count,self.commands,c.slot_for_id.view(-1),\n            self.cached_copy_counts,self.max_swaps,32,1024,p.experts,p.protected_begin,num_warps=8)',1)
apply_phase('38-cached-incoming-promotion',{name:s})

cpp=(RUN/'row_store_cache_proto1.cpp').read_text(encoding='utf-8').replace('CachedRowStore','RowStore').replace('CachedPleStore','PleStore')
cfg=read('python/freetoken/engine/config.py')
cfg=cfg.replace('    ple_backend: str = "disk"','    ple_backend: str = "disk"\n    ple_row_cache_rows: int = 0',1)
cfg=cfg.replace('    def __post_init__(self):','''    def __post_init__(self):
        if not 0 <= self.ple_row_cache_rows <= (1 << 24) or 0 < self.ple_row_cache_rows < 8:
            raise ValueError('PLE row cache needs zero or8..16777216 rows')
        if self.ple_row_cache_rows and self.ple_backend != 'disk':
            raise ValueError('PLE row caching requires the disk backend')''',1)
table=read('python/freetoken/models/qwen4_exp/ple_disk.py')
table=table.replace('        max_graph_rows: int = 256,','        max_graph_rows: int = 256,\n        row_cache_rows: int = 0,',1)
table=table.replace('        self._device = torch.device("cuda", torch.cuda.current_device())',
    '        if row_cache_rows:\n            self._store.set_row_cache(row_cache_rows)\n        self._device = torch.device("cuda", torch.cuda.current_device())',1)
model=read('python/freetoken/models/qwen4_exp/model.py')
model=model.replace('                max_graph_rows=max(256, engine_config.cuda_graph_max_bs or 0),',
    '                max_graph_rows=max(256, engine_config.cuda_graph_max_bs or 0),\n                row_cache_rows=engine_config.ple_row_cache_rows,',1)
test=(RUN/'test-ple-row-cache-v2.py').read_text(encoding='utf-8')
test=test.replace('from ple_row_cache_variant_v2 import module',"import pytest\n_row_store=pytest.importorskip('freetoken.kernel._row_store')")
test=test.replace('def main():',"@pytest.mark.parametrize('cols',[90,160,264])\ndef test_cached_rows_preserve_payloads_and_storage_boundaries(cols):")
test=test.replace('    native=module()','    native=_row_store').replace('    rows,cols=257,90','    rows=257')
test=test.replace(",dir=r'C:\\workspace\\tmp'",'')
a=test.index("    Path('ple-row-cache-components-v2.json')")
test=test[:a]
apply_phase('39-bounded-ple-row-cache',{
    'python/freetoken/kernel/csrc/row_store/row_store_ext.cpp':cpp,
    'python/freetoken/engine/config.py':cfg,
    'python/freetoken/models/qwen4_exp/ple_disk.py':table,
    'python/freetoken/models/qwen4_exp/model.py':model,
    'tests/kernels/test_row_cache.py':test})
print(json.dumps(dict(source=str(SOURCE),phases=[r['name'] for r in phases],scope='unversioned consolidated source, not profile5 yet')))
