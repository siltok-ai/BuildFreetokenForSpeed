"""Create isolated installed-wheel probes, retaining frozen inputs and references."""
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parent
names = ['test_dev6_installed_component_v1.py', 'audit_dev6_installed_identity_v1.py',
    'dev6_installed_state_v2.py', 'dev6_installed_performance_v2.py',
    'validate_dev6_installed_state_v1.py', 'validate_dev6_runtime_gate_v1.py',
    'run_quality_gate_dev6_v1.py']
for name in names:
    target = ROOT / name.replace('dev6', 'dev7')
    if target.exists():
        raise FileExistsError(target)
    text = (ROOT / name).read_text(encoding='utf-8').replace('dev6', 'dev7')
    if name == 'dev6_installed_state_v2.py':
        anchor = '    pointers = {role: value.data_ptr() for role, value in cache.bank_caches.items()}'
        insert = '''    sparse_batches = []
    sparse_forward = llm.engine.forward_batch
    def observe_sparse(batch, sampling):
        if not batch.is_decode:
            sparse_batches.append(sum(r.device_len - r.cached_len for r in batch.reqs))
        return sparse_forward(batch, sampling)
    llm.engine.forward_batch = observe_sparse
    sparse_ids = [int(t) for t in (ROOT/'strata-reference-quality.tokens').read_text().split(',')][:16]
    sparse_output = llm.generate([sparse_ids], SamplingParams(temperature=0, top_p=1, max_tokens=2))[0]
    llm.engine.forward_batch = sparse_forward
    if 16 not in sparse_batches or len(sparse_output['token_ids']) != 2:
        raise RuntimeError('Real 16-token sparse prefill was not exercised')
    result['sparse_prefill_16'] = dict(passed=True, prefill_tokens=sparse_batches,
        input_ids=sparse_ids, output=sparse_output, routed_top_k=10)
'''
        if anchor not in text:
            raise RuntimeError('Missing state insertion point')
        text = text.replace(anchor, insert + anchor)
    if name == 'validate_dev6_installed_state_v1.py':
        text = text.replace('gates = dict(model_state=',
            "gates = dict(sparse_prefill_16=candidate['sparse_prefill_16']['passed'], model_state=")
        text = text.replace("superseded_v1='memory_ratio0.8 could load explicit capacity but existing runtime budget rejected same-capacity KV rebuild recoverably; v2 uses0.9, no engine modification'",
            "scope='dev7 adds real 16-token sparse prefill; otherwise frozen dev6 protocol'")
    target.write_text(text, encoding='utf-8')
quality = ROOT / 'quality-gate-dev7'
quality.mkdir()
shutil.copy2(ROOT/'quality-gate-dev6/protocol.json', quality/'protocol.json')
print('Created isolated dev7 installed probes and unchanged quality protocol')
