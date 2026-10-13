"""Audit the installed optional cache after a real KV rebuild, with frozen dev5 logits."""
import hashlib
import json
import math
from pathlib import Path
from compare_execution_logits import compare

ROOT = Path(__file__).resolve().parent
target = ROOT/'dev6-installed-state-gate-v1.json'
if target.exists():
    raise FileExistsError(target)
candidate = json.loads((ROOT/'dev6-installed-state-v2.json').read_text(encoding='utf-8'))
previous = json.loads((ROOT/'dev5-installed-state-v1.json').read_text(encoding='utf-8'))
baseline = {r['label']: r for r in previous['runs']}
checks = [compare(baseline[r['label']], r, 'previous_dev5_'+r['label']) for r in candidate['runs']]
by_name = {r['label']: r for r in candidate['runs']}
for left, right in [('technical-cold','technical-hot'), ('prefix-cold','prefix-hot'),
                    ('prefix-cold','prefix-restored'), ('long-cold','long-hot')]:
    checks.append(compare(by_name[left], by_name[right], right))
diagnostic = json.loads((ROOT/'dev6-installed-state-v2-diagnostic.json').read_text())
resource = json.loads((ROOT/'dev6-installed-state-v2-resource.json').read_text())
identity = json.loads((ROOT/'dev6-installed-identity-v1.json').read_text())
components = json.loads((ROOT/'dev6-installed-component-v1-resource.json').read_text())
gates = dict(model_state=candidate['passed'] and candidate['correctness_pass'],
    installed_identity=identity['passed'] and identity['version']=='0.1.3+speed.strataq2dev6',
    component_tests=components['returncode']==0,
    native_configuration=not candidate['process_engine_override'] and candidate['native_engine_only'],
    matrix=all(candidate['matrix_checks'].values()), sparse_context=all(candidate['long_context_checks'].values()),
    cancel_reuse=candidate['cancel_reuse']['passed'], previous_and_prefix_logits=all(c['passed'] for c in checks),
    finite_components=bool(diagnostic['rows']) and all(math.isfinite(r['relative_l2']) and math.isfinite(r['max_error'])
        and r['relative_l2']<=2e-3 for r in diagnostic['rows']),
    sparse_selection=bool(diagnostic['selections']) and all(r['equal'] for r in diagnostic['selections']),
    rebuild=all(v for k,v in candidate['rebuild_checks'].items() if k!='fixed_capacity'),
    owned_process_exit=resource['returncode']==0 and not resource.get('aborted'),
    memory_headroom=resource['min_commit_available']>=4*2**30)
report = dict(passed=all(gates.values()), gates=gates, checks=checks, components=len(diagnostic['rows']),
    max_component_relative_l2=max(r['relative_l2'] for r in diagnostic['rows']),
    sparse_selections=len(diagnostic['selections']), wheel_sha256=identity['wheel_sha256'],
    state_source_sha256=hashlib.sha256((ROOT/'dev6_installed_state_v2.py').read_bytes()).hexdigest(),
    min_commit_available=resource['min_commit_available'], rebuild_checks=candidate['rebuild_checks'],
    full_runtime_acceptance=False, speed_acceptance=False,
    superseded_v1='memory_ratio0.8 could load explicit capacity but existing runtime budget rejected same-capacity KV rebuild recoverably; v2 uses0.9, no engine modification')
target.write_text(json.dumps(report, indent=2), encoding='utf-8')
print(json.dumps({k:v for k,v in report.items() if k!='checks'}, indent=2))
if not report['passed']:
    raise RuntimeError('Installed cache state gate failed')
