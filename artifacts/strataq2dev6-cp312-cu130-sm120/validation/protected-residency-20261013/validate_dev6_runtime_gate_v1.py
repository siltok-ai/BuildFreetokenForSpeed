"""Bind fixed-state and declared quality gates before measuring the installed revision."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
target = ROOT/'dev6-installed-runtime-gate-v1.json'
if target.exists():
    raise FileExistsError(target)
state = json.loads((ROOT/'dev6-installed-state-gate-v1.json').read_text())
actual = json.loads((ROOT/'quality-gate-dev6/freetoken.json').read_text(encoding='utf-8'))
previous = json.loads((ROOT/'quality-gate-dev5/freetoken.json').read_text(encoding='utf-8'))
strata = json.loads((ROOT/'quality-gate-dev3/strata.json').read_text(encoding='utf-8'))
resource = json.loads((ROOT/'dev6-quality-v1-resource.json').read_text())
base = {r['name']:r for r in previous['runs']}
reference = {r['name']:r for r in strata['runs']}
checks = [dict(name=r['name'], input_identity=r['input_sha256']==base[r['name']]['input_sha256'],
    token_ids_equal=r['output']['token_ids']==base[r['name']]['output']['token_ids'],
    no_previous_regression=r['quality_pass'] or not base[r['name']]['quality_pass'],
    no_strata_only_regression=r['quality_pass'] or not reference[r['name']]['quality_pass'])
    for r in actual['runs']]
gates = dict(state=state['passed'], same_installed_version=actual['version']=='0.1.3+speed.strataq2dev6',
    complete_quality=len(checks)==20 and len({r['name'] for r in actual['runs']})==20,
    frozen_inputs=all(c['input_identity'] for c in checks),
    previous_outputs=all(c['token_ids_equal'] for c in checks),
    previous_quality=all(c['no_previous_regression'] for c in checks),
    strata_quality=all(c['no_strata_only_regression'] for c in checks),
    quality_exit=resource['returncode']==0 and not resource.get('aborted'),
    memory_headroom=resource['min_commit_available']>=4*2**30)
report = dict(passed=all(gates.values()), gates=gates, checks=checks,
    wheel_sha256=state['wheel_sha256'], quality_score=actual['score'], quality_total=actual['total'],
    previous_score=previous['score'], strata_score=strata['score'],
    state_gate_sha256=hashlib.sha256((ROOT/'dev6-installed-state-gate-v1.json').read_bytes()).hexdigest(),
    full_runtime_acceptance=False, speed_acceptance=False,
    scope='Allows standalone installed-wheel performance measurements; no Speed/package/other-model acceptance')
target.write_text(json.dumps(report, indent=2), encoding='utf-8')
print(json.dumps({k:v for k,v in report.items() if k!='checks'}, indent=2))
if not report['passed']:
    raise RuntimeError('Installed runtime gate failed')
