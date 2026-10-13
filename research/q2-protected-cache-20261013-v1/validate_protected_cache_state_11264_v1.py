"""Full-model prototype state audit against the frozen installed dev5 reference."""
import hashlib
import json
import math
from pathlib import Path
from compare_execution_logits import compare

ROOT = Path(__file__).resolve().parent


def main():
    target = ROOT / 'prototype-protected-cache-state-gate-11264-v1.json'
    if target.exists():
        raise FileExistsError(target)
    candidate = json.loads((ROOT / 'prototype-protected-cache-state-11264-v1.json').read_text(encoding='utf-8'))
    previous = json.loads((ROOT / 'dev5-installed-state-v1.json').read_text(encoding='utf-8'))
    baseline = {r['label']: r for r in previous['runs']}
    checks = [compare(baseline[r['label']], r, 'previous_dev5_' + r['label']) for r in candidate['runs']]
    by_name = {r['label']: r for r in candidate['runs']}
    for left, right in [('technical-cold', 'technical-hot'), ('prefix-cold', 'prefix-hot'),
                        ('prefix-cold', 'prefix-restored'), ('long-cold', 'long-hot')]:
        checks.append(compare(by_name[left], by_name[right], right))
    diagnostic = json.loads((ROOT / 'prototype-protected-cache-state-11264-v1-diagnostic.json').read_text(encoding='utf-8'))
    resource = json.loads((ROOT / 'prototype-protected-cache-state-11264-v1-resource.json').read_text(encoding='utf-8'))
    gates = dict(model_state=candidate['passed'] and candidate['correctness_pass'],
                 matrix=all(candidate['matrix_checks'].values()),
                 sparse_context=all(candidate['long_context_checks'].values()),
                 cancel_reuse=candidate['cancel_reuse']['passed'],
                 previous_and_prefix_logits=all(c['passed'] for c in checks),
                 finite_components=bool(diagnostic['rows']) and all(
                     math.isfinite(r['relative_l2']) and math.isfinite(r['max_error']) and r['relative_l2'] <= 2e-3
                     for r in diagnostic['rows']),
                 sparse_selection=bool(diagnostic['selections']) and all(r['equal'] for r in diagnostic['selections']),
                 owned_process_exit=resource['returncode'] == 0 and not resource.get('aborted'),
                 memory_headroom=resource['min_commit_available'] >= 4 * 2**30)
    report = dict(passed=all(gates.values()), gates=gates, checks=checks,
                  components=len(diagnostic['rows']), max_component_relative_l2=max(r['relative_l2'] for r in diagnostic['rows']),
                  prototype=candidate['prototype'], min_commit_available=resource['min_commit_available'],
                  full_runtime_acceptance=False, speed_acceptance=False,
                  scope='Permits bounded prototype performance measurement only; graph/prefix/cancel/sparse-state and fixed logits against frozen dev5. No complete quality suite, rebuild, other-model or multi-request release gate.',
                  superseded_component_fixture='Combined component v1 captured after an in-place rewrite without refreshing raw IDs; corrected v2 is authoritative. Preserve v1 rather than count it as an accepted test.')
    target.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in report.items() if k != 'checks'}, indent=2))
    if not report['passed']:
        raise RuntimeError('Protected-cache state gate failed')


if __name__ == '__main__':
    main()
