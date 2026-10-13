"""Bounded routed-expert runtime gate, preserving the obsolete bitwise failure."""
import hashlib
import json
import math
from pathlib import Path
from compare_execution_logits import compare

ROOT = Path(__file__).resolve().parent
def read(name):
    return json.loads((ROOT / name).read_text(encoding='utf-8'))
def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    target = ROOT / 'dev5-routed-runtime-gate-v1.json'
    if target.exists():
        raise FileExistsError(target)
    model = read('dev5-installed-state-v1.json')
    runs = {r['label']: r for r in model['runs']}
    checks = [compare(runs[a], runs[b], b) for a, b in [
        ('technical-cold', 'technical-hot'), ('prefix-cold', 'prefix-hot'),
        ('prefix-cold', 'prefix-restored'), ('long-cold', 'long-hot')]]
    diagnostic = read('dev5-installed-state-v1-diagnostic.json')
    precision = read('dev5-precision-contract-v1.json')
    candidate = read('quality-gate-dev5/freetoken.json')
    previous = read('quality-gate-dev3/freetoken.json')
    strata = read('quality-gate-dev3/strata.json')
    by_name = lambda d: {r['name']: r for r in d['runs']}
    cq, pq, sq = map(by_name, [candidate, previous, strata])
    candidate_only_failures = [n for n in cq if not cq[n]['quality_pass'] and sq[n]['quality_pass']]
    wheel = ROOT / 'dev5-derived-native-base-v1/wheels/freetoken-0.1.3+speed.strataq2dev5-cp312-cp312-win_amd64.whl'
    identity = json.loads((ROOT / 'dev5-native-base-identity-v1.log').read_text(encoding='utf-8').strip().splitlines()[-1])
    gpu_log = (ROOT / 'dev5-installed-gpu-tests-v1-resource.stdout.log').read_text(encoding='utf-8')
    gates = dict(
        actual_version=model['runtime_version'] == '0.1.3+speed.strataq2dev5',
        wheel_identity=sha(wheel) == 'cb8a235b0eb2845361dbf93a8f296d5ab6bada9a11c1a02a5a7849300db4bf12',
        native_base_identity=identity['source_replay_verified'] and identity['wheel_and_binary_identity_verified'] and identity['licensed_native_base_verified'] and identity['version'] == model['runtime_version'],
        independent_integer_reference_and_graph_tests='18 passed' in gpu_log and '1 failed' in gpu_log,
        bounded_expert_precision=precision['passed'],
        actual_outputs=model['passed'] and model['correctness_pass'] and all(model['matrix_checks'].values()),
        sparse_long_prefix=all(model['long_context_checks'].values()),
        cancel_reuse=model['cancel_reuse']['passed'],
        execution_logits=all(c['passed'] for c in checks),
        long_component_finite_bounded=bool(diagnostic['rows']) and all(
            math.isfinite(r['relative_l2']) and math.isfinite(r['max_error']) and r['relative_l2'] <= 2e-3
            for r in diagnostic['rows']),
        qsa_selection=bool(diagnostic['selections']) and all(r['equal'] for r in diagnostic['selections']),
        quality_protocol_identity=sha(ROOT / 'quality-gate-dev5/protocol.json') == sha(ROOT / 'quality-gate-dev3/protocol.json')
            and cq.keys() == pq.keys() == sq.keys()
            and all(cq[n]['input_sha256'] == pq[n]['input_sha256'] == sq[n]['input_sha256'] for n in cq),
        no_new_quality_failures=not candidate_only_failures and candidate['score'] >= previous['score'],
    )
    evidence_names = ['dev5-installed-state-v1.json', 'dev5-installed-state-v1-diagnostic.json',
        'dev5-precision-contract-v1.json', 'dev5-native-base-identity-v1.log',
        'dev5-installed-gpu-tests-v1-resource.stdout.log', 'quality-gate-dev5/freetoken.json',
        'quality-gate-dev5/protocol.json', 'quality-gate-dev3/strata.json']
    report = dict(passed=all(gates.values()), gates=gates, version=model['runtime_version'],
        wheel_sha256=sha(wheel), scope='Actual installed dev5 routed-expert wheel: bounded component/FP32 agreement, real graph/prefix/cancellation, sparse long context and fixed 20-case quality. Permits standalone decode measurement, not full release or Speed acceptance.',
        native_engine_only=True, process_engine_override=False, no_bitwise_component_gate=True,
        full_runtime_acceptance=False, speed_acceptance=False,
        legacy_gpu_suite=precision['legacy_suite'],
        required_before_release='Update the obsolete bitwise test contract in a NEW frozen revision; preserve this failed test and all dev5 bytes.',
        quality=dict(candidate=candidate['score'], previous=previous['score'], strata=strata['score'], total=candidate['total'], candidate_only_failures=candidate_only_failures),
        long_component_count=len(diagnostic['rows']), qsa_slots=len(diagnostic['selections']),
        max_long_component_relative_l2=max(r['relative_l2'] for r in diagnostic['rows']),
        logits_checks=checks,
        evidence=[dict(path=str(ROOT / n), sha256=sha(ROOT / n)) for n in evidence_names])
    target.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k not in ('logits_checks', 'evidence')}, ensure_ascii=False))
    if not report['passed']:
        raise RuntimeError('Runtime gate incomplete')

if __name__ == '__main__':
    main()
