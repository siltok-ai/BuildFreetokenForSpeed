"""Bounded decode-repair runtime gate, distinct from Speed release acceptance."""
import hashlib
import json
import math
from pathlib import Path
from compare_execution_logits import compare

ROOT=Path(__file__).resolve().parent
def read(name):return json.loads((ROOT/name).read_text(encoding='utf-8'))
def main():
    target=ROOT/'dev4-native-runtime-gate-v1.json'
    if target.exists():raise FileExistsError(target)
    eager=read('dev4-long-matrix-v2.json');graph=read('dev4-graph-matrix-v1.json')
    a={r['label']:r for r in eager['runs']};b={r['label']:r for r in graph['runs']}
    checks=[]
    for mode,runs in [('eager',a),('graph_chunk128',b)]:
        for left,right in [('technical-cold','technical-hot'),('prefix-cold','prefix-hot'),('prefix-cold','prefix-restored')]:
            checks.append(compare(runs[left],runs[right],mode+'_'+right))
    checks.append(compare(a['long-cold'],a['long-hot'],'long_cold_hot'))
    for label in sorted(a.keys() & b.keys()):checks.append(compare(a[label],b[label],'eager_graph_chunk_'+label))
    diagnostic=read('dev4-long-matrix-v2-diagnostic.json');rows=diagnostic['rows']
    component=read('dev4-native-component-v1.json')
    wheel=ROOT/'dev4-built-native-materials-v2/wheels/freetoken-0.1.3+speed.strataq2dev4-cp312-cp312-win_amd64.whl'
    wheel_sha=hashlib.sha256(wheel.read_bytes()).hexdigest()
    gates=dict(
        actual_version=all(d['runtime_version']=='0.1.3+speed.strataq2dev4' for d in [eager,graph,]),
        wheel_identity=wheel_sha=='102615fc21180ed7ae38c69b4260b6b56158ef3253538adf9e6bbeac70556611',
        component_native_reference=component['passed'] and len(component['checks'])>=16,
        real_outputs=all(d['correctness_pass'] and d['quality_pass'] and all(d['matrix_checks'].values()) for d in [eager,graph]),
        cancel_reuse=all(d['cancel_reuse']['passed'] for d in [eager,graph]),
        sparse_long_prefix=all(eager['long_context_checks'].values()),
        execution_logits=all(c['passed'] for c in checks),
        long_component_finite_bounded=bool(rows) and all(math.isfinite(r['relative_l2']) and math.isfinite(r['max_error']) and r['relative_l2']<=2e-3 for r in rows),
        qsa_selection=bool(diagnostic['selections']) and all(r['equal'] for r in diagnostic['selections']),
    )
    report=dict(passed=all(gates.values()),gates=gates,version=eager['runtime_version'],wheel_sha256=wheel_sha,
        scope='Actual installed dev4 wheel with native AOT, all first-layer projection formats and head, real short/long sparse prefix reuse, graph/chunk execution, logits and cancellation. Allows standalone major-repair performance evaluation; not complete multi-request quality, old-model regression or Speed acceptance.',
        no_bitwise_component_gate=True,full_runtime_acceptance=False,speed_acceptance=False,
        component_count=len(component['checks']),long_component_count=len(rows),
        max_long_component_relative_l2=max(r['relative_l2'] for r in rows),qsa_slots=len(diagnostic['selections']),
        logits_checks=checks,
        evidence=[dict(path=str(ROOT/n),sha256=hashlib.sha256((ROOT/n).read_bytes()).hexdigest()) for n in [
            'dev4-long-matrix-v2.json','dev4-graph-matrix-v1.json','dev4-long-matrix-v2-diagnostic.json','dev4-native-component-v1.json']])
    target.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k not in ('logits_checks','evidence')},ensure_ascii=False))
    if not report['passed']:raise RuntimeError('Runtime gate incomplete')

if __name__=='__main__':main()
