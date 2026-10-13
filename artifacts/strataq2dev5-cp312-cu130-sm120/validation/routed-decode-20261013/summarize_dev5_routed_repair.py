"""Installed-wheel throughput, resource and trace evidence with explicit limits."""
import hashlib
import json
from pathlib import Path
from statistics import median
ROOT = Path(__file__).resolve().parent
def read(name):
    return json.loads((ROOT / name).read_text(encoding='utf-8'))
def summary(rows):
    return dict(requests=len(rows), actual_output_tokens=[r['output_tokens'] for r in rows],
        median_decode_tps=median(r['decode_tps'] for r in rows),
        range_decode_tps=[min(r['decode_tps'] for r in rows), max(r['decode_tps'] for r in rows)],
        median_ttft_ms=median(r['ttft_ms'] for r in rows),
        median_miss_rate=median(r['moe_stats']['miss_rate'] for r in rows),
        median_expert_payload_bytes_per_forward=median(r['expert_payload_h2d_bytes']/(r['decode_layer_calls']/48) for r in rows),
        peak_cuda_allocated=max(r['cuda_allocated'] for r in rows),
        eos_requests=sum(r['eos'] for r in rows), truncated_requests=sum(r['truncated'] for r in rows))
def main():
    target = ROOT / 'dev5-routed-decode-repair-summary-v1.json'
    if target.exists():
        raise FileExistsError(target)
    capacities = {}
    for slots, revision in [(4096,'v1'),(6144,'v2')]:
        prefix = f'dev5-native-performance-{slots}-{revision}'
        perf = read(prefix+'.json')
        resource = read(prefix+'-resource.json')
        if (len(perf['runs']) != 12 or resource['returncode'] != 0 or not perf['quality_pass']
                or not perf['all_measurements_sufficient'] or perf['version'] != '0.1.3+speed.strataq2dev5'
                or not perf['native_engine_only'] or perf['process_engine_override']):
            raise RuntimeError('Incomplete installed performance measurement')
        groups = {}
        for context in ['short','diverse4096']:
            selected = [r for r in perf['runs'] if r['label'].startswith(context+'-')]
            groups[context] = summary(selected)
            for temp in ['cold','hot']:
                groups[context+'_'+temp] = summary([r for r in selected if r['label'].endswith('-'+temp)])
        capacities[str(slots)] = dict(groups=groups,
            cold_hot_trajectory_checks=[dict(label=r['label'],equal=r['cold_hot_token_ids_equal'],
                first_difference=r['cold_hot_first_difference'],length_equal=r['cold_hot_length_equal'])
                for r in perf['runs'] if r['label'].endswith('-hot')],
            peak_commit_bytes=resource['peak_commit_used'],min_commit_available_bytes=resource['min_commit_available'],
            no_mtp=perf['no_mtp'],source=perf['source'],quality_smoke_pass=perf['quality_pass'],
            report=str(ROOT/(prefix+'.json')),report_sha256=hashlib.sha256((ROOT/(prefix+'.json')).read_bytes()).hexdigest())
    a,b=read('dev5-native-performance-4096-v1.json'),read('dev5-native-performance-6144-v2.json')
    by_label={r['label']:r for r in a['runs']}
    capacity_identity=[]
    for row in b['runs']:
        small=by_label[row['label']]
        if small['input_sha256']!=row['input_sha256']:
            raise RuntimeError('Capacity input protocol differs')
        capacity_identity.append(dict(label=row['label'],input_equal=True,
            full_token_ids_equal=small['output']['token_ids']==row['output']['token_ids']))
    gate=read('dev5-routed-runtime-gate-v1.json')
    trace=read('dev5-routed-trace-analysis-v1.json')
    report=dict(status='installed_major_expert_decode_repair_measured',version=gate['version'],wheel_sha256=gate['wheel_sha256'],
        no_mtp=True,native_engine_only=True,process_engine_override=False,
        bounded_runtime_gate=gate['passed'],quality_20_case=gate['quality'],capacities=capacities,
        capacity_input_and_output_identity=capacity_identity,trace_family_summary=[dict(path=r['path'],families=r['family_ms_per_decode'],
            gpu_sum_ms=r['gpu_kernel_ms_per_decode'],gpu_union_ms=r['gpu_kernel_union_ms_per_decode']) for r in trace['rows']],
        stable_100_tps_across_short_and_4k=False,full_runtime_acceptance=False,speed_acceptance=False,
        limitations=['All measured long outputs reach the 512-token cap; these are not natural EOS measurements',
            'Short cold/hot trajectories diverge near token 104-105; bounded logits/state checks and 20-case quality have passed, but exact trajectory equality is not claimed',
            '6144 slots leave only about 2.1 GiB system commit headroom; not a recommended default or a multi-request resource acceptance',
            'Frozen installed GPU suite remains 18 passed / 1 legacy bitwise failure; separately passed bounded precision contract. Update its contract in a new revision before release',
            'Prefill remains slower than historical mixed implementation; measured separately and retained',
            'Strata 100+ reference uses different residency/CPU-GPU execution organization and substantially greater automatic residency'],
        priorities=['Decode first: reduce expert miss/copy through joint RAM/VRAM residency planning and a compact host complement, using Strata design as reference',
            'Trace actual CPU/GPU dependencies, native Strata GPU stages and copy/compute overlap; do not add overlapping waits to GPU durations',
            'Retain prefill correctness/resource/regression checks; use batched compressed MMQ as a later substantive improvement',
            'Evaluate MTP after current residency regressions; then Speed adaptation, old-model and packaged acceptance'])
    target.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(dict(status=report['status'],quality=report['quality_20_case'],capacities={k:v['groups'] for k,v in capacities.items()},stable_100=False)))
if __name__ == '__main__':
    main()
