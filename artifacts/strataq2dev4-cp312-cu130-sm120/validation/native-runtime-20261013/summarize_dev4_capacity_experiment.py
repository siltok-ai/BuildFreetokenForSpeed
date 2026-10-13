"""Compare identical requests across native FreeToken expert-cache capacities."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parent
def load(name):return json.loads((ROOT/name).read_text(encoding='utf-8'))
def main():
    target=ROOT/'dev4-capacity-experiment-summary-v1.json'
    if target.exists():raise FileExistsError(target)
    small=load('dev4-native-performance-1024-v1.json');large=load('dev4-native-performance-4096-v1.json')
    resource=load('dev4-native-performance-4096-v1-resource.json')
    if large['status']!='native_decode_repair_performance_measured' or resource['returncode']!=0 or not large['quality_pass']:raise RuntimeError('Incomplete experiment')
    lookup={r['label']:r for r in small['runs']};rows=[]
    for b in large['runs']:
        a=lookup[b['label']]
        if a['input_sha256']!=b['input_sha256']:raise RuntimeError('Different input')
        def metrics(r):
            return dict(decode_tps=r['decode_tps'],miss_rate=r['moe_stats']['miss_rate'],
                expert_payload_bytes_per_forward=r['expert_payload_h2d_bytes']/(r['decode_layer_calls']/48),
                cuda_allocated=r['cuda_allocated'],ttft_ms=r['ttft_ms'],output_tokens=r['output_tokens'],
                eos=r['eos'],truncated=r['truncated'],prefix_cached_tokens=r['prefix_cached_tokens'])
        rows.append(dict(label=b['label'],cache1024=metrics(a),cache4096=metrics(b),
            tps_ratio=b['decode_tps']/a['decode_tps'],
            token_sequences_equal_across_capacity=a['output']['token_ids']==b['output']['token_ids']))
    report=dict(status='capacity_benefit_measured',requests=len(rows),rows=rows,
        cold_hot_sequences_equal=all(r['cold_hot_token_ids_equal'] for r in large['runs'] if r['label'].endswith('-hot')),
        quality_pass=large['quality_pass'],peak_commit_bytes=resource['peak_commit_used'],
        min_commit_available_bytes=resource['min_commit_available'],
        scope='One repetition, four actual 512-token requests, same dev4 wheel/graph/sampling/input. Directional expert-cache capacity experiment, not a repeated stable deployment benchmark.',
        next='Prioritize reducing decode miss bytes and host/GPU duplicate residency, then dedicated expert vector compute; retain prefill timing separately.')
    target.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report))
if __name__=='__main__':main()
