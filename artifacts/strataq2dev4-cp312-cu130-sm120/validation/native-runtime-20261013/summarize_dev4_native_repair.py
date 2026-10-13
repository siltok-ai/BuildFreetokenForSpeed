"""Summarize complete measured native revision without reclassifying prototypes."""
import hashlib
import json
from pathlib import Path
import statistics
from analyze_freetoken_decode_traces import analyze
ROOT=Path(__file__).resolve().parent
def read(name):return json.loads((ROOT/name).read_text(encoding='utf-8'))
def summarize(rows):
    return dict(requests=len(rows),actual_output_tokens=[r['output_tokens'] for r in rows],
        median_decode_tps=statistics.median(r['decode_tps'] for r in rows),
        range_decode_tps=[min(r['decode_tps'] for r in rows),max(r['decode_tps'] for r in rows)],
        median_ttft_ms=statistics.median(r['ttft_ms'] for r in rows),
        median_miss_rate=statistics.median(r['moe_stats']['miss_rate'] for r in rows),
        median_expert_payload_bytes_per_forward=statistics.median(r['expert_payload_h2d_bytes']/(r['decode_layer_calls']/48) for r in rows),
        peak_cuda_allocated=max(r['cuda_allocated'] for r in rows))
def main():
    target=ROOT/'dev4-native-decode-repair-summary-v1.json'
    if target.exists():raise FileExistsError(target)
    perf=read('dev4-native-performance-1024-v1.json');resource=read('dev4-native-performance-1024-v1-resource.json')
    if perf['status']!='native_decode_repair_performance_measured' or resource['returncode']!=0:raise RuntimeError('Incomplete measurement')
    if len(perf['runs'])!=12 or not perf['quality_pass'] or not perf['all_measurements_sufficient']:raise RuntimeError('Measurement gate incomplete')
    traces=[analyze(p) for p in sorted(ROOT.glob('dev4-native-diagnostic-1024-v1.*.trace.json'))]
    if not traces:raise RuntimeError('Native trace not available')
    groups={}
    for context in ['short','diverse4096']:
        for temp in ['cold','hot']:
            groups[context+'_'+temp]=summarize([r for r in perf['runs'] if r['label'].startswith(context+'-') and r['label'].endswith('-'+temp)])
        groups[context]=summarize([r for r in perf['runs'] if r['label'].startswith(context+'-')])
    old=read('first-performance-dev3-1024-v2.json')
    comparisons={}
    for context in ['short','diverse4096']:
        oldrows=[r for r in old['runs'] if r['label'].startswith(context+'-')]
        oldcold=[r for r in oldrows if r['label'].endswith('-cold')]
        previous=statistics.median(r['decode_tps'] for r in oldrows)
        comparisons[context]=dict(dev3_median_decode_tps=previous,dev4_median_decode_tps=groups[context]['median_decode_tps'],
            decode_speedup=groups[context]['median_decode_tps']/previous,
            dev3_cold_median_ttft_ms=statistics.median(r['ttft_ms'] for r in oldcold),
            dev4_cold_median_ttft_ms=groups[context+'_cold']['median_ttft_ms'])
    report=dict(status='native_major_decode_repair_measured',version=perf['version'],wheel_sha256=read('dev4-native-runtime-gate-v1.json')['wheel_sha256'],
        native_engine_only=True,no_mtp=True,expert_cache_slots=1024,groups=groups,same_protocol_dev3_comparison=comparisons,
        cold_hot_full_sequences_equal=all(r['cold_hot_token_ids_equal'] for r in perf['runs'] if r['label'].endswith('-hot')),
        eos_requests=sum(r['eos'] for r in perf['runs']),truncated_requests=sum(r['truncated'] for r in perf['runs']),
        actual_output_tokens=sum(r['output_tokens'] for r in perf['runs']),peak_commit_bytes=resource['peak_commit_used'],
        min_commit_available_bytes=resource['min_commit_available'],traces=traces,
        limitations=['Bounded native component/model gate; full old-model/multi-request/Speed acceptance remains open',
            '512-token capped outputs, not time to natural EOS','Strata auto resident reference uses a much larger expert cache; not a cache-matched comparison',
            'Prefill remains a known slower path; token-wise MMVQ preserves shared arithmetic but does not exploit matrix batch weight reuse'],
        priorities=['Decode expert residence and miss payload with Windows commit-aware planning',
            'Dedicated routed Q2 x Q8 expert vector kernel instead of padding single-token routes into 16-token tensor-core tiles',
            'Check copy/compute overlap and graph lifecycle with trace; avoid adding CPU waits to GPU time',
            'Keep prefill correctness/regression evidence; improve efficient batch computation after dominant decode issues',
            'MTP if needed after native decode repair, with proper state rollback; Speed integration and packaged acceptance afterward'],
        full_runtime_acceptance=False,speed_acceptance=False)
    target.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k!='traces'},ensure_ascii=False))
    for r in traces:print(json.dumps(dict(path=Path(r['path']).name,gpu_ms=r['gpu_kernel_ms_per_decode'],top=r['kernels'][:5]),ensure_ascii=False))
if __name__=='__main__':main()
