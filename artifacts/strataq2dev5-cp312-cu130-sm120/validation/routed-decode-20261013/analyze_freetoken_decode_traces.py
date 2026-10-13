"""Summarize realized decode GPU work. CPU waits are not added to GPU durations."""
import bisect
import collections
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parent

def analyze(path):
    events = json.loads(path.read_text(encoding='utf-8'))['traceEvents']
    markers = [e for e in events if e.get('cat')=='user_annotation' and e.get('name')=='probe.forward.decode']
    if not markers:
        raise RuntimeError('No realized decode markers: '+str(path))
    start = markers[0]['ts']
    device_ranges = sorted([e for e in events if e.get('cat')=='gpu_user_annotation' and e.get('name')=='probe.forward.decode'],key=lambda e:e['ts'])
    if len(device_ranges)!=len(markers):
        raise RuntimeError('CPU/GPU decode marker counts differ: '+str(path))
    range_starts = [e['ts'] for e in device_ranges]
    def device_decode(e):
        i = bisect.bisect_right(range_starts,e['ts'])-1
        return i>=0 and e['ts']+e.get('dur',0)<=device_ranges[i]['ts']+device_ranges[i]['dur']+1
    kernels = [e for e in events if e.get('cat')=='kernel' and device_decode(e)]
    totals = collections.defaultdict(lambda: [0.0, 0])
    grids = collections.defaultdict(lambda: [0.0, 0])
    runtime = collections.defaultdict(lambda: [0.0, 0])
    streams = collections.Counter()
    for e in kernels:
        totals[e['name']][0] += e['dur']; totals[e['name']][1] += 1
        streams[e['args'].get('stream')] += 1
        if e['name']=='_gemm':
            key = (tuple(e['args'].get('grid', [])), e['args'].get('registers per thread'), e['args'].get('shared memory'))
            grids[key][0] += e['dur']; grids[key][1] += 1
    for e in events:
        if e.get('cat') in ('cuda_runtime','cuda_driver','gpu_memcpy') and e.get('ts',0)>=start:
            runtime[e['name']][0] += e.get('dur',0); runtime[e['name']][1] += 1
    def rows(table):
        return [dict(name=name, gpu_or_cpu_ms_per_decode=dur/1000/len(markers), calls_per_decode=n/len(markers))
                for name,(dur,n) in sorted(table.items(),key=lambda x:x[1][0],reverse=True)]
    annotations = sorted([e for e in events if e.get('cat')=='gpu_user_annotation' and e.get('name','').startswith('gguf.linear|')],key=lambda e:e['ts'])
    families = collections.defaultdict(lambda: [0.0, 0])
    unknown = 0
    if annotations:
        starts = [e['ts'] for e in annotations]
        for e in kernels:
            if e['name']!='_gemm':
                continue
            i = bisect.bisect_right(starts,e['ts'])-1
            if i<0 or e['ts']+e['dur']>annotations[i]['ts']+annotations[i]['dur']+1:
                unknown += 1; continue
            name = re.sub(r'blk\.\d+\.', '',annotations[i]['name'].replace('gguf.linear|',''))
            families[name][0] += e['dur']; families[name][1] += 1
    intervals = sorted((e['ts'],e['ts']+e['dur']) for e in kernels)
    busy = 0.0
    left,right = intervals[0]
    for a,b in intervals[1:]:
        if a<=right:
            right=max(right,b)
        else:
            busy+=right-left;left,right=a,b
    busy+=right-left
    return dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                decode_forwards=len(markers),gpu_kernel_ms_per_decode=sum(e['dur'] for e in kernels)/1000/len(markers),
                gpu_kernel_union_ms_per_decode=busy/1000/len(markers),gpu_stream_kernel_counts=dict(streams),
                kernels=rows(totals),runtime_and_memcpy=rows(runtime),
                gemm_grids=[dict(grid=k[0],registers=k[1],shared_bytes=k[2],ms_per_decode=v[0]/1000/len(markers),calls_per_decode=v[1]/len(markers)) for k,v in sorted(grids.items(),key=lambda x:x[1][0],reverse=True)],
                labelled_projection_families=rows(families),unattributed_labelled_gemm_calls=unknown,
                scope='Diagnostic GPU kernel durations contained in GPU decode-forward annotation ranges; pending prefill and postprocessing excluded. CPU/GPU marker counts verified; only CPU count is denominator. GPU duration sums and union reported separately. CPU blocking waits overlap GPU work and are not an extra additive cost. This is not a long-output throughput benchmark.')

def main():
    out = ROOT/'first-decode-structural-trace-analysis-v2.json'
    if out.exists():
        raise FileExistsError(out)
    patterns = ['dev3-first-diagnostic-profile.*.trace.json','q2opt2-first-diagnostic-profile.*.trace.json',
                'dev3-labelled-eager-profile.*.trace.json','dev3-historical-mixed-diagnostic-profile.*.trace.json']
    result = dict(status='observed_structural_analysis',runs=[],
                  historical_q2opt2_job_limit='Frozen q2opt2 profile produced four traces but the job failed its 4K cold/hot output-equality assertion. Its completed short traces remain valid measurements of the historical execution path; do not claim the entire job passed correctness.')
    for pattern in patterns:
        for path in sorted(ROOT.glob(pattern)):
            row = analyze(path);result['runs'].append(row)
            print(json.dumps(dict(path=path.name,decode_forwards=row['decode_forwards'],gpu_ms_per_decode=row['gpu_kernel_ms_per_decode'],top=row['kernels'][:3]),ensure_ascii=False))
    out.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')

if __name__=='__main__':
    main()
