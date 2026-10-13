"""Derive a diagnostic harness that excludes prefill activity collection."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
source = ROOT / 'first_freetoken_performance_dev5_v1.py'
target = ROOT / 'first_freetoken_decode_only_diagnostic_v1.py'
if target.exists():
    raise FileExistsError(target)
text = source.read_text(encoding='utf-8')
old = '    arrivals=[];terminals=[];prefills=[];timers=[];phase=[None];decode_start_stats=[]'
new = old + '\n    active_profiler=[None];decode_collection_started=[False]'
if text.count(old) != 1:
    raise RuntimeError('Unexpected harness initialization')
text = text.replace(old, new)
old = "        phase[0]='decode' if batch.is_decode else 'prefill'"
new = old + '''
        if batch.is_decode and active_profiler[0] is not None and not decode_collection_started[0]:
            torch.cuda.synchronize()
            active_profiler[0].toggle_collection_dynamic(True, [torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA])
            decode_collection_started[0]=True'''
if text.count(old) != 1:
    raise RuntimeError('Unexpected forward hook')
text = text.replace(old, new)
old = '''                        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:
                            item=run(case,label,min(a.tokens,32),not hot)
                        prof.export_chrome_trace(str(out.with_suffix('.'+label+'.trace.json')))'''
new = '''                        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA],post_processing_timeout_s=60) as prof:
                            active_profiler[0]=prof;decode_collection_started[0]=False
                            prof.toggle_collection_dynamic(False, [torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA])
                            item=run(case,label,min(a.tokens,32),not hot)
                        active_profiler[0]=None
                        prof.export_chrome_trace(str(out.with_suffix('.'+label+'.trace.json')))'''
if text.count(old) != 1:
    raise RuntimeError('Unexpected profiling context')
text = text.replace(old, new)
target.write_text(text, encoding='utf-8')
manifest = dict(source=str(source), source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                target=str(target), target_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
                changes=['Disable CPU/CUDA activity during prefill; synchronize and enable before first decode forward',
                         'Bound Python post-processing to 60 seconds; preserve CUDA graph/markers/counters'],
                aborted_previous_diagnostic=dict(path='prototype-protected-cache-diagnostic-8192-v1-resource.json',
                    reason='Full 4K prefill activity post-processing exceeded three minutes while GPU idle; stopped verified owned PID 22768 after 306-second total diagnostic. Two short traces preserved; no completed four-trace gate.'),
                throughput_acceptance=False)
(ROOT / 'decode-only-profiler-derivation-v1.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
print(json.dumps(manifest, indent=2))
