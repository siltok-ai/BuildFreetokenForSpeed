"""Compare actual native diagnostics without adding CPU waits to GPU work."""
import hashlib
import json
from pathlib import Path
import re
from analyze_freetoken_decode_traces import analyze
ROOT = Path(__file__).resolve().parent
def read(name):
    return json.loads((ROOT / name).read_text(encoding='utf-8'))
def main():
    target = ROOT / 'dev5-capacity-and-strata-stage-analysis-v1.json'
    if target.exists():
        raise FileExistsError(target)
    small=read('dev5-routed-trace-analysis-v1.json')['rows']
    large=[]
    for path in sorted(ROOT.glob('dev5-native-diagnostic-6144-v1.short-*.trace.json')):
        row=analyze(path)
        families={k:0. for k in ['gguf_main','expert_routed','expert_alignment','expert_copy','lru','other']}
        for kernel in row['kernels']:
            name=kernel['name']
            if name=='_bf16_rows' or 'gguf_native' in name:key='gguf_main'
            elif name=='_routed_vector':key='expert_routed'
            elif name.startswith('_moe_align'):key='expert_alignment'
            elif 'fast_index_copy_multi' in name:key='expert_copy'
            elif 'lru_ensure' in name:key='lru'
            else:key='other'
            families[key]+=kernel['gpu_or_cpu_ms_per_decode']
        row['family_ms_per_decode']=families
        large.append(row)
    if len(large)!=2 or read('dev5-native-diagnostic-6144-v1-resource.json')['returncode']!=0:
        raise RuntimeError('6144 short diagnostic incomplete')
    reference=read('strata-native-stage-diagnostic-6144-v1.json')
    if reference['status']!='diagnostic_gpu_stages_complete' or not reference['quality_pass'] or not reference['no_mtp']:
        raise RuntimeError('Native Strata diagnostic incomplete')
    logpath=ROOT/'strata-native-stage-diagnostic-6144-v1.native.stderr.log'
    lines=logpath.read_text(encoding='utf-8').splitlines()
    cpu=[line for line in lines if line.startswith('strata decode timing:')]
    gpu=[line for line in lines if line.startswith('strata decode GPU stages')]
    labels=['warmup']+[r['label'] for r in reference['runs']]+['quality']
    if len(cpu)!=len(labels) or len(gpu)!=len(labels):
        raise RuntimeError('Request log alignment differs')
    stages=[]
    for label,c,g in zip(labels,cpu,gpu):
        gd,qs,total=g.split('GDN layers: ',1)[1].split(' | ')
        parse=lambda text: {name.strip():float(value) for name,value in re.findall(r'\s*(.*?)\s+(\d+\.\d+)(?=\s|$)',text)}
        row=dict(label=label,cpu_raw=c,gpu_raw=g,gdn=parse(gd),qsa=parse(qs.removeprefix('QSA layers: ')),
            native_total_ms_per_window=float(re.search(r'total (\d+\.\d+)',total)[1]),
            gpu_profile_windows=int(re.search(r'over (\d+) windows',total)[1]),
            decode_windows=int(re.search(r': (\d+) windows',c)[1]))
        row['wait_intervals_ms']=sum(v for kind in ['gdn','qsa'] for n,v in row[kind].items() if n in ['waitA','waitB','waitCPU'])
        stages.append(row)
    resource=read('strata-native-stage-diagnostic-6144-v1-resource.json')
    report=dict(scope='FreeToken 4096/6144 CUDA graph diagnostics plus frozen Strata native serve CPU/GPU event stages, all without MTP. Separate long-output uninstrumented reports are throughput evidence.',
        freetoken4096=small,freetoken6144short=large,strata6144native=stages,
        strata_log_sha256=hashlib.sha256(logpath.read_bytes()).hexdigest(),
        strata_resources={k:resource[k] for k in ['peak_commit_used','min_commit_available','min_physical_available']},
        limitations=['Strata GPU event intervals include waitA/B/CPU; do not add their waits again to host times',
            'GPU profile window counts differ from decode window counts due to extra execution/preparation; retain both and do not treat all entries as the same delivered token',
            'Native Strata stages include different grouping boundaries; no direct one-to-one comparison to FreeToken kernel families',
            'Strata resident cache can adapt between requests; FreeToken cold resets dynamic experts each cold run. Same slot count is not identical hit behavior',
            'Short profiler throughput is instrumented and capped at 32/64; not a substitute for 512-token repeated throughput',
            'Strata diagnostic physical memory availability fell below 0.4 GiB despite sufficient commit; that configuration is not resource acceptance'],
        full_runtime_acceptance=False,speed_acceptance=False)
    target.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(dict(freetoken6144=[dict(path=Path(r['path']).name,families=r['family_ms_per_decode']) for r in large],
        strata=[dict(label=r['label'],cpu_windows=r['decode_windows'],gpu_windows=r['gpu_profile_windows'],native_total_ms=r['native_total_ms_per_window'],wait_ms=r['wait_intervals_ms']) for r in stages])))
if __name__=='__main__':
    main()
