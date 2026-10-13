"""Append immutable research evidence; preserve previous revisions and large traces."""
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import shutil

ROOT=Path(__file__).resolve().parent
PRODUCER=Path(r'C:\workspace\worktrees\build-freetoken-flash-next-strata-q2')
DESKTOP=Path(r'C:\Users\PC\Desktop\Flash-Next-Strata-Q2-20261012')
DEST=PRODUCER/'artifacts/strataq2dev4-cp312-cu130-sm120/validation/native-runtime-20261013'
def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()
def main():
    if DEST.exists():raise FileExistsError(DEST)
    perf=json.loads((ROOT/'dev4-native-performance-4096-v1.json').read_text(encoding='utf-8'))
    resource=json.loads((ROOT/'dev4-native-performance-4096-v1-resource.json').read_text(encoding='utf-8'))
    if perf['status']!='native_decode_repair_performance_measured' or resource['returncode']!=0:raise RuntimeError('Capacity experiment incomplete')
    files=[p for p in ROOT.glob('dev4-*') if p.is_file()]
    for name in ['prepare_dev4_native_source.py','freeze_dev4_native_materials.py',
        'build_dev4_native_probe.ps1','dev4_native_component_validation.py','validate_dev4_runtime_evidence.py',
        'prepare_dev4_performance_harness.py','first_freetoken_performance_dev4_v1.py',
        'summarize_dev4_native_repair.py','summarize_dev4_capacity_experiment.py','archive_dev4_native_runtime.py','compare_execution_logits.py',
        'candidate_native_probe.py','diagnose_long_prefix.py','analyze_freetoken_decode_traces.py','analyze_dev4_capacity_traces.py',
        'supervise.py','capture_freetoken_reference.py']:
        files.append(ROOT/name)
    if any(not p.exists() for p in files):raise RuntimeError('Missing evidence')
    DEST.mkdir(parents=True)
    entries=[];external=[]
    for p in sorted(set(files)):
        row=dict(path=p.name,bytes=p.stat().st_size,sha256=digest(p))
        if p.suffix=='.safetensors' or p.name.endswith('.trace.json'):
            external.append(dict(**row,absolute_path=str(p),reason='Large diagnostic file retained at original immutable path'))
        else:
            shutil.copy2(p,DEST/p.name)
            if digest(DEST/p.name)!=row['sha256']:raise RuntimeError('Copy mismatch')
            entries.append(row)
    doc=PRODUCER/'docs/strata-q2-dev4-native-decode.txt'
    shutil.copy2(doc,DEST/doc.name)
    entries.append(dict(path=doc.name,bytes=doc.stat().st_size,sha256=digest(doc)))
    index=dict(created_at_utc=datetime.now(timezone.utc).isoformat(),version='0.1.3+speed.strataq2dev4',
        scope='Fresh installed native revision, bounded runtime gate, standalone decode measurements and cache capacity experiment',
        runtime_acceptance_scope='Allows independent major-repair evaluation only',
        full_runtime_acceptance=False,speed_acceptance=False,files=entries,external_large_files=external,
        immutable_inputs=['Full fixed GGUF inventory','dev1/dev2/dev3 profiles/patches/materials','Historical q2opt2 environment'],
        failures_preserved=['Build v1 stopped before compilation using Windows PowerShell5; v2 uses pwsh7',
            'Model long matrix v1 explicitly stopped due to incompatible diagnostic tail size; v2 uses established 3072 setting',
            'Verifier v1 expected wrong extension log names; verifier v2 corrected without engine source change',
            'Environment first attempt failed index metadata dependency resolution; offline pinned v2 succeeded'],
        producer_checks=dict(repository_validator='dev4-producer-validator-v1.log: verified true',unit_tests=16,unit_test_exit_code=0,
            unit_test_command='python -X utf8 -B -m unittest discover -s tests -v',
            note='16 existing producer tests completed successfully in session 30708, 12.629 seconds; not GPU or Speed acceptance'))
    (DEST/'evidence-index.json').write_text(json.dumps(index,ensure_ascii=False,indent=2),encoding='utf-8')
    snapshot=ROOT/'correctness-progress-before-native-dev4.json'
    if snapshot.exists():raise FileExistsError(snapshot)
    shutil.copy2(ROOT/'correctness-progress.json',snapshot)
    progress=json.loads((ROOT/'correctness-progress.json').read_text(encoding='utf-8'))
    progress['native_decode_repair']=dict(version=index['version'],summary=str(ROOT/'dev4-native-decode-repair-summary-v1.json'),
        runtime_gate=str(ROOT/'dev4-native-runtime-gate-v1.json'),bounded_runtime_gate_passed=True,
        native_build_complete=True,whole_release_acceptance=False,speed_acceptance=False,
        archive=str(DEST/'evidence-index.json'),expert_capacity_experiment=str(ROOT/'dev4-native-performance-4096-v1.json'),
        capacity_trace=str(ROOT/'dev4-capacity-trace-analysis-v1.json'),
        next='Decode first: expert residence/miss payload and routed expert execution. Remove full-cache expert align scan (1.18 ms at 4096 slots) and 16-token decode tile; LRU only 0.2 ms, not the same issue. Preserve prefill regression evidence. Old models, multi-request, MTP and Speed still open.')
    (ROOT/'correctness-progress.json').write_text(json.dumps(progress,ensure_ascii=False,indent=2),encoding='utf-8')
    DESKTOP.mkdir(exist_ok=True)
    shutil.copy2(doc,DESKTOP/'FreeToken-dev4原生decode修复与性能.txt')
    shutil.copy2(ROOT/'dev4-native-decode-repair-summary-v1.json',DESKTOP/'dev4-native-decode-repair-summary-v1.json')
    shutil.copy2(ROOT/'dev4-capacity-experiment-summary-v1.json',DESKTOP/'dev4-capacity-experiment-summary-v1.json')
    (ROOT/'dev4-native-evidence-archive-receipt.json').write_text(json.dumps(dict(index=str(DEST/'evidence-index.json'),
        index_sha256=digest(DEST/'evidence-index.json'),files=len(entries),external_large_files=len(external)),indent=2),encoding='utf-8')
    print(json.dumps(dict(index=str(DEST/'evidence-index.json'),files=len(entries),external_large_files=len(external))))
if __name__=='__main__':main()
