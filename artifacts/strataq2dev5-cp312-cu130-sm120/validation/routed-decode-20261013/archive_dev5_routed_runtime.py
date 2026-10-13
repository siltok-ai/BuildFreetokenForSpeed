"""Append immutable dev5 evidence and a current goal snapshot; preserve old archives."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil

ROOT=Path(__file__).resolve().parent
PRODUCER=Path(r'C:\workspace\worktrees\build-freetoken-flash-next-strata-q2')
DESKTOP=Path(r'C:\Users\PC\Desktop\Flash-Next-Strata-Q2-20261012')
DEST=PRODUCER/'artifacts/strataq2dev5-cp312-cu130-sm120/validation/routed-decode-20261013'
def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(4*1024*1024),b''):
            h.update(block)
    return h.hexdigest()
def read(path):
    return json.loads(path.read_text(encoding='utf-8'))
def main():
    if DEST.exists():raise FileExistsError(DEST)
    summary=read(ROOT/'dev5-routed-decode-repair-summary-v1.json')
    gate=read(ROOT/'dev5-routed-runtime-gate-v1.json')
    if not gate['passed'] or summary['status']!='installed_major_expert_decode_repair_measured':
        raise RuntimeError('Installed evidence incomplete')
    validator=json.loads((ROOT/'dev5-producer-validator-v2.log').read_text(encoding='utf-8').strip().splitlines()[-1])
    unit_log=(ROOT/'dev5-producer-unit-tests-v2.log').read_text(encoding='utf-8')
    if not validator['verified'] or 'Ran 16 tests' not in unit_log or '\nOK\n' not in unit_log:
        raise RuntimeError('Producer checks incomplete')
    files=set()
    for pattern in ['dev5-*','prototype-expert-*','strata-native-stage-diagnostic-*','strata-stage-diagnostic-*','profiling-tools-availability-v1.json']:
        files.update(p for p in ROOT.glob(pattern) if p.is_file())
    for path in ROOT.glob('*.py'):
        if any(word in path.name for word in ['dev5','expert_dp4a','expert_route_plan','expert_observation','strata_stage','strata_gpu_stages']):
            files.add(path)
    for name in ['compare_execution_logits.py','candidate_native_probe.py','diagnose_long_prefix.py',
        'analyze_freetoken_decode_traces.py','supervise.py','capture_freetoken_reference.py',
        'candidate-source-dev5-materials.json','complete-weights.json','complete-gguf-inventory.json']:
        files.add(ROOT/name)
    files.update((ROOT/'quality-gate-dev5').glob('*'))
    files={p for p in files if p.is_file()}
    DEST.mkdir(parents=True)
    entries=[];external=[]
    for path in sorted(files):
        relative=path.relative_to(ROOT)
        row=dict(path=relative.as_posix(),bytes=path.stat().st_size,sha256=digest(path))
        if path.suffix=='.safetensors' or path.name.endswith('.trace.json'):
            external.append(dict(**row,absolute_path=str(path),reason='Large immutable evidence retained at its original path'))
        else:
            target=DEST/relative
            target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(path,target)
            if digest(target)!=row['sha256']:raise RuntimeError('Evidence copy differs')
            entries.append(row)
    material_inputs=[PRODUCER/'build/profiles/strataq2dev5.json',PRODUCER/'patches/strataq2dev5-series.json',
        PRODUCER/'patches/series/strataq2dev5-0011-routed-q2-expert-decode.patch',
        PRODUCER/'docs/strata-q2-dev5-routed-decode.txt',PRODUCER/'docs/strata-q2-expert-residency-design-20261013.txt']
    for path in material_inputs:
        relative=Path('materials')/path.relative_to(PRODUCER)
        target=DEST/relative
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(path,target)
        row=dict(path=relative.as_posix(),bytes=path.stat().st_size,sha256=digest(path))
        if digest(target)!=row['sha256']:raise RuntimeError('Material copy differs')
        entries.append(row)
    reference=Path(r'C:\workspace\references\strata-flash-next-research')
    reference_names=['src/kernels/cuda/s2_expert_grouped.cu','include/strata/kernels/s2_expert_grouped.hpp',
        'src/core/expert_source.cpp','include/strata/core/expert_source.hpp','src/program/generate.cpp','src/core/verify.cpp','docs/DETAILS.md']
    index=dict(created_at_utc=datetime.now(timezone.utc).isoformat(),version=gate['version'],wheel_sha256=gate['wheel_sha256'],
        native_provenance='New Python revision; unchanged native inputs and byte-for-byte reuse of dev4 actual full compilation, all 27 AOT and 3 extensions verified',
        scope='Bounded installed runtime/quality gate; repeated native no-MTP decode and 4096/6144 resource experiments; PyTorch CUDA graph traces and frozen Strata native event stages',
        bounded_runtime_gate_passed=True,full_runtime_acceptance=False,speed_acceptance=False,
        stable_100_tps_across_short_and_4k=False,files=entries,external_large_files=external,
        strata_reference=dict(commit='6674a0065fb96bacde33e3eb10f91a1df86f95f2',files=[dict(path=str(reference/n),sha256=digest(reference/n)) for n in reference_names]),
        failures_preserved=['Source GPU test v1 lacked pytest; v2 passed 12 tests',
            'Prototype v13 strict cold/hot output assertion failed after one measured cold run',
            'v14 diagnostic chunk128 hit historical cold-label assumption; explicit bounded chunk4096 retry passed',
            'Installed GPU suite 18 passed / 1 legacy bitwise failure; independent bounded precision contract passed'],
        producer_checks=dict(repository_validator='dev5-producer-validator-v2.log: verified true',unit_tests=16,unit_test_exit_code=0),
        open_work=['Joint RAM/VRAM host-complement residency and actual overlap analysis','Legacy test contract in a new revision','Prefill regression','Multi-request/old-model/MTP/Speed/packaged acceptance'])
    (DEST/'evidence-index.json').write_text(json.dumps(index,ensure_ascii=False,indent=2),encoding='utf-8')
    snapshot=ROOT/'correctness-progress-before-routed-dev5.json'
    if snapshot.exists():raise FileExistsError(snapshot)
    shutil.copy2(ROOT/'correctness-progress.json',snapshot)
    progress=read(ROOT/'correctness-progress.json')
    progress['status']='major_decode_regressions_repaired_residency_analysis_active'
    progress['version']=gate['version'];progress['wheel_sha256']=gate['wheel_sha256']
    progress['priority']='Decode first: joint RAM/VRAM expert residence, miss/copy and CPU/GPU overlap using trace and Strata reference. Preserve prefill correctness/resource/regression. No early fine tuning or Speed acceptance.'
    progress['routed_expert_decode_repair']=dict(summary=str(ROOT/'dev5-routed-decode-repair-summary-v1.json'),
        gate=str(ROOT/'dev5-routed-runtime-gate-v1.json'),archive=str(DEST/'evidence-index.json'),
        comparison_trace=str(ROOT/'dev5-capacity-and-strata-stage-analysis-v1.json'),
        short_tps4096=90.709126979239,long4k_tps4096=86.7800466697904,
        short_tps6144=104.0487952261893,long4k_tps6144=99.35384745438816,
        bounded_runtime_gate_passed=True,quality_score=15,quality_total=20,
        short_cold_hot_trajectory_note='First divergence 104/105; 4K full trajectories and both capacity outputs equal',
        legacy_gpu_suite='18 passed / 1 historical bitwise assertion failed; separately bounded precision passed',
        whole_release_acceptance=False,speed_acceptance=False,
        next_design=str(PRODUCER/'docs/strata-q2-expert-residency-design-20261013.txt'))
    (ROOT/'correctness-progress.json').write_text(json.dumps(progress,ensure_ascii=False,indent=2),encoding='utf-8')
    DESKTOP.mkdir(exist_ok=True)
    for name in ['dev5-routed-runtime-gate-v1.json','dev5-routed-decode-repair-summary-v1.json',
        'dev5-routed-trace-analysis-v1.json','dev5-capacity-and-strata-stage-analysis-v1.json']:
        shutil.copy2(ROOT/name,DESKTOP/name)
    shutil.copy2(PRODUCER/'docs/strata-q2-dev5-routed-decode.txt',DESKTOP/'FreeToken-dev5专家decode修复与性能.txt')
    shutil.copy2(PRODUCER/'docs/strata-q2-expert-residency-design-20261013.txt',DESKTOP/'FreeToken-Q2专家驻留完整设计.txt')
    receipt=dict(index=str(DEST/'evidence-index.json'),index_sha256=digest(DEST/'evidence-index.json'),files=len(entries),external_large_files=len(external))
    (ROOT/'dev5-routed-evidence-archive-receipt.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
    print(json.dumps(receipt))
if __name__=='__main__':main()
