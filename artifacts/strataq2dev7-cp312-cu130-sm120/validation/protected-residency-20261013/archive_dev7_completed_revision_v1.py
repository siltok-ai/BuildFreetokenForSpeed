"""Freeze completed installed-wheel evidence and copy new materials without replacing old versions."""
import hashlib
import json
from pathlib import Path
import shutil
import statistics

ROOT = Path(__file__).resolve().parent
PRODUCER = Path('C:/workspace/worktrees/build-freetoken-flash-next-strata-q2')


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    archive = PRODUCER/'artifacts/strataq2dev7-cp312-cu130-sm120'
    summary_path = ROOT/'dev7-installed-summary-v1.json'
    if archive.exists() or summary_path.exists():
        raise FileExistsError('New material/evidence namespace must be unused')
    gate = json.loads((ROOT/'dev7-installed-runtime-gate-v1.json').read_text())
    performance = json.loads((ROOT/'dev7-native-performance-11264-v1.json').read_text(encoding='utf-8'))
    resource = json.loads((ROOT/'dev7-native-performance-11264-v1-resource.json').read_text())
    prototype = json.loads((ROOT/'prototype-protected-cache-performance-11264-v1.json').read_text(encoding='utf-8'))
    historical = json.loads((ROOT/'dev5-native-performance-6144-v2.json').read_text(encoding='utf-8'))
    if not gate['passed'] or resource['returncode'] != 0 or resource.get('aborted'):
        raise RuntimeError('Installed runtime or performance job failed')
    references = [{r['label']:r for r in d['runs']} for d in (prototype, historical)]
    comparisons = [dict(label=r['label'], input_matches=all(r['input_sha256']==b[r['label']]['input_sha256'] for b in references),
        full_output_tokens_match=all(r['output']['token_ids']==b[r['label']]['output']['token_ids'] for b in references))
        for r in performance['runs']]
    if (len(comparisons)!=12 or not all(all(c[k] for k in ('input_matches','full_output_tokens_match')) for c in comparisons)
            or not performance['all_measurements_sufficient'] or any(r['output_tokens']!=512 for r in performance['runs'])):
        raise RuntimeError('Long-output performance protocol/output gate differs')
    contexts = {}
    for context in ('short','diverse4096'):
        runs = [r for r in performance['runs'] if r['label'].startswith(context)]
        contexts[context] = dict(median_tps=statistics.median(r['decode_tps'] for r in runs),
            minimum_tps=min(r['decode_tps'] for r in runs), maximum_tps=max(r['decode_tps'] for r in runs),
            actual_output_tokens=[r['output_tokens'] for r in runs],
            median_miss_rate=statistics.median(r['missing_expert_selections']/r['active_expert_selections'] for r in runs),
            median_h2d_payload_per_token=statistics.median(r['expert_payload_h2d_bytes']/(r['output_tokens']-1) for r in runs))
    summary = dict(status='installed_standalone_q2_revision_verified_speed_pending',
        version=performance['version'], wheel_sha256=gate['wheel_sha256'],
        native_engine_only=True, process_engine_override=False, no_mtp=True,
        contexts=contexts, output_comparisons=comparisons, component_tests=24,
        finite_state_components=662, max_state_component_relative_l2=0.0, sparse_selections_equal=12,
        quality=dict(score=15,total=20,unchanged_full_output_tokens=True,new_regressions=0),
        selected_geometry=dict(protected=9216,dynamic=1024,prefill=1024,total_expert_slots=11264,
            kv_tokens=8192,max_running_requests=1,max_prefill_tokens=4096,memory_ratio=.9),
        resource=dict(peak_commit=resource['peak_commit_used'],min_commit_available=resource['min_commit_available'],
            peak_torch_reserved=max(r['cuda_reserved'] for r in performance['runs']),
            peak_torch_allocated=max(r['cuda_allocated'] for r in performance['runs'])),
        limitations=['Single-request measured configuration only','Fixed protected expert pool; capacity changes require reload',
            'Raw dev7 state v2 retained: test expected two tokens from a truncated chat header, which naturally produced EOS; v3 honors EOS and verifies actual16-token prefill',
            'Prefill TTFT regression from dev4 remains open; decode is priority',
            'Old-model, Speed backend/HTTP, packaged and MTP validation remain'],
        speed_acceptance=False,packaged_acceptance=False,
        next='Freeze producer and Speed candidate version, verify application behavior, then evaluate MTP with state rollback and delivered throughput')
    summary_path.write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
    archive.mkdir()
    materials = ROOT/'dev7-derived-native-base-v1'
    for name in ('wheels','kernel-cache'):
        shutil.copytree(materials/name, archive/name)
    shutil.copy2(materials/'build-record.json',archive/'build-record.json')
    shutil.copy2(materials/'candidate-source/preparation-record.json',archive/'preparation-record.json')
    evidence = archive/'validation/protected-residency-20261013'
    evidence.mkdir(parents=True)
    files = sorted({p for p in ROOT.glob('dev7-*') if p.is_file()} |
                   {p for p in ROOT.glob('*dev7*.py') if p.is_file()} |
                   {p for p in (ROOT/'quality-gate-dev7').glob('*') if p.is_file()} |
                   {ROOT/'validate_dev7_runtime_gate_v1.py',ROOT/'validate_dev7_installed_state_v2.py'})
    copied, external = [], []
    for path in files:
        relative = path.relative_to(ROOT).as_posix()
        info = dict(path=relative,bytes=path.stat().st_size,sha256=sha(path))
        if info['bytes']>16*2**20:
            external.append(info|dict(original_path=str(path)))
        else:
            destination = evidence/relative
            destination.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(path,destination)
            if sha(destination)!=info['sha256']:
                raise RuntimeError('Evidence copy changed')
            copied.append(info)
    producer_tool = PRODUCER/'build/derive_python_materials.py'
    shutil.copy2(producer_tool,evidence/'derive_python_materials.py')
    copied.append(dict(path='derive_python_materials.py',bytes=producer_tool.stat().st_size,sha256=sha(producer_tool)))
    index = dict(format=1,copied_files=copied,external_references=external,
        summary_sha256=sha(summary_path),build_record_sha256=sha(archive/'build-record.json'),
        immutable=True,speed_acceptance=False,packaged_acceptance=False)
    (evidence/'index.json').write_text(json.dumps(index,indent=2)+'\n',encoding='utf-8')
    receipt = dict(materials=str(archive),summary=summary,
        copied_files=len(copied),external_references=len(external),index_sha256=sha(evidence/'index.json'))
    (ROOT/'dev7-completed-revision-archive-receipt-v1.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in receipt.items() if k!='summary'}))


if __name__=='__main__':
    main()
