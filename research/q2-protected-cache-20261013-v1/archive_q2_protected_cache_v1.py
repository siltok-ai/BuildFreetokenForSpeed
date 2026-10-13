"""Freeze the completed protected-cache experiment in the producer branch."""
import hashlib
import json
from pathlib import Path
import shutil
import statistics

ROOT = Path(__file__).resolve().parent
PRODUCER = Path('C:/workspace/worktrees/build-freetoken-flash-next-strata-q2')
DEST = PRODUCER / 'research/q2-protected-cache-20261013-v1'


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def read(name):
    return json.loads((ROOT / name).read_text(encoding='utf-8'))


def performance(name, resource_name):
    data, resources = read(name), read(resource_name)
    if len(data['runs']) != 12 or not data['quality_pass'] or resources['returncode'] != 0:
        raise RuntimeError('Incomplete long-output measurement')
    contexts = {}
    for prefix in ('short-', 'diverse4096-'):
        runs = [r for r in data['runs'] if r['label'].startswith(prefix)]
        contexts[prefix.rstrip('-')] = dict(
            median_tps=statistics.median(r['decode_tps'] for r in runs),
            minimum_tps=min(r['decode_tps'] for r in runs), maximum_tps=max(r['decode_tps'] for r in runs),
            median_miss_rate=statistics.median(r['moe_stats']['miss_rate'] for r in runs),
            median_payload_bytes_per_token=statistics.median(r['expert_payload_h2d_bytes'] / r['output_tokens'] for r in runs),
            actual_outputs=[r['output_tokens'] for r in runs], eos=[r['eos'] for r in runs],
            cold_hot_first_differences=[r.get('cold_hot_first_difference') for r in runs if r['label'].endswith('-hot')],
            cold_ttft_ms=[r['ttft_ms'] for r in runs if r['label'].endswith('-cold')],
            hot_ttft_ms=[r['ttft_ms'] for r in runs if r['label'].endswith('-hot')])
    return dict(contexts=contexts, peak_commit_used=resources['peak_commit_used'],
                min_commit_available=resources['min_commit_available'], min_physical_available=resources['min_physical_available'],
                no_mtp=data['no_mtp'], version=data['version'],
                prototype=data.get('prototype'), cold_cache_semantics=data.get('cold_cache_semantics'))


def main():
    if DEST.exists():
        raise FileExistsError(DEST)
    small = performance('prototype-protected-cache-performance-8192-v1.json', 'prototype-protected-cache-performance-8192-v1-resource.json')
    large = performance('prototype-protected-cache-performance-11264-v1.json', 'prototype-protected-cache-performance-11264-v1-resource.json')
    baseline = read('dev5-native-performance-6144-v2.json')
    token_checks = []
    reference = {r['label']: r for r in baseline['runs']}
    for name in ('prototype-protected-cache-performance-8192-v1.json', 'prototype-protected-cache-performance-11264-v1.json'):
        for run in read(name)['runs']:
            same = run['output']['token_ids'] == reference[run['label']]['output']['token_ids']
            token_checks.append(dict(measurement=name, label=run['label'], all_tokens_equal=same))
            if not same:
                raise RuntimeError('Owner policy changed delivered token sequence')
    gates = [read('prototype-protected-cache-state-gate-v1.json'), read('prototype-protected-cache-state-gate-11264-v1.json')]
    if not all(g['passed'] for g in gates):
        raise RuntimeError('State gate failed')
    traces = read('prototype-protected-cache-trace-analysis-11264-v1.json')
    if len(traces['rows']) != 4:
        raise RuntimeError('Incomplete decode-only trace analysis')
    summary = dict(status='bounded_prototype_ready_to_port_to_free_token_candidate',
                   installed_base_version='0.1.3+speed.strataq2dev5', protected_8192=small, protected_11264=large,
                   token_checks=token_checks, state_components=[g['components'] for g in gates],
                   max_state_component_relative_l2=max(g['max_component_relative_l2'] for g in gates),
                   traces=[dict(path=r['path'], families=r['family_ms_per_decode'], decode_forwards=r['decode_forwards'])
                           for r in traces['rows']],
                   decision='Stop additional cache-policy exploration for this version; port the 11264-slot configuration into a new optional FreeToken revision and validate actual installed materials before Speed integration.',
                   full_runtime_acceptance=False, speed_acceptance=False,
                   remaining=['Remove process overrides/global payload and add explicit configuration/resource contract',
                              'Reject unsafe fixed-owner cache resizing before destructive runtime rebuild; retain KV-only recapture',
                              'New frozen source/profile/patch/version/material identity',
                              'Actual installed GPU/state/quality/long-output and old-model regressions',
                              'Speed backend/HTTP/cancel/resource and packaged validation'],
                   limitations=['No MTP; maximum one active request, KV reserve 8192, tested prefill <=4096',
                                'One 4K output measured just below 100 tps; this is not an all-request throughput guarantee',
                                'Protected owners survive cold resets; cold means prefix/dynamic-cache cold',
                                'Short cold/hot divergence at 104/105 is identical to frozen dev5, not newly introduced',
                                '4K prefill regression versus dev3 remains; first shape compilation TTFT retained',
                                'Full-prefill diagnostic stopped at 306 seconds; decode-only replacement completed',
                                'Combined component v1 input fixture superseded by corrected v2'])
    summary_path = ROOT / 'q2-protected-residency-summary-v1.json'
    if summary_path.exists():
        raise FileExistsError(summary_path)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    files = set()
    for prefix in ('prototype-expert-residency-v1', 'prototype-protected-admission-v1', 'prototype-q2-protected-cache-',
                   'prototype-protected-cache-', 'q2-owner-commit-', 'q2-protected-residency-summary-v1'):
        files.update(p for p in ROOT.glob(prefix + '*') if p.is_file())
    for pattern in ('prototype_expert_residency_v1.py', 'prototype_protected_admission_v1.py',
                    'probe_q2_owner_commit_v1.py', 'prototype_q2_protected_cache_v1.py',
                    'test_prototype_q2_protected_cache_v*.py', 'run_protected_cache_v*.py',
                    'validate_protected_cache_state*.py', 'analyze_protected_cache_traces*.py',
                    'prepare_decode_only_profiler_v1.py', 'decode-only-profiler-derivation-v1.json',
                    'first_freetoken_decode_only_diagnostic_v1.py', 'capture_freetoken_reference.py',
                    'candidate_native_probe.py', 'first_freetoken_performance_dev5_v1.py',
                    'compare_execution_logits.py', 'analyze_freetoken_decode_traces.py', 'diagnose_long_prefix.py',
                    'supervise.py', 'strata-official-expert-profile.bin', 'complete-gguf-inventory.json',
                    'dev5-installed-environment-freeze-v1.txt', 'archive_q2_protected_cache_v1.py'):
        files.update(ROOT.glob(pattern))
    DEST.mkdir(parents=True)
    copied, external = [], []
    for path in sorted(files):
        entry = dict(path=path.name, bytes=path.stat().st_size, sha256=sha(path))
        if entry['bytes'] > 16 << 20:
            entry['absolute_path'] = str(path)
            external.append(entry)
        else:
            shutil.copy2(path, DEST / path.name)
            if sha(DEST / path.name) != entry['sha256']:
                raise RuntimeError('Copy hash mismatch')
            copied.append(entry)
    index = dict(copied=copied, external=external, archive=str(DEST), immutable=True,
                 installed_native_base='strataq2dev5-cp312-cu130-sm120',
                 source_inputs_unchanged=True, full_runtime_acceptance=False, speed_acceptance=False,
                 stopped_diagnostic='prototype-protected-cache-diagnostic-8192-v1 is incomplete, preserved for diagnosis only',
                 numerical_component_v1_superseded=True)
    (DEST / 'index.json').write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding='utf-8')
    receipt = dict(archive=str(DEST), index_sha256=sha(DEST / 'index.json'), copied_files=len(copied), external_files=len(external))
    (ROOT / 'q2-protected-residency-archive-receipt-v1.json').write_text(json.dumps(receipt, indent=2), encoding='utf-8')
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
