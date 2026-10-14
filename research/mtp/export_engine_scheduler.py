"""Archive the tested greedy scheduler checkpoint after the immutable prefix-state patch."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
import sys

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'tools'))
from prepare_source import apply_patch, prepare, sha256


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--development', type=Path, required=True)
    parser.add_argument('--replay', type=Path, required=True)
    args = parser.parse_args()
    source = args.development / 'build-source'
    previous_path = REPO / 'research/mtp/evidence/engine-prefix-state-20261014-v1/index.json'
    previous = json.loads(previous_path.read_text(encoding='utf-8'))
    foundation = json.loads((REPO / previous['predecessor']).read_text(encoding='utf-8'))
    base = previous['source_revision']
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=source, text=True).strip()
    if subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'], cwd=source):
        raise RuntimeError('Commit the development source before exporting')
    patch = REPO / 'patches/series/strataq2mtp1-0016-greedy-scheduler.patch'
    if patch.exists():
        raise FileExistsError(patch)
    files = subprocess.check_output(['git', 'diff', '--name-only', base, head], cwd=source, text=True).splitlines()
    patch.write_bytes(subprocess.check_output(['git', 'diff', '--binary', base, head, '--', 'python', 'tests'], cwd=source))
    replayed, _ = prepare(REPO / foundation['base_profile'], args.replay)
    for i, data in enumerate((foundation, previous)):
        apply_patch(replayed, REPO / data['patch'], args.replay / f'logs/prior-{i}.log')
    before = {name: sha256(replayed / name) if (replayed / name).exists() else None for name in files}
    apply_patch(replayed, patch, args.replay / 'logs/scheduler.log')
    expected = {name: sha256(source / name) for name in files}
    if expected != {name: sha256(replayed / name) for name in files}:
        raise RuntimeError('Scheduler replay differs from actual source bytes')
    long_run = json.loads((args.development / 'integrated-mtp-v2.json').read_text(encoding='utf-8'))
    components = json.loads((args.development / 'runtime-components-v2.json').read_text(encoding='utf-8'))
    if long_run['status'] != 'passed' or components['status'] != 'passed':
        raise RuntimeError('Greedy scheduler tests did not pass')
    for name, digest in expected.items():
        if long_run['source_sha256'].get(name) != digest or components['source_sha256'].get(name) != digest:
            raise RuntimeError('Exported scheduler differs from tested source: ' + name)
    if len(long_run['runs']) != 4 or any(not r['output_matches_baseline'] or r['actual_output_tokens'] != 512
                                        for r in long_run['runs']):
        raise RuntimeError('Four full-length verified outputs are required')
    output = REPO / 'research/mtp/evidence/engine-scheduler-20261014-v1'
    output.mkdir(exist_ok=False)
    names = ['promote_runtime_helpers_v1.py', 'extend_verified_delivery_v1.py', 'probe_integrated_mtp_v1.py',
             'probe_integrated_mtp_v2.py', *[f'runtime-components-v{i}.{s}' for i in (1, 2) for s in ('json', 'log')],
             *[f'integrated-mtp-v{i}{s}' for i in (1, 2)
               for s in ('.json', '.log', '-resource.json', '-resource.stdout.log', '-resource.stderr.log')]]
    archives = []
    for name in names:
        raw = (args.development / name).read_bytes()
        path = output / (name + '.gz')
        path.write_bytes(gzip.compress(raw, mtime=0))
        archives.append(dict(path=path.relative_to(REPO).as_posix(), sha256=sha256(path),
                             raw_sha256=hashlib.sha256(raw).hexdigest()))
    record = dict(status='development_greedy_scheduler_full_length_outputs_verified',
        predecessor=previous_path.relative_to(REPO).as_posix(), predecessor_sha256=sha256(previous_path),
        source_base_commit=base, source_revision=head, patch=patch.relative_to(REPO).as_posix(), patch_sha256=sha256(patch),
        baseline_sha256=before, replayed_source_sha256=expected, replay_matches_tested_bytes=True,
        source_override=True, component_tests='80 passed, 12 subtests passed', mtp_generation_driver_wired=True,
        full_engine_acceptance=False, lifecycle_acceptance=False, profile_created=False, wheel_created=False,
        speed_mtp_acceptance=False, performance_acceptance=False,
        isolated_results=[{k:r[k] for k in ('label', 'actual_output_tokens', 'output_matches_baseline', 'actual_decode_tps', 'ttft_seconds')}
                          for r in long_run['runs']],
        performance_scope='Full-weight native scheduler with CPU timing observer; one 512-token run per scenario, not Speed or a stable benchmark',
        pending=['EOS/stop/cancel/pressure/rebuild/non-greedy lifecycle', 'separate long-output performance and trace',
                 'independent profile/wheel/consumer binding', 'Speed backend/HTTP/old-model/package acceptance'],
        evidence=archives)
    (output / 'index.json').write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(dict(status=record['status'], patch_sha256=record['patch_sha256'], files=len(files))))


if __name__ == '__main__':
    main()
