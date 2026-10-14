"""Freeze and replay the first MTP engine checkpoint without changing accepted profiles."""
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
from prepare_source import apply_patch, prepare


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--development', type=Path, required=True)
    parser.add_argument('--replay', type=Path, required=True)
    args = parser.parse_args()
    source = args.development / 'build-source'
    base = subprocess.check_output(['git', 'rev-list', '--max-parents=0', 'HEAD'], cwd=source, text=True).strip()
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=source, text=True).strip()
    if subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'], cwd=source):
        raise RuntimeError('Commit the development checkpoint before export')
    patch = REPO / 'patches/series/strataq2mtp1-0014-verification-foundation.patch'
    if patch.exists():
        raise FileExistsError(patch)
    diff = subprocess.check_output(['git', 'diff', '--binary', base, revision, '--', 'python', 'tests'], cwd=source)
    files = subprocess.check_output(['git', 'diff', '--name-only', base, revision], cwd=source, text=True).splitlines()
    if not files or any(not p.endswith('.py') for p in files):
        raise RuntimeError('This checkpoint must only change Python source and tests')
    patch.write_bytes(diff)
    replayed, _ = prepare(REPO / 'build/profiles/strataq2dev7.json', args.replay)
    baseline = {p: sha(replayed / p) for p in files}
    apply_patch(replayed, patch, args.replay / 'logs/mtp-foundation.log')
    expected = {p: sha(source / p) for p in files}
    if expected != {p: sha(replayed / p) for p in files}:
        raise RuntimeError('Replayed bytes differ from tested development source')
    evidence = REPO / 'research/mtp/evidence/engine-foundation-20261014-v1'
    evidence.mkdir(exist_ok=False)
    archived = []
    for name in ['preparation-record.json', 'run_foundation_tests_v1.py', *[
            f'foundation-tests-v{i}.{suffix}' for i in (1, 2, 3) for suffix in ('json', 'log')]]:
        original = args.development / name
        raw = original.read_bytes()
        target = evidence / (name + '.gz')
        packed = gzip.compress(raw, mtime=0)
        if gzip.decompress(packed) != raw:
            raise RuntimeError('Evidence roundtrip failed')
        target.write_bytes(packed)
        archived.append(dict(path=target.relative_to(REPO).as_posix(), sha256=sha(target),
                             raw_sha256=hashlib.sha256(raw).hexdigest()))
    passed = json.loads((args.development / 'foundation-tests-v3.json').read_text(encoding='utf-8'))
    if passed['status'] != 'passed':
        raise RuntimeError('Final component tests did not pass')
    for p in files:
        if p in passed['sha256'] and passed['sha256'][p] != expected[p]:
            raise RuntimeError('Tested source changed after verification: ' + p)
    record = dict(status='development_checkpoint_component_tests_passed',
        base_profile='build/profiles/strataq2dev7.json', base_profile_sha256=sha(REPO / 'build/profiles/strataq2dev7.json'),
        patch=patch.relative_to(REPO).as_posix(), patch_sha256=sha(patch),
        source_base_commit=base, source_revision=revision,
        baseline_sha256=baseline, replayed_source_sha256=expected, replay_matches_tested_bytes=True,
        component_tests='34 passed, 12 subtests passed; includes native CUDA state equality and graph live-slot retargeting',
        full_engine_acceptance=False, profile_created=False, wheel_created=False, speed_mtp_acceptance=False,
        source_version='0.1.3+speed.strataq2dev7 with explicitly declared development source override',
        evidence=archived)
    (evidence / 'index.json').write_text(json.dumps(record, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({k: record[k] for k in ('status', 'patch_sha256', 'source_revision', 'replay_matches_tested_bytes')}))


if __name__ == '__main__':
    main()
