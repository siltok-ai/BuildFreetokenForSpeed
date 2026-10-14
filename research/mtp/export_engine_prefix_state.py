"""Freeze the second engine checkpoint after the immutable verification foundation."""
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
    foundation_path = REPO / 'research/mtp/evidence/engine-foundation-20261014-v1/index.json'
    foundation = json.loads(foundation_path.read_text(encoding='utf-8'))
    base = foundation['source_revision']
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=source, text=True).strip()
    if subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'], cwd=source):
        raise RuntimeError('Commit the tested development source before export')
    patch = REPO / 'patches/series/strataq2mtp1-0015-owned-prefix-state.patch'
    if patch.exists():
        raise FileExistsError(patch)
    files = subprocess.check_output(['git', 'diff', '--name-only', base, revision], cwd=source, text=True).splitlines()
    if not files or any(not name.endswith('.py') for name in files):
        raise RuntimeError('Prefix-state checkpoint must only change Python source/tests')
    patch.write_bytes(subprocess.check_output(['git', 'diff', '--binary', base, revision, '--', 'python', 'tests'], cwd=source))
    replayed, _ = prepare(REPO / foundation['base_profile'], args.replay)
    apply_patch(replayed, REPO / foundation['patch'], args.replay / 'logs/foundation.log')
    before = {name: sha256(replayed / name) if (replayed / name).exists() else None for name in files}
    apply_patch(replayed, patch, args.replay / 'logs/prefix-state.log')
    expected = {name: sha256(source / name) for name in files}
    if expected != {name: sha256(replayed / name) for name in files}:
        raise RuntimeError('Replayed prefix-state bytes differ from tested development source')
    tests = json.loads((args.development / 'prefix-tests-v3.json').read_text(encoding='utf-8'))
    if tests['status'] != 'passed' or any(tests['source_sha256'].get(name) != digest for name, digest in expected.items()):
        raise RuntimeError('Tested source differs from the exported checkpoint')
    output = REPO / 'research/mtp/evidence/engine-prefix-state-20261014-v1'
    output.mkdir(exist_ok=False)
    archives = []
    names = ['run_prefix_tests_v1.py', 'probe_mtp_config_v1.py', 'real-gguf-mtp-config-v1.json',
             'real-gguf-mtp-config-v1.log', 'idle-speed-resource-audit-v1.json', *[
                 f'prefix-tests-v{i}.{suffix}' for i in (1, 2, 3) for suffix in ('json', 'log')]]
    for name in names:
        raw = (args.development / name).read_bytes()
        path = output / (name + '.gz')
        path.write_bytes(gzip.compress(raw, mtime=0))
        if gzip.decompress(path.read_bytes()) != raw:
            raise RuntimeError('Evidence compression roundtrip failed')
        archives.append(dict(path=path.relative_to(REPO).as_posix(), sha256=sha256(path),
                             raw_sha256=hashlib.sha256(raw).hexdigest()))
    record = dict(status='development_prefix_state_component_tests_passed',
        predecessor=foundation_path.relative_to(REPO).as_posix(), predecessor_sha256=sha256(foundation_path),
        source_base_commit=base, source_revision=revision,
        patch=patch.relative_to(REPO).as_posix(), patch_sha256=sha256(patch),
        baseline_sha256=before, replayed_source_sha256=expected, replay_matches_tested_bytes=True,
        component_tests='65 passed, 12 subtests passed; real prefix fork/page recycling, CUDA residual and auxiliary state, exact budget geometry',
        source_version='dev7 with explicit development source override; not an accepted dev7 wheel',
        full_engine_acceptance=False, profile_created=False, wheel_created=False, speed_mtp_acceptance=False,
        mtp_generation_driver_wired=False, evidence=archives)
    (output / 'index.json').write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(dict(status=record['status'], patch_sha256=record['patch_sha256'], files=len(files))))


if __name__ == '__main__':
    main()
