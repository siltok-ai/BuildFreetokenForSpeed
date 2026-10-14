"""Freeze the bounded-window fix and actual scheduler lifecycle evidence."""
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
    prior_path = REPO / 'research/mtp/evidence/engine-scheduler-20261014-v1/index.json'
    prior = json.loads(prior_path.read_text(encoding='utf-8'))
    prefix = json.loads((REPO / prior['predecessor']).read_text(encoding='utf-8'))
    foundation = json.loads((REPO / prefix['predecessor']).read_text(encoding='utf-8'))
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=source, text=True).strip()
    if subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'], cwd=source):
        raise RuntimeError('Commit the reviewed source before exporting')
    files = subprocess.check_output(['git', 'diff', '--name-only', prior['source_revision'], head], cwd=source, text=True).splitlines()
    expected_files = {'python/freetoken/models/qwen4_exp/mtp_runtime.py', 'tests/models/qwen4_exp/test_mtp_runtime.py'}
    if set(files) != expected_files:
        raise RuntimeError('Unexpected bounded-window source inventory')
    actual = json.loads((args.development / 'mtp-lifecycle-v5.json').read_text(encoding='utf-8'))
    components = json.loads((args.development / 'lifecycle-components-v1.json').read_text(encoding='utf-8'))
    if actual['status'] != 'passed' or components['status'] != 'passed':
        raise RuntimeError('Lifecycle/component tests have not passed')
    expected = {name: sha256(source / name) for name in files}
    for name, digest in expected.items():
        if components['source_sha256'].get(name) != digest:
            raise RuntimeError('Component source identity differs: ' + name)
        if name.startswith('python/') and actual['source_sha256'].get(name) != digest:
            raise RuntimeError('Actual lifecycle source identity differs: ' + name)
    patch = REPO / 'patches/series/strataq2mtp1-0017-bounded-windows.patch'
    if patch.exists():
        raise FileExistsError(patch)
    patch.write_bytes(subprocess.check_output(['git', 'diff', '--binary', prior['source_revision'], head, '--', 'python', 'tests'], cwd=source))
    replayed, _ = prepare(REPO / foundation['base_profile'], args.replay)
    for i, data in enumerate((foundation, prefix, prior)):
        apply_patch(replayed, REPO / data['patch'], args.replay / f'logs/prior-{i}.log')
    before = {name: sha256(replayed / name) if (replayed / name).exists() else None for name in files}
    apply_patch(replayed, patch, args.replay / 'logs/bounded.log')
    if expected != {name: sha256(replayed / name) for name in files}:
        raise RuntimeError('Lifecycle patch replay differs from actual tested bytes')
    output = REPO / 'research/mtp/evidence/engine-lifecycle-20261014-v1'
    output.mkdir(exist_ok=False)
    archives = []
    names = ['run_lifecycle_components_v1.py', 'lifecycle-components-v1.json', 'lifecycle-components-v1.log',
             *[f'probe_mtp_lifecycle_v{i}.py' for i in range(1, 6)],
             *[f'mtp-lifecycle-v{i}.json' for i in range(1, 6)],
             *[f'mtp-lifecycle-v{i}-resource{s}' for i in range(1, 8)
               for s in ('.json', '.log', '.stdout.log', '.stderr.log')]]
    for name in names:
        raw = (args.development / name).read_bytes()
        path = output / (name + '.gz')
        path.write_bytes(gzip.compress(raw, mtime=0))
        archives.append(dict(path=path.relative_to(REPO).as_posix(), sha256=sha256(path),
                             raw_sha256=hashlib.sha256(raw).hexdigest()))
    record = dict(status='development_basic_scheduler_lifecycle_verified',
        predecessor=prior_path.relative_to(REPO).as_posix(), predecessor_sha256=sha256(prior_path),
        source_base_commit=prior['source_revision'], source_revision=head,
        patch=patch.relative_to(REPO).as_posix(), patch_sha256=sha256(patch),
        baseline_sha256=before, replayed_source_sha256=expected, replay_matches_tested_bytes=True,
        source_override=True, component_tests='83 passed, 12 subtests passed',
        basic_scheduler_lifecycle_verified=True, full_engine_acceptance=False, speed_mtp_acceptance=False,
        profile_created=False, wheel_created=False, performance_acceptance=False,
        lifecycle_checks=actual['checks'], evidence=archives,
        failure_notes=['v1 import guard, before model load', 'v2 tokenizer return_dict argument',
                       'v3 native sampling uses a separate _UGEN generator; torch.manual_seed alone does not reset it',
                       'v4 cancellation harness used the wrong ForwardOutput event field; v5 uses the existing tuple ABI',
                       'v5 cancellation harness assumed more than one committed row; corrected to cached_len=cpu_length-1+bundle length',
                       'v6 harness script creation command quoting failed; the supervisor rejected the missing script before model loading'],
        scope='Length/EOS/ignore-EOS/multi-token stop/native sampling fallback/hot prefix/fork/in-flight cancel and reuse/idle rejection/rebuild/chunked prefill/eight repeated cleanup requests/shutdown; development source, not Speed',
        stress_limit='Artificial pressure prompts actually ended at EOS after 1/2 tokens; chunked finalization retained no useful full snapshot. This is cleanup coverage, not evidence of prefix/state eviction under pressure.',
        pending=['actual retained-prefix/state eviction pressure and page-boundary tool anchor',
                 'failure-path service gates', 'independent installed profile/wheel and performance',
                 'Speed on/off backend/HTTP, old-model and package validation'])
    (output / 'index.json').write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(dict(status=record['status'], patch_sha256=record['patch_sha256'], checks=len(actual['checks']))))


if __name__ == '__main__':
    main()
