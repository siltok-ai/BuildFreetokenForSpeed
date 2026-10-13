"""核对保护驻留补丁、排名数据和原生复用，不授予运行或 Speed 验收。"""
import argparse
import hashlib
import json
from pathlib import Path
import struct
import sys
import tempfile
import zipfile

from prepare_source import ROOT, load_profile, prepare, sha256
from validate_python_revision import validate as validate_python_revision
from validate_native_revision import validate as validate_native_revision
sys.path.insert(0, str(ROOT / 'build'))
from derive_python_materials import validate_python_only_changes


def validate(profile_path, materials=None):
    """重放实际差异并检查 wheel 中的固定排名和 CUDA 字节，防止混用版本。"""
    profile, series = load_profile(profile_path)
    expected = {'python/freetoken/engine/config.py', 'python/freetoken/engine/engine.py',
                'python/freetoken/server/args.py', 'python/freetoken/version.py',
                'python/freetoken/moe/q2_residency.py',
                'python/freetoken/moe/configs/q2_residency/strata_flash_next.json',
                'tests/moe/test_q2_residency.py', 'tests/models/qwen4_exp/test_gguf_execution.py'}
    with tempfile.TemporaryDirectory(prefix='protected-q2-replay-') as temporary:
        source, _ = prepare(profile_path, Path(temporary) / 'source')
        prior, _ = prepare(ROOT / 'build/profiles/strataq2dev5.json', Path(temporary) / 'prior')
        native_source, _ = prepare(ROOT / 'build/profiles/strataq2dev4.json', Path(temporary) / 'native')
        changed = validate_python_only_changes(prior, source)
        if changed != expected or changed != set(series['patches'][-1]['files']):
            raise ValueError('Protected revision differs from its frozen patch scope')
        native_delta = validate_python_only_changes(native_source, source)
        data_path = source / 'python/freetoken/moe/configs/q2_residency/strata_flash_next.json'
        data = json.loads(data_path.read_text())
        original = ROOT / 'research/q2-protected-cache-20261013-v1/strata-official-expert-profile.bin'
        raw = original.read_bytes()
        if sha256(original) != data['source']['sha256'] or data['source']['revision'] != '6674a0065fb96bacde33e3eb10f91a1df86f95f2':
            raise ValueError('Protected ranking source identity differs')
        version, layers, experts, slots, ranked = struct.unpack_from('<5I', raw, 4)
        if raw[:4] != b'STRP' or (version, layers, experts, slots, ranked) != (1, 48, 512, 24576, 24576):
            raise ValueError('Unexpected Strata profile geometry')
        pairs = [list(struct.unpack_from('<HH', raw, 24 + 4 * i)) for i in range(ranked)]
        if data['ranked'] != pairs or len({tuple(p) for p in pairs}) != ranked:
            raise ValueError('Packaged ranking differs from frozen Strata profile')
        result = dict(version=profile['custom_version'], source_replay_verified=True,
            changed_files=len(changed), ranking_identity_verified=True, native_inputs_unchanged=True,
            runtime_acceptance=False)
        if materials is not None:
            materials = Path(materials)
            validate_python_revision(profile_path, materials)
            record = json.loads((materials / 'build-record.json').read_text())
            if record['binary_reuse']['base_materials'] != 'strataq2dev4-cp312-cu130-sm120':
                raise ValueError('Unproven native base')
            if set(record['binary_reuse']['changed_source_files']) != native_delta:
                raise ValueError('Derived source delta differs')
            native = validate_native_revision(ROOT / 'build/profiles/strataq2dev4.json',
                                              ROOT / 'artifacts/strataq2dev4-cp312-cu130-sm120')
            if not native['full_native_compilation_recorded']:
                raise ValueError('Native base has no compilation evidence')
            with zipfile.ZipFile(materials / 'wheels' / profile['wheel_filename']) as wheel:
                expected_data = {p.relative_to(source / 'python').as_posix(): p.read_bytes()
                    for p in (source / 'python/freetoken/moe/configs').rglob('*.json')}
                actual_data = {n: wheel.read(n) for n in wheel.namelist()
                               if n.startswith('freetoken/moe/configs/') and n.endswith('.json')}
                if actual_data != expected_data:
                    raise ValueError('Wheel is missing or has altered MoE package data')
                csrc = {p.relative_to(source / 'python').as_posix(): p.read_bytes()
                        for p in (source / 'python/freetoken/kernel/csrc').rglob('*') if p.is_file()}
                if csrc != {n: wheel.read(n) for n in wheel.namelist() if n.startswith('freetoken/kernel/csrc/')}:
                    raise ValueError('Native source bytes differ')
            result.update(wheel_and_binary_identity_verified=True, aot_count=record['aot_count'],
                wheel_sha256=record['wheel_sha256'], build_record_sha256=sha256(materials / 'build-record.json'))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', type=Path, default=ROOT / 'build/profiles/strataq2dev6.json')
    parser.add_argument('--materials', type=Path)
    args = parser.parse_args()
    print(json.dumps(validate(args.profile, args.materials)))


if __name__ == '__main__':
    main()
