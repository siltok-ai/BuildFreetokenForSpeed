"""按发布preset生成Speed manifest；硬件自动档位由Speed解析，不修改全局配置。"""
from pathlib import Path
import argparse
import json
import os
import shutil

from flash_next_weights import ROOT, LOCK, layout, read, verify_checkpoint, write
from flash_next_mtp import verify_runtime
from current_release import load_current
from prepare_source import load_profile, sha256

PRESETS = {
    'sm120-auto': 'integration/speed/presets/flash-next-sm120-auto.json',
    'sm120-16g-portable1': 'integration/speed/presets/flash-next-sm120-16g-portable1.json',
}


def configuration(mode, preset):
    """将模型开关、专家预算和准入预算放在同一个明确的测试方案中。"""
    if mode not in preset['modes']:
        raise ValueError('Mode '+mode+' is not supported by preset '+preset['name'])
    selected = preset['modes'][mode]
    extra = dict(attention_backend='qsa_sparse', ple_backend='disk', cache_type='radix',
        memory_ratio=.9, kv_cache_tokens=8192, cuda_graph_max_bs=1, mtp_enabled=mode == 'on')
    if 'resource_profile' not in preset:
        # 历史固定preset保持原合同；新资源包交由Speed的现有硬件档位统一解析。
        extra.update(max_prefill_length=4096, moe_q2_resident_experts=selected['resident'],
            moe_q2_dynamic_slots=1024, moe_cache_size=selected['total_slots'],
            vram_estimate_bytes=int(selected['vram_estimate_gib']*2**30),
            host_commit_estimate_bytes=int(selected['host_commit_estimate_gib']*2**30))
    extra.update(preset.get('runtime_extra', {}))
    if mode == 'on' and preset.get('compact_mtp', False):
        extra['mtp_compact_state'] = True
    if mode == 'on' and 'mtp_draft_policy' in preset:
        extra['mtp_draft_policy'] = preset['mtp_draft_policy']
    if 'startup_timeout_seconds' in preset:
        extra.update(startup_timeout_seconds=preset['startup_timeout_seconds'], request_deadline_seconds=1800)
    artifacts = dict(weights=dict(path='checkpoints/flash-next-strata-q2', layout='hf_model'))
    if mode == 'on':
        artifacts['mtp'] = dict(path='checkpoints/flash-next-mtp', layout='freetoken_mtp')
    manifest = dict(schema='silitok.models/1', models=[dict(id='flash-next-strata-q2-'+mode,
        name='Flash Next Strata Q2 '+mode, type='language', family='qwen3.8', features=['chat'],
        engine=dict(pipeline='chat', backend='Qwen36Backend', offload=False, extra=extra), artifacts=artifacts)])
    if 'resource_profile' in preset:
        manifest['models'][0]['engine']['resource_profile'] = preset['resource_profile']
    return manifest, dict(resource=preset['resource'])


def link_checkpoints(source, destination):
    """只创建新的共享checkpoint入口，保留已有文件与链接，不复制或移动权重。"""
    if os.path.lexists(destination):
        raise FileExistsError(destination)
    if os.name == 'nt':
        import _winapi
        _winapi.CreateJunction(str(source), str(destination))
    else:
        destination.symlink_to(source, target_is_directory=True)


def main():
    """先验真共享权重，再在独立部署目录生成配置；旧输出始终保留。"""
    current = load_current()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--mode', choices=['off','on','off-matched'], default='off')
    parser.add_argument('--preset', choices=sorted(PRESETS),
                        default=read(ROOT/current['speed_preset'])['name'])
    parser.add_argument('--deployment-root', type=Path,
                        help='New configuration root sharing checkpoints from --root; defaults to --root')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--profile', type=Path,
                        help='Explicit version profile inside build/profiles; default uses current release')
    parser.add_argument('--source-lock', type=Path, default=LOCK)
    args = parser.parse_args()
    preset = read(ROOT/PRESETS[args.preset])
    if args.profile is not None:
        profile_path = (ROOT/args.profile).resolve(strict=True)
        if profile_path.parent != (ROOT/'build/profiles').resolve():
            raise ValueError('Version profile must be inside build/profiles')
        profile, _ = load_profile(profile_path)
        materials = 'artifacts/'+profile_path.stem+'-cp312-cu130-sm120'
        record = read(ROOT/materials/'build-record.json')
        if (record['custom_version'] != profile['custom_version']
                or sha256(ROOT/materials/'wheels'/profile['wheel_filename']) != record['wheel_sha256']):
            raise ValueError('Wheel differs from its frozen profile/materials')
        current = dict(version=profile['custom_version'], materials=materials,
                       profile=profile_path.relative_to(ROOT).as_posix())
    if preset.get('required_version', current['version']) != current['version']:
        raise ValueError('Preset and FreeToken version differ; select the matching --profile')
    manifest, resource = configuration(args.mode, preset)
    lock = read(args.source_lock)
    paths = layout(args.root, lock)
    deployment = (args.deployment_root or paths['root']).resolve()
    if args.dry_run:
        print(json.dumps(dict(manifest=manifest, config=resource, scope=preset['scope'],
                             freetoken_version=current['version']),ensure_ascii=False))
        return
    targets = [deployment/'manifest.json', deployment/'speed-config/config.yaml', deployment/'speed-inputs.json']
    if any(p.exists() for p in targets):
        raise FileExistsError('Existing deployment configuration is preserved; use a new deployment root')
    shared = deployment != paths['root']
    if shared and os.path.lexists(deployment/'checkpoints'):
        raise FileExistsError('Existing checkpoint entry is preserved; use a new deployment root')
    verify_checkpoint(args.root, lock)
    if args.mode == 'on':
        verify_runtime(paths['mtp'])
    deployment.mkdir(parents=True, exist_ok=True)
    if shared:
        if 'metadata' in lock:
            link_checkpoints(paths['root']/'checkpoints', deployment/'checkpoints')
        else:
            shutil.copytree(paths['root']/'checkpoints', deployment/'checkpoints')
    write(targets[0], manifest)
    # JSON 是 YAML 子集，使用同一序列化器保持配置键和数值的精确性。
    targets[1].parent.mkdir(exist_ok=True)
    write(targets[1], resource)
    write(targets[2], dict(format=1, mode=args.mode, preset=preset['name'], scope=preset['scope'],
        freetoken_version=current['version'], materials=current['materials'],
        weight_root=str(paths['root']), deployment_root=str(deployment), preset_file=PRESETS[args.preset],
        kernel_cache=current['materials']+'/kernel-cache',
        note='Set SILITOK_FREETOKEN_KERNEL_CACHE_DIR explicitly to this release cache. Set SILITOK_CONFIG to the speed-config directory; pass --models-dir to this model root.'))
    print(json.dumps(dict(status='prepared', model_root=str(deployment), weight_root=str(paths['root']), mode=args.mode, files=[str(p) for p in targets]),ensure_ascii=False))


if __name__ == '__main__':
    main()
