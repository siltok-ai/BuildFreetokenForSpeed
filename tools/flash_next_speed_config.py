"""按已记录的单机性能配置生成 Speed manifest；不修改机器全局配置。"""
from pathlib import Path
import argparse
import json

from flash_next_weights import ROOT, LOCK, layout, read, verify_checkpoint, write
from flash_next_mtp import verify_runtime
from current_release import load_current


def configuration(mode, preset):
    """将模型开关、专家预算和准入预算放在同一个明确的测试方案中。"""
    selected = preset['modes'][mode]
    extra = dict(attention_backend='qsa_sparse', ple_backend='disk', cache_type='radix',
        memory_ratio=.9, kv_cache_tokens=8192, cuda_graph_max_bs=1, max_prefill_length=4096,
        moe_q2_resident_experts=selected['resident'], moe_q2_dynamic_slots=1024,
        moe_cache_size=selected['total_slots'], mtp_enabled=mode == 'on',
        vram_estimate_bytes=int(selected['vram_estimate_gib']*2**30),
        host_commit_estimate_bytes=int(selected['host_commit_estimate_gib']*2**30))
    artifacts = dict(weights=dict(path='checkpoints/flash-next-strata-q2', layout='hf_model'))
    if mode == 'on':
        artifacts['mtp'] = dict(path='checkpoints/flash-next-mtp', layout='freetoken_mtp')
    manifest = dict(schema='silitok.models/1', models=[dict(id='flash-next-strata-q2-'+mode,
        name='Flash Next Strata Q2 '+mode, type='language', family='qwen3.8', features=['chat'],
        engine=dict(pipeline='chat', backend='Qwen36Backend', offload=False, extra=extra), artifacts=artifacts)])
    return manifest, dict(resource=preset['resource'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--mode', choices=['off','on','off-matched'], default='off')
    parser.add_argument('--preset', choices=['sm120-24g-measured'], required=True)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    current = load_current()
    preset = read(ROOT/'integration/speed/presets/flash-next-sm120-24g.json')
    manifest, resource = configuration(args.mode, preset)
    if args.dry_run:
        print(json.dumps(dict(manifest=manifest, config=resource, scope=preset['scope']),ensure_ascii=False))
        return
    paths = layout(args.root, read(LOCK))
    targets = [paths['root']/'manifest.json', paths['root']/'speed-config/config.yaml', paths['root']/'speed-inputs.json']
    if any(p.exists() for p in targets):
        raise FileExistsError('Existing deployment configuration is preserved; use a new model root')
    verify_checkpoint(args.root, read(LOCK))
    if args.mode == 'on':
        verify_runtime(paths['mtp'])
    write(targets[0], manifest)
    # JSON 是 YAML 子集，使用同一序列化器保持配置键和数值的精确性。
    targets[1].parent.mkdir(exist_ok=True)
    write(targets[1], resource)
    write(targets[2], dict(format=1, mode=args.mode, preset=preset['name'], scope=preset['scope'],
        freetoken_version=current['version'], materials=current['materials'],
        kernel_cache=current['materials']+'/kernel-cache',
        note='Set SILITOK_FREETOKEN_KERNEL_CACHE_DIR explicitly to this release cache. Set SILITOK_CONFIG to the speed-config directory; pass --models-dir to this model root.'))
    print(json.dumps(dict(status='prepared', model_root=str(paths['root']), mode=args.mode, files=[str(p) for p in targets]),ensure_ascii=False))


if __name__ == '__main__':
    main()
