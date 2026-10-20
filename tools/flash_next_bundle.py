"""制作独立可搬迁的 Flash Next 资源包；接收方无需运行转换或准备工具。"""
from pathlib import Path
import argparse
import json
import shutil

from flash_next_gguf import read_metadata, model_config, export_metadata
from flash_next_weights import ROOT, read, write, layout, verify_file
from current_release import load_current

CURRENT = load_current()
SOURCE_LOCK = ROOT/CURRENT['weight_source_lock']
PROFILE = ROOT/CURRENT['profile']
PRESET = ROOT/CURRENT['speed_preset']


def shard_rows(lock):
    """文件路径相对于 checkpoint，来源信息仅用于说明，不参与运行时身份校验。"""
    return [dict(path='gguf/'+Path(row['name']).name, filename=row['name'], size=row['bytes'])
            for row in lock['gguf']['files']]


def prepare_checkpoint(root, lock, gguf_directory=None):
    """复制真实 GGUF 文件并从包内分片生成配套配置，不创建链接或外部引用。"""
    paths = layout(root, lock)
    output = paths['checkpoint']
    if output.exists():
        raise FileExistsError('Use a new checkpoint directory: '+str(output))
    source = Path(gguf_directory) if gguf_directory else paths['gguf']
    for row in lock['gguf']['files']:
        verify_file(source/row['name'], row)
    (output/'gguf').mkdir(parents=True)
    for row in lock['gguf']['files']:
        shutil.copy2(source/row['name'], output/'gguf'/Path(row['name']).name)
    metadata = read_metadata(output/shard_rows(lock)[0]['path'])
    config = model_config(metadata, shard_rows(lock), lock['expert_geometry'])
    export_metadata(output, metadata, config)
    write(output/'gguf-preparation.json', dict(format=2, model=lock['model'],
        metadata_source='embedded_gguf', gguf_repo=lock['gguf']['repo'], gguf_revision=lock['gguf']['revision']))
    return dict(status='prepared', checkpoint=str(output), metadata_source='embedded_gguf')


def verify_checkpoint(root, lock):
    """制包端显式核验来源字节及生成配置；消费者运行时不需要制包端回执。"""
    output = layout(root, lock)['checkpoint']
    for row, expected in zip(shard_rows(lock), lock['gguf']['files']):
        verify_file(output/row['path'], expected)
    metadata = read_metadata(output/shard_rows(lock)[0]['path'])
    expected = model_config(metadata, shard_rows(lock), lock['expert_geometry'])
    if read(output/'config.json') != expected:
        raise ValueError('Generated configuration differs from embedded GGUF geometry')
    if (output/'chat_template.jinja').read_text(encoding='utf-8') != metadata['tokenizer.chat_template']:
        raise ValueError('Chat template differs from the GGUF template')
    for name in ('tokenizer.json', 'tokenizer_config.json', 'generation_config.json'):
        read(output/name)
    return dict(status='verified', checkpoint=str(output), complete_gguf=True, metadata_source='embedded_gguf')


def prepare_bundle(output, gguf_directory, mtp_directory, lock=None):
    """把主模型、现成 MTP 和两档 manifest 组合为可直接交付的独立模型目录。"""
    from flash_next_mtp import verify_runtime
    from flash_next_speed_config import configuration
    lock = lock or read(SOURCE_LOCK)
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError('Use a new resource package directory: '+str(output))
    verify_runtime(mtp_directory)
    prepare_checkpoint(output, lock, gguf_directory)
    target = layout(output, lock)['mtp']
    target.mkdir()
    manifest = read(Path(mtp_directory)/'runtime.json')
    for name in ['runtime.json', *(row['path'] for row in manifest['tensors'])]:
        shutil.copy2(Path(mtp_directory)/name, target/name)
    preset = read(PRESET)
    off, resource = configuration('off', preset)
    on, _ = configuration('on', preset)
    models = off['models']+on['models']
    for model in models:
        model['engine']['backend'] = 'FreeTokenBackend'
    write(output/'manifest.json', dict(schema='silitok.models/1', models=models))
    (output/'speed-config').mkdir()
    write(output/'speed-config/config.yaml', resource)
    write(output/'sources.json', dict(format=1, freetoken_version=read(PROFILE)['custom_version'],
        gguf=lock['gguf'], metadata='Derived from the GGUF vocabulary, merges, template and architecture metadata',
        mtp=lock['mtp'], preset=preset['name'], runtime_requires_source_repositories=False))
    shutil.copytree(SOURCE_LOCK.parent/'licenses', output/'licenses')
    shutil.copy2(SOURCE_LOCK.parent/'README.txt', output/'README.txt')
    return dict(status='prepared', package=str(output), model_ids=[m['id'] for m in models],
        required_freetoken_version=read(PROFILE)['custom_version'])


def main():
    """制作者使用匹配 FreeToken 环境；终端用户只消费生成的目录。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gguf-dir', type=Path, required=True,
        help='Directory containing Q2_0/<the two original GGUF shard names>')
    parser.add_argument('--mtp-dir', type=Path, required=True, help='Prepared compact MTP runtime directory')
    args = parser.parse_args()
    print(json.dumps(prepare_bundle(args.output, args.gguf_dir, args.mtp_dir), ensure_ascii=False))


if __name__ == '__main__':
    main()
