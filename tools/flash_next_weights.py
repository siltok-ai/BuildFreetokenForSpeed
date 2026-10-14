"""准备完整 Strata Q2 checkpoint；权重只读校验，路径可随模型目录搬迁。"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
LOCK = ROOT / 'model-weights/qwen3.8-flash-next-strata-q2/source-lock.json'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2)+'\n',
                          encoding='utf-8', newline='\n')


def verify_file(path, identity):
    """用完整 SHA 核验本机文件版本，拒绝校验中发生的替换或修改。"""
    path = Path(path)
    before = path.stat()
    if before.st_size != identity['bytes']:
        raise ValueError('size mismatch: '+str(path))
    with path.open('rb') as stream:
        actual = hashlib.file_digest(stream, 'sha256').hexdigest()
    after = path.stat()
    if actual != identity['sha256']:
        raise ValueError('SHA256 mismatch: '+str(path))
    if (before.st_size, before.st_mtime_ns, before.st_ino) != (after.st_size, after.st_mtime_ns, after.st_ino):
        raise ValueError('file changed while verifying: '+str(path))
    return after


def source_directory(root, source):
    return Path(root)/'sources'/source['repo']/source['revision']


def layout(root, lock):
    root = Path(root).resolve()
    return dict(root=root, gguf=source_directory(root, lock['gguf']),
                metadata=source_directory(root, lock['metadata']),
                checkpoint=root/'checkpoints/flash-next-strata-q2',
                mtp=root/'checkpoints/flash-next-mtp',
                mtp_source=root/'sources/Qwen/Qwen3.8-Flash-Next'/lock['mtp']['revision']/'mtp-source-bf16')


def plan(root, lock):
    """计划不访问网络或创建目录，完整列明两片和五份元数据。"""
    paths = layout(root, lock)
    return dict(format=1, scope=lock['scope'], paths={k:str(v) for k,v in paths.items()},
                sources={k:lock[k] for k in ('gguf','metadata','mtp')},
                downloaded_bytes=sum(f['bytes'] for k in ('gguf','metadata') for f in lock[k]['files']))


def download(root, lock, workers=8, metadata_only=False):
    """只使用固定 revision；已有文件必须验真，不覆盖损坏来源。"""
    from download_q2_experts import download_ranges
    for kind in (('metadata',) if metadata_only else ('gguf','metadata')):
        source = lock[kind]
        directory = source_directory(root, source)
        for row in source['files']:
            destination = directory/row['name']
            if destination.exists():
                verify_file(destination, row)
                continue
            if kind == 'gguf':
                identity = dict(repo=source['repo'], revision=source['revision'],
                                filename=row['name'], size=row['bytes'], sha256=row['sha256'])
                download_ranges(directory, workers, 'huggingface', identity)
            else:
                destination.parent.mkdir(parents=True, exist_ok=True)
                partial = destination.with_suffix(destination.suffix+'.partial')
                url = f"https://huggingface.co/{source['repo']}/resolve/{source['revision']}/{row['name']}"
                with urllib.request.urlopen(url, timeout=120) as response, partial.open('wb') as target:
                    shutil.copyfileobj(response, target)
                verify_file(partial, row)
                partial.replace(destination)
            verify_file(destination, row)
    return dict(status='downloaded_and_verified', paths=plan(root, lock)['paths'])


def build_config(output, gguf_directory, metadata_directory, lock):
    """从已锁定元数据建立全 GGUF 配置；时间回执只代表当前文件系统。"""
    for row in lock['metadata']['files']:
        verify_file(Path(metadata_directory)/row['name'], row)
    config = read(Path(metadata_directory)/'config.json')
    config.pop('quantization_config', None)
    config.pop('freetoken_expert_source', None)
    config.pop('freetoken_gguf_source', None)
    shards = []
    for row in lock['gguf']['files']:
        path = (Path(gguf_directory)/row['name']).resolve()
        stat = verify_file(path, row)
        with path.open('rb') as stream:
            if stream.read(8) != b'GGUF\x03\x00\x00\x00':
                raise ValueError('GGUF v3 header mismatch: '+str(path))
        try:
            os.path.relpath(path, Path(output).resolve())
        except ValueError as exc:
            raise ValueError('Keep checkpoint and GGUF sources on the same volume for portable paths') from exc
        shards.append(dict(repo=lock['gguf']['repo'], revision=lock['gguf']['revision'],
                           filename=row['name'], size=row['bytes'], sha256=row['sha256'],
                           path=str(path), verified_mtime_ns=stat.st_mtime_ns))
    first = shards[0]
    config['freetoken_gguf_source'] = dict(format='gguf_strata', revision=lock['gguf']['revision'], shards=shards)
    config['freetoken_expert_source'] = dict(format='q2_0', scope='routed_experts_only',
        **{k:first[k] for k in ('path','sha256','size','repo','revision','verified_mtime_ns')},
        **lock['expert_geometry'])
    return config


def prepare(root, lock, gguf_directory=None, metadata_directory=None, refresh=False):
    """新建独立 checkpoint；搬迁后只能刷新由此工具持有且来源相同的目录。"""
    paths = layout(root, lock)
    output = paths['checkpoint']
    marker = output/'strata-q2-preparation.json'
    identity = dict(format=1, model=lock['model'], gguf_revision=lock['gguf']['revision'],
                    metadata_revision=lock['metadata']['revision'])
    if output.exists():
        observed = read(marker) if marker.is_file() else {}
        if not refresh or {k:v for k,v in observed.items() if k != 'sources'} != identity:
            raise FileExistsError('Existing or unowned checkpoint is preserved: '+str(output))
    elif refresh:
        raise FileNotFoundError('Refresh requires a checkpoint created by this tool')
    portable = read(marker).get('sources', {}) if refresh else {}
    gguf_directory = Path(gguf_directory) if gguf_directory else output/portable['gguf'] if portable else paths['gguf']
    metadata_directory = Path(metadata_directory) if metadata_directory else output/portable['metadata'] if portable else paths['metadata']
    # 在任何写入前确认所有来源都能形成可迁移的同盘描述。
    sources = dict(gguf=Path(os.path.relpath(gguf_directory.resolve(), output.resolve())).as_posix(),
                   metadata=Path(os.path.relpath(metadata_directory.resolve(), output.resolve())).as_posix())
    config = build_config(output, gguf_directory, metadata_directory, lock)
    if refresh:
        verify_file(output/'hf-config.original.json', next(r for r in lock['metadata']['files'] if r['name'] == 'config.json'))
        # 仅重建来源回执；意外修改过的模型配置不能借刷新被悄悄接受。
        before = read(output/'config.json')
        for field in ('freetoken_gguf_source','freetoken_expert_source'):
            before.pop(field, None)
        comparable = copy.deepcopy(config)
        for field in ('freetoken_gguf_source','freetoken_expert_source'):
            comparable.pop(field, None)
        if before != comparable:
            raise ValueError('Checkpoint configuration changed; create a new checkpoint')
        for row in lock['metadata']['files']:
            if row['name'] != 'config.json':
                verify_file(output/row['name'], row)
    else:
        output.mkdir(parents=True)
        shutil.copyfile(metadata_directory/'config.json', output/'hf-config.original.json')
        for row in lock['metadata']['files']:
            if row['name'] != 'config.json':
                shutil.copyfile(metadata_directory/row['name'], output/row['name'])
    write(output/'config.json', config)
    # FT2 引擎要求绝对路径；可搬迁描述另存，refresh 在目标机器生成新的绝对回执。
    write(marker, dict(identity, sources=sources))
    return dict(status='refreshed' if refresh else 'prepared', checkpoint=str(output),
                weight_layout='gguf_strata', shards=2, mtp_enabled=False)


def verify_checkpoint(root, lock):
    """全文件重新验真，同时确认现有回执确实对应本机文件版本。"""
    output = layout(root, lock)['checkpoint']
    marker = read(output/'strata-q2-preparation.json')
    expected_marker = dict(format=1, model=lock['model'], gguf_revision=lock['gguf']['revision'],
                           metadata_revision=lock['metadata']['revision'])
    if {k:v for k,v in marker.items() if k != 'sources'} != expected_marker:
        raise ValueError('Checkpoint ownership/source identity changed')
    config = read(output/'config.json')
    original_identity = next(r for r in lock['metadata']['files'] if r['name'] == 'config.json')
    verify_file(output/'hf-config.original.json', original_identity)
    original = read(output/'hf-config.original.json')
    original.pop('quantization_config', None)
    original.pop('freetoken_expert_source', None)
    original.pop('freetoken_gguf_source', None)
    actual_base = copy.deepcopy(config)
    actual_base.pop('freetoken_gguf_source', None)
    actual_base.pop('freetoken_expert_source', None)
    if actual_base != original:
        raise ValueError('Checkpoint configuration differs from certified metadata')
    rows = config['freetoken_gguf_source']['shards']
    if len(rows) != 2:
        raise ValueError('Complete GGUF pair required')
    if config['freetoken_gguf_source'].get('format') != 'gguf_strata' or config['freetoken_gguf_source'].get('revision') != lock['gguf']['revision']:
        raise ValueError('Complete GGUF source identity differs')
    for row, expected in zip(rows, lock['gguf']['files']):
        if any(row[k] != expected[v] for k,v in (('filename','name'),('size','bytes'),('sha256','sha256'))):
            raise ValueError('GGUF receipt differs from source lock')
        if row['repo'] != lock['gguf']['repo'] or row['revision'] != lock['gguf']['revision']:
            raise ValueError('GGUF receipt repository/revision differs')
        portable = marker.get('sources', {})
        expected_path = ((output/portable['gguf']) if portable else layout(root,lock)['gguf'])/expected['name']
        if Path(row['path']).resolve() != expected_path.resolve():
            raise ValueError('Checkpoint moved; refresh absolute paths and local timestamps')
        stat = verify_file(output/row['path'], expected)
        if stat.st_mtime_ns != row['verified_mtime_ns']:
            raise ValueError('Verified bytes have new local timestamps; run refresh after relocation')
    first = rows[0]
    expert = config['freetoken_expert_source']
    if expert.get('format') != 'q2_0' or expert.get('scope') != 'routed_experts_only' or any(expert.get(k) != v for k,v in lock['expert_geometry'].items()):
        raise ValueError('Complete expert geometry differs from the certified source')
    if any(expert[k] != first[k] for k in ('path','size','sha256','repo','revision','verified_mtime_ns')):
        raise ValueError('Expert source differs from GGUF shard one')
    for row in lock['metadata']['files']:
        if row['name'] != 'config.json':
            verify_file(output/row['name'], row)
    return dict(status='verified', checkpoint=str(output), complete_gguf=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['plan','download','prepare','refresh','verify'])
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--gguf-dir', type=Path)
    parser.add_argument('--metadata-dir', type=Path)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--metadata-only', action='store_true')
    args = parser.parse_args()
    lock = read(LOCK)
    if not 1 <= args.workers <= 32:
        parser.error('workers must be between 1 and 32')
    if args.command == 'plan':
        result = plan(args.root, lock)
    elif args.command == 'download':
        result = download(args.root, lock, args.workers, args.metadata_only)
    elif args.command in ('prepare','refresh'):
        result = prepare(args.root, lock, args.gguf_dir, args.metadata_dir, args.command == 'refresh')
    else:
        result = verify_checkpoint(args.root, lock)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
