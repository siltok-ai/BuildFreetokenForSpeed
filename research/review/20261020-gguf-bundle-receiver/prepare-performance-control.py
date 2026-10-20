"""用相同的 GGUF 派生元数据构造 FT9 对照；不恢复任何 RadixArk 文件。"""
from pathlib import Path
import json
import shutil
import _winapi

package = Path(r'C:\workspace\flash-next move-test')
root = Path(r'C:\workspace\cfg\siltok-speed\flash-portable-control')
root.mkdir(parents=True)
source = package/'checkpoints/flash-next-strata-q2'
target = root/'checkpoints/flash-next-strata-q2'
target.mkdir(parents=True)
for path in source.iterdir():
    if path.is_file(): shutil.copy2(path,target/path.name)
lock = json.loads(Path(r'C:\workspace\wt\freetoken-mats\flash-perf\model-weights\qwen3.8-flash-next-gguf-q2\source-lock.json').read_text())
config = json.loads((target/'config.json').read_text())
config['freetoken_gguf_source']['revision'] = lock['gguf']['revision']
for row,identity in zip(config['freetoken_gguf_source']['shards'],lock['gguf']['files']):
    path = (source/row['path']).resolve()
    row.update(path=str(path),sha256=identity['sha256'],verified_mtime_ns=path.stat().st_mtime_ns,
        repo=lock['gguf']['repo'],revision=lock['gguf']['revision'])
first = config['freetoken_gguf_source']['shards'][0]
config['freetoken_expert_source'].update({key:first[key] for key in
    ('path','size','sha256','repo','revision','verified_mtime_ns')})
(target/'config.json').write_text(json.dumps(config,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
_winapi.CreateJunction(str(package/'checkpoints/flash-next-mtp'),str(root/'checkpoints/flash-next-mtp'))
shutil.copy2(package/'manifest.json',root/'manifest.json')
print(json.dumps(dict(root=str(root),metadata_source='same GGUF-derived tokenizer/template/config as candidate',
    legacy_receipts='Only old FT9 loader input; not a new Speed check',radix_restored=False)))
