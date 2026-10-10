"""仅获取锁定的 HF 非专家文件与完整 Q2_0 专家第一片。"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def verify(path: Path, entry: dict) -> None:
    """重新读取实际文件核验尺寸与 SHA，来源声明不能代替字节校验。"""
    if path.stat().st_size != entry['bytes']:
        raise ValueError(f'file size mismatch: {path}')
    with path.open('rb') as stream:
        actual = hashlib.file_digest(stream, 'sha256').hexdigest()
    if actual != entry['sha256']:
        raise ValueError(f'file SHA256 mismatch: {path}')


def main() -> None:
    """输出下载计划或按固定 revision 下载；不触碰生产模型和安装环境。"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    lock = json.loads((ROOT/'source-lock.json').read_text(encoding='utf-8'))
    hf = lock['direct_sources']['hf']
    expert = lock['direct_sources']['experts']
    plan = {'hf_repo':hf['repo'], 'hf_revision':hf['revision'],
            'hf_files':[e['name'] for e in hf['files']],
            'expert_repo':expert['repo'], 'expert_revision':expert['revision'],
            'expert_file':expert['filename'], 'total_bytes':lock['source_payload_bytes'],
            'output':str(args.output.resolve())}
    print(json.dumps(plan,ensure_ascii=False,indent=2),flush=True)
    if args.dry_run:
        return
    from huggingface_hub import hf_hub_download

    trunk = args.output/'hf'
    for entry in hf['files']:
        path = Path(hf_hub_download(repo_id=hf['repo'],revision=hf['revision'],
                                  filename=entry['name'],local_dir=str(trunk),token=False))
        verify(path,entry)
        print('verified HF file: '+entry['name'],flush=True)
    # 复用已经固定第一片尺寸/SHA的下载器，不接受任意量化级别或 master 回退。
    subprocess.run([sys.executable,'-B',str(ROOT.parents[1]/'tools/download_q2_experts.py'),
                    '--output',str(args.output/'experts'),'--source','huggingface',
                    '--transport','hub'],check=True)
    verify(args.output/'experts'/expert['filename'],
           {'bytes':expert['bytes'],'sha256':expert['sha256']})
    print('all locked source files verified',flush=True)


if __name__ == '__main__':
    main()
