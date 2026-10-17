"""在推理前记录被测代码和材料字节，不创建CUDA上下文。"""
from pathlib import Path
import hashlib
import importlib.metadata as metadata
import json
import os
import subprocess


def digest_files(root, paths):
    """按相对路径固定文件身份，聚合摘要也包含文件名，避免相同字节错位。"""
    inventory = {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                 for p in sorted(paths)}
    aggregate = hashlib.sha256(json.dumps(inventory, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    return dict(sha256=aggregate, files=inventory)


def capture(speed_origin=None):
    """核对当前解释器实际安装内容；Speed源码、编译包和FreeToken材料分别留证。"""
    dist = metadata.distribution('freetoken')
    root = Path(dist.locate_file('')).resolve()
    files = [Path(dist.locate_file(entry)).resolve() for entry in dist.files
             if str(entry).startswith('freetoken/') and Path(str(entry)).suffix in {'.py', '.pyd'}]
    cache = Path(os.environ['FREETOKEN_KERNEL_CACHE_DIR']).resolve(strict=True)
    dlls = list(cache.rglob('*.dll'))
    if len(dlls) != 27:
        raise ValueError(f'Expected the paired 27-DLL cache, found {len(dlls)}')
    result = dict(freetoken_version=dist.version,
        freetoken_installed_code=digest_files(root, files),
        kernel_cache=str(cache), aot_dlls=digest_files(cache, dlls),
        freetoken_disable_jit=os.environ.get('FREETOKEN_DISABLE_JIT'),
        packages={name: metadata.version(name) for name in
                  ('torch', 'triton-windows', 'apache-tvm-ffi', 'transformers', 'numpy')})
    if speed_origin:
        origin = Path(speed_origin).resolve(strict=True)
        result['speed_origin'] = str(origin)
        if origin.suffix.lower() == '.pyd':
            result['speed_bundle_sha256'] = hashlib.sha256(origin.read_bytes()).hexdigest()
        else:
            package = origin.parent
            result['speed_python'] = digest_files(package, [p for p in package.rglob('*.py') if not any(part.startswith('.') or part in {'build', '__pycache__'} or part.endswith('.egg-info') for part in p.relative_to(package).parts)])
            result['speed_head'] = subprocess.check_output(['git', '-C', str(package.parent),
                                                           'rev-parse', 'HEAD'], text=True).strip()
            diff = subprocess.check_output(['git', '-C', str(package.parent), 'diff', 'HEAD', '--binary'])
            result['speed_worktree_diff_sha256'] = hashlib.sha256(diff).hexdigest()
    return result
