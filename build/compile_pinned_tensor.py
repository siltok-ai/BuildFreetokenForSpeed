"""用当前明确的 Windows 工具链单独编译 CUDA 主存与工作区扩展并保存真实回执。"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    """只接收已重放的单个源码，在新目录编译，不读取或替换旧材料。"""
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--torch',required=True)
    args=parser.parse_args()
    source=args.source.resolve(strict=True)
    output=args.output.resolve()
    if output.exists():raise FileExistsError(output)
    if sys.platform!='win32' or sys.version_info[:2]!=(3,12) or sys.maxsize<=2**32:
        raise RuntimeError('Pinned-tensor producer requires CPython3.12 Windows x64')
    import torch
    import torch.utils.cpp_extension as extension
    extension.SUBPROCESS_DECODE_ARGS=('utf-8','replace')
    if torch.__version__!=args.torch:raise RuntimeError('Pinned-tensor compiler Torch differs from profile')
    output.mkdir(parents=True)
    build=output/'ninja'
    build.mkdir()
    flags=['/utf-8','/O2','/std:c++20','/DNOMINMAX','/DWIN32_LEAN_AND_MEAN']
    cuda=Path(os.environ['CUDA_PATH']).resolve(strict=True)
    includes=[str(source.parent/'include'),str(cuda/'include')]
    ldflags=['/LIBPATH:'+str(cuda/'lib/x64'),'cudart.lib']
    native=extension.load(name='_pinned_tensor',sources=[str(source)],extra_cflags=flags,
        extra_include_paths=includes,extra_ldflags=ldflags,
        build_directory=str(build),with_cuda=False,verbose=True)
    binary=Path(native.__file__).resolve(strict=True)
    if not binary.is_relative_to(output):raise ValueError('Compiled module escapes its output')
    compiler=subprocess.run(['cl'],capture_output=True)
    digest=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
    receipt=dict(format=1,module='_pinned_tensor',build_python=sys.executable,
        python=sys.version,torch=torch.__version__,torch_cuda=torch.version.cuda,
        source_sha256=digest(source),source=str(source),cflags=flags,include_paths=includes,ldflags=ldflags,
        hip_compat_sha256=digest(source.parent/'include/freetoken/hip_compat.h'),
        msvc_compiler=(compiler.stdout+compiler.stderr).decode('utf-8',errors='replace'),
        build_ninja_sha256=digest(build/'build.ninja'),
        artifact=dict(path=binary.relative_to(output).as_posix(),
            bytes=binary.stat().st_size,sha256=digest(binary)))
    (output/'build-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(status='compiled_not_runtime_accepted',artifact=receipt['artifact'])))


if __name__=='__main__':main()
