"""用当前明确的 Windows 工具链单独编译 PLE 行存储扩展并保存真实回执。"""
import argparse
import hashlib
import json
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
        raise RuntimeError('Row-store producer requires CPython3.12 Windows x64')
    import torch
    import torch.utils.cpp_extension as extension
    extension.SUBPROCESS_DECODE_ARGS=('utf-8','replace')
    if torch.__version__!=args.torch:raise RuntimeError('Row-store compiler Torch differs from profile')
    output.mkdir(parents=True)
    build=output/'ninja'
    build.mkdir()
    flags=['/utf-8','/O2','/std:c++20','/DNOMINMAX','/DWIN32_LEAN_AND_MEAN']
    native=extension.load(name='_row_store',sources=[str(source)],extra_cflags=flags,
        build_directory=str(build),with_cuda=False,verbose=True)
    binary=Path(native.__file__).resolve(strict=True)
    if not binary.is_relative_to(output):raise ValueError('Compiled module escapes its output')
    compiler=subprocess.run(['cl'],capture_output=True)
    digest=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
    receipt=dict(format=1,module='_row_store',build_python=sys.executable,
        python=sys.version,torch=torch.__version__,torch_cuda=torch.version.cuda,
        source_sha256=digest(source),source=str(source),cflags=flags,
        msvc_compiler=(compiler.stdout+compiler.stderr).decode('utf-8',errors='replace'),
        build_ninja_sha256=digest(build/'build.ninja'),
        artifact=dict(path=binary.relative_to(output).as_posix(),
            bytes=binary.stat().st_size,sha256=digest(binary)))
    (output/'build-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(status='compiled_not_runtime_accepted',artifact=receipt['artifact'])))


if __name__=='__main__':main()
