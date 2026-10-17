"""仅在真实安装目录的embedded Python内执行外部探针，并核对产品来源。"""
from pathlib import Path
import argparse
import hashlib
import json
import runpy
import sys
import tempfile


def normalized(path):
    """消除Windows长路径前缀后比较实际绝对路径。"""
    return Path(str(Path(path).resolve()).removeprefix('\\\\?\\'))


def main():
    """验证解释器、封装提交和模块loader，禁止回退到开发源码或venv。"""
    parser=argparse.ArgumentParser()
    parser.add_argument('--install-root',type=Path,required=True)
    parser.add_argument('--expected-commit',required=True)
    parser.add_argument('--origin-report',type=Path,required=True)
    parser.add_argument('--script',type=Path,required=True)
    args,forwarded=parser.parse_known_args()
    if args.origin_report.exists(): raise FileExistsError(args.origin_report)
    root=normalized(args.install_root)
    if normalized(sys.executable)!=root/'python/python.exe':
        raise RuntimeError(f'Not installed Python: {sys.executable}')
    import silitok_speed as product
    origin=normalized(product.__spec__.origin)
    if origin!=root/'python/Lib/site-packages/silitok_speed/__init__.pyd':
        raise RuntimeError(f'Not installed Speed: {origin}')
    manifest=product._sp_manifest
    if manifest['build_commit']!=args.expected_commit or manifest['build_dirty']:
        raise RuntimeError('Installed source identity differs from accepted clean commit')
    bundle=origin.read_bytes()
    if bundle[-16:]!=b'SPEED-BUNDLE-V2!': raise RuntimeError('Invalid installed bundle trailer')
    length=int.from_bytes(bundle[-24:-16],'little')
    digest=hashlib.sha256(bundle[-24-length:-24]).hexdigest()
    cache=normalized(Path(tempfile.gettempdir())/'siltok-speed-assets'/digest)
    record=dict(status='running',python=sys.executable,speed_origin=str(origin),
                build_commit=manifest['build_commit'],build_dirty=manifest['build_dirty'],
                build_version=manifest['build_version'],bundle_sha256=hashlib.sha256(bundle).hexdigest(),
                bundle_cache=str(cache),script=str(args.script),arguments=forwarded,
                scope='compiled_staged_payload_not_installer')
    print(json.dumps(record),flush=True)
    try:
        sys.argv=[str(args.script),*forwarded]
        try:
            runpy.run_path(str(args.script),run_name='__main__')
        except SystemExit as error:
            if error.code not in (None,0): raise
        modules={name:module for name,module in sys.modules.items()
                 if name=='silitok_speed' or name.startswith('silitok_speed.')}
        invalid={name:str(getattr(module,'__file__','')) for name,module in modules.items()
                 if name!='silitok_speed' and
                 getattr(getattr(module,'__spec__',None),'loader',None) is not product._sp_loader}
        if invalid: raise RuntimeError(f'Product modules escaped protected installed loader: {invalid}')
        record.update(status='passed',product_modules={name:str(getattr(module,'__file__',''))
                                                      for name,module in sorted(modules.items())})
    except BaseException as error:
        record.update(status='failed',error=repr(error))
        raise
    finally:
        args.origin_report.write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf-8')


if __name__=='__main__':
    main()
