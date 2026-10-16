"""在独立验收进程中调用正式TCP性能工具；不修改产品授权或安装文件。"""
from pathlib import Path
import argparse
import json
import runpy
import sys


def main():
    """先固定被测包身份，再应用已授权的进程内验收替身。"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--compiled', action='store_true')
    args, forwarded = parser.parse_known_args()
    if not args.compiled:
        sys.path.insert(0, str(args.source.resolve(strict=True)))
    import silitok_speed
    origin = str(silitok_speed.__spec__.origin)
    if args.compiled and not origin.lower().endswith('.pyd'):
        raise RuntimeError(f'Compiled acceptance imported source: {origin}')
    if not args.compiled and Path(origin).resolve().parent != args.source.resolve()/'silitok_speed':
        raise RuntimeError(f'Unexpected source package: {origin}')
    from silitok_speed.engine import Engine
    from silitok_speed import discovery
    Engine.require_license = lambda self: None
    Engine.is_licensed = property(lambda self: True)
    if hasattr(discovery, 'ModelRegistry'):
        discovery.ModelRegistry.require_license = lambda self: None
    print(json.dumps(dict(python=sys.executable, speed_origin=origin, compiled=args.compiled,
                          process_only_license_test_bypass=True)), flush=True)
    script = args.source/'tools/bench/benchmark_freetoken_http.py'
    sys.argv = [str(script), *forwarded]
    runpy.run_path(str(script), run_name='__main__')


if __name__ == '__main__':
    main()
