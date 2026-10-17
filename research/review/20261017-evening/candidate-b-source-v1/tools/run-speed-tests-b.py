"""Run selected application tests with explicit source identity and a process-only license fixture."""
from pathlib import Path
import argparse
import json
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--source', type=Path, required=True)
parser.add_argument('--license-fixture', action='store_true')
parser.add_argument('--report', type=Path)
parser.add_argument('pytest_args', nargs=argparse.REMAINDER)
args = parser.parse_args()
if args.report and args.report.exists():
    raise FileExistsError(args.report)
sys.path.insert(0, str(args.source))
import silitok_speed
if not Path(silitok_speed.__file__).resolve().is_relative_to(args.source.resolve()):
    raise RuntimeError(silitok_speed.__file__)
if args.license_fixture:
    from silitok_speed.engine import Engine
    Engine.require_license = lambda self: None
    Engine.is_licensed = property(lambda self: True)
    from silitok_speed import discovery
    discovery.ModelRegistry.require_license = lambda self: None

print(json.dumps(dict(source=silitok_speed.__file__, python=sys.executable,
                      process_only_license_fixture=args.license_fixture)), flush=True)
import pytest
pytest_args = args.pytest_args
if pytest_args[:1] == ['--']:
    pytest_args = pytest_args[1:]
code = pytest.main(['-q', '--tb=short', '-rs', '-p', 'no:cacheprovider', *pytest_args])
if args.report:
    args.report.write_text(json.dumps(dict(status='passed' if code == 0 else 'failed', exit_code=int(code),
        source=silitok_speed.__file__, python=sys.executable, pytest_args=pytest_args,
        process_only_license_fixture=args.license_fixture), indent=2)+'\n', encoding='utf-8')
raise SystemExit(code)
