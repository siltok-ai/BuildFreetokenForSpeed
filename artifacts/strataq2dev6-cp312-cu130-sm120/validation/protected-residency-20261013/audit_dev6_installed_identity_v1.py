"""Bind installed Python, ranking and native bytes to the frozen derived wheel."""
import hashlib
import json
from pathlib import Path
import sys
import zipfile

import freetoken

ROOT = Path(__file__).resolve().parent
MATERIALS = ROOT / 'dev6-derived-native-base-v1'
SOURCE = ROOT / 'dev6-frozen-replay-v1/build-source/python'
installed = Path(freetoken.__file__).parent.parent
record = json.loads((MATERIALS / 'build-record.json').read_text())
checks = {}
with zipfile.ZipFile(MATERIALS / 'wheels' / record['wheel_filename']) as wheel:
    for name in wheel.namelist():
        if name.startswith('freetoken/') and (name.endswith('.py') or '/moe/configs/' in name or name.endswith('.pyd')):
            data = wheel.read(name)
            if (installed / name).read_bytes() != data:
                raise RuntimeError('Installed material differs: ' + name)
            if not name.endswith('.pyd') and (SOURCE / name).read_bytes() != data:
                raise RuntimeError('Frozen source differs: ' + name)
            checks[name] = hashlib.sha256(data).hexdigest()
if freetoken.__version__ != record['custom_version']:
    raise RuntimeError('Installed version differs')
target = ROOT / 'dev6-installed-identity-v1.json'
if target.exists():
    raise FileExistsError(target)
target.write_text(json.dumps(dict(passed=True, executable=sys.executable,
    version=freetoken.__version__, wheel_sha256=record['wheel_sha256'],
    build_record_sha256=hashlib.sha256((MATERIALS/'build-record.json').read_bytes()).hexdigest(),
    checked_files=checks, runtime_acceptance=False), indent=2), encoding='utf-8')
print(json.dumps(dict(passed=True, version=freetoken.__version__, checked_files=len(checks))))
