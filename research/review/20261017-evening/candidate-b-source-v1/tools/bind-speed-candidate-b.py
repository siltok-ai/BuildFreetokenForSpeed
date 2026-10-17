"""Bind the final native candidate to Speed without changing historical candidate-A inputs."""
from pathlib import Path
import hashlib
import json
import shutil
import subprocess

SPEED=Path(r'C:\workspace\wt\siltok-speed\flash-perf')
BUILD=Path(r'C:\workspace\wt\freetoken-mats\flash-perf')
SUPPLIER='705ce5c226ac963e9d7c111585ee18b2d3cc24ab'
if subprocess.check_output(['git','-C',str(BUILD),'rev-parse','HEAD'],text=True).strip()!=SUPPLIER:
    raise RuntimeError('Supplier changed before binding')
materials=BUILD/'artifacts/strataq2review2b-cp312-cu130-sm120'
record=json.loads((materials/'build-record.json').read_text())
wheel=materials/'wheels'/record['wheel_filename']
if hashlib.sha256(wheel.read_bytes()).hexdigest()!=record['wheel_sha256']:
    raise ValueError('Supplier wheel changed')
path=SPEED/'silitok_speed/qwen36/release.json'
release=json.loads(path.read_text())
release.update(version=record['custom_version'],producer_commit=SUPPLIER,
               consumer_contract='integration/speed/consumer-contract-strataq2review2b.json')
release['wheel'].update(path='wheels/'+record['wheel_filename'],sha256=record['wheel_sha256'])
release['kernel_cache']['package_path']='kernel_cache_strataq2review2b'
cache=SPEED/'silitok_speed/qwen36'/release['kernel_cache']['package_path']
if cache.exists():
    raise FileExistsError(cache)
project=SPEED/'silitok_speed/pyproject.toml'
text=project.read_text(encoding='utf-8')
if 'strataq2review2a' not in text:
    raise RuntimeError('Unexpected existing product version')
shutil.copytree(materials/'kernel-cache',cache)
path.write_text(json.dumps(release,indent=2)+'\n',encoding='utf-8',newline='\r\n')
project.write_text(text.replace('strataq2review2a','strataq2review2b'),encoding='utf-8',newline='\r\n')
print(json.dumps(dict(version=record['custom_version'],supplier=SUPPLIER,
    scope='candidate_binding_source_consumer_acceptance_pending')))
