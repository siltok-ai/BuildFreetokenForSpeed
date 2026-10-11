"""Append completed runtime acceptance without replacing the earlier rejected runs."""
import hashlib
import json
from pathlib import Path
import shutil

root = Path(__file__).parent
build = Path(r'C:\workspace\worktrees\build-freetoken-q2-material-review')
speed = Path(r'C:\workspace\worktrees\speed-q2-material-review')
materials = build/'artifacts/q2opt2-cp312-cu130-sm120'
target = materials/'validation/runtime-review-20261011'
target.mkdir(exist_ok=False)
old = json.loads((materials/'validation/review-20261011/validation-summary.json').read_text(encoding='utf-8'))
for name in ('packaged-flash-backend-capacity-002', 'packaged-flash-http-49p5-002',
             'packaged-flash-http-default'):
    shutil.copytree(root/name, target/name)
for name in ('run_backend_capacity.py', 'run_http_capacity.py', 'run_http_default_capacity.py',
             'analyze_commit.py', 'commit-measurements.json'):
    shutil.copyfile(root/name, target/name)
identity = json.loads((target/'packaged-flash-http-default/identity.json').read_text(encoding='utf-8'))
modules = identity['build_manifest']['modules']
mismatches = [name for name, info in modules.items()
              if hashlib.sha256((speed/'silitok_speed'/name).read_bytes()).hexdigest() != info['source_sha256']]
assert not mismatches
summary = dict(old)
summary.update(speed_commit='f66e44361e58672b56c28cbd3c2cfc6b60dd5bfb',
               materials_code_commit='19615a6cd2e287aa67c08e5962964c479a3a62f0',
               compiled_module_count=len(modules), compiled_source_mismatches=mismatches,
               packaged_speed_validated=True, normal_full_flash_next_validated=True,
               packaged_flash_next_runs_passed=3, packaged_default_http_validated=True,
               flash_next_pending_reason=None,
               previous_partial_evidence='../review-20261011/validation-summary.json',
               commit_measurements=json.loads((target/'commit-measurements.json').read_text()),
               recommended_model_commit_budget_gib=50, test_commit_safety_gib=1,
               default_budget_lowered=False,
               budget_decision='49.5 GiB HTTP passed but observed 49.78 GiB increment exceeds its estimate. Default 50 GiB covers the observed 49.79 GiB HTTP increment; no material reduction supported.',
               scope='New compiled package, all three models on this Windows host. Full Flash Next default-budget HTTP and complete backend passed. Installer and clean-machine acceptance excluded.')
(target/'validation-summary.json').write_text(json.dumps(summary, indent=2)+'\n', encoding='utf-8')
files = {path.relative_to(target).as_posix():
         {'bytes':path.stat().st_size,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
         for path in target.rglob('*') if path.is_file()}
(target/'evidence.json').write_text(json.dumps({'files':files}, indent=2)+'\n', encoding='utf-8')
contract_path = build/'integration/speed/consumer-contract-q2opt2.json'
contract = json.loads(contract_path.read_text(encoding='utf-8'))
contract.update(validation_status='validated_compiled_runtime_on_original_windows_host',
                packaged_speed_validated=True, normal_full_flash_next_validated=True,
                speed_source_identity=(target/'validation-summary.json').relative_to(build).as_posix(),
                packaged_speed_validation_scope=summary['scope'])
contract['speed_adapter']['repository'] = 'https://github.com/siltok-ai/siltok-speed'
contract_path.write_text(json.dumps(contract, indent=2)+'\n', encoding='utf-8')
print(json.dumps({'evidence_files':len(files),'compiled_modules':len(modules),'mismatches':mismatches}))
