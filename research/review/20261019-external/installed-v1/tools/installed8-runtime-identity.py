"""Record actual installed wheel and AOT bytes without allocating a model."""
from pathlib import Path
import argparse,json,runpy,importlib.util
import silitok_speed
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
assert str(silitok_speed.__spec__.origin).lower().endswith('.pyd')
assert not a.output.exists()
assert importlib.util.find_spec('silitok_speed.qwen36') is None
assert not any(name.startswith('qwen36/') for name in silitok_speed._sp_manifest['resources'])
assert not any(name.startswith('runtimes/freetoken/') and name.endswith('.dll') for name in silitok_speed._sp_manifest['resources'])
capture=runpy.run_path(str(Path(__file__).with_name('run-identity.py')))['capture']
d=capture(silitok_speed.__spec__.origin)
d.update(status='passed',scope='actual_installed_payload_not_clean_os')
a.output.write_text(json.dumps(d,indent=2)+'\n',encoding='utf-8')
