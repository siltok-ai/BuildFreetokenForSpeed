"""接收者验收入口：仅注入进程级测试授权，按安装版正常 release_profile 启动。"""
from pathlib import Path
import argparse
import importlib.util
import json
import os
import sys
import threading
import time
from unittest.mock import patch

import silitok_speed
import uvicorn

p=argparse.ArgumentParser()
p.add_argument('--stop-file',type=Path,required=True)
p.add_argument('--identity',type=Path,required=True)
p.add_argument('--config-root',type=Path,default=Path(r'C:\ProgramData\Siltok\Speed\config'))
p.add_argument('--runtime-dir',type=Path,default=Path(r'C:\ProgramData\Siltok\Speed\runtime'))
args=p.parse_args()
spec=importlib.util.spec_from_file_location('receiver_test_license',Path(__file__).with_name('receiver-test-license.py'))
fixture=importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixture)

def run(app,**kwargs):
    """保留真实服务配置和生命周期，以停止文件触发正常 uvicorn shutdown。"""
    from silitok_speed.config import get_config
    import importlib.metadata
    cfg=get_config()
    args.identity.write_text(json.dumps(dict(python=sys.executable,product=silitok_speed.__file__,
        version=silitok_speed.__version__,freetoken=importlib.metadata.version('freetoken'),
        license_mode='test_stub',device_license_validated=False,sys_path=sys.path,
        models_dir=cfg.models_dir,resource=cfg.resource.model_dump(),host=kwargs['host'],port=kwargs['port'],
        kernel_cache=os.environ['SILITOK_FREETOKEN_KERNEL_CACHE_DIR']),indent=2)+'\n')
    server=uvicorn.Server(uvicorn.Config(app,**kwargs))
    def stop():
        while not args.stop_file.exists() and not server.should_exit:
            time.sleep(.2)
        server.should_exit=True
    threading.Thread(target=stop,daemon=True).start()
    server.run()

with fixture.test_license_monitor(),patch.object(uvicorn,'run',run):
    from silitok_speed.release_profile import main
    raise SystemExit(main(['serve','--config-root',str(args.config_root),
        '--runtime-dir',str(args.runtime_dir)]))
