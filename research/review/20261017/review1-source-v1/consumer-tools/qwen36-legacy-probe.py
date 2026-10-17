"""执行实际 Speed；许可证临时绕过仅存在于此独立验收进程。"""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys

p=argparse.ArgumentParser()
p.add_argument('--mode',choices=['backend','http'],required=True)
p.add_argument('--source',type=Path,required=True)
p.add_argument('--packaged',action='store_true')
p.add_argument('--tcp',action='store_true')
args,forward=p.parse_known_args()
if not args.packaged:
    sys.path.insert(0,str(args.source))
import silitok_speed
print(json.dumps({'speed_origin':silitok_speed.__spec__.origin,'packaged':args.packaged,
                  'license_test_bypass':args.mode=='http'},ensure_ascii=False),flush=True)
if args.packaged and not str(silitok_speed.__spec__.origin).endswith('.pyd'):
    raise RuntimeError('packaged acceptance imported source Speed')
if args.mode=='http':
    from silitok_speed.engine import Engine
    from silitok_speed.discovery import ModelRegistry
    # 用户明确授权的临时验收绕过；不修改产品文件和原生授权模块。
    Engine.require_license=lambda self: None
    Engine.is_licensed=property(lambda self: True)
    ModelRegistry.require_license=lambda self: None
tool=args.source/'tools'/('bench/benchmark_freetoken_qwen36.py' if args.mode=='backend' else 'smoke/freetoken_qwen36.py')
spec=importlib.util.spec_from_file_location('qwen_acceptance',tool)
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
if args.tcp:
    import socket
    import threading
    import time
    import httpx
    import uvicorn
    if args.mode != 'http': raise ValueError('--tcp requires HTTP mode')
    class NetworkClient:
        """只在验收脚本中将 TestClient 替换为随机本地端口的真实 Uvicorn。"""
        def __init__(self, app):
            self.sock=socket.socket()
            self.sock.bind(('127.0.0.1',0));self.sock.listen(128)
            self.port=self.sock.getsockname()[1]
            self.server=uvicorn.Server(uvicorn.Config(app,host='127.0.0.1',port=self.port,access_log=False,log_level='warning'))
        def __enter__(self):
            self.thread=threading.Thread(target=self.server.run,kwargs={'sockets':[self.sock]},daemon=True)
            self.thread.start()
            deadline=time.monotonic()+30
            while not self.server.started:
                if not self.thread.is_alive() or time.monotonic()>deadline: raise RuntimeError('TCP server failed to start')
                time.sleep(.05)
            print(json.dumps({'http_transport':'tcp','host':'127.0.0.1','port':self.port}),flush=True)
            self.client=httpx.Client(base_url=f'http://127.0.0.1:{self.port}',timeout=1800,trust_env=False)
            return self.client
        def __exit__(self,*exc):
            self.client.close();self.server.should_exit=True;self.thread.join(45);self.sock.close()
            if self.thread.is_alive(): raise RuntimeError('TCP server did not shut down')
    module.TestClient=NetworkClient
if args.mode=='backend':
    cases=[dict(case,max_tokens=2048 if case['enable_thinking'] else case['max_tokens']) for case in module.QUALITY_CASES]
    cases.extend([
      {'id':'chinese','prompt':'用中文简要解释为什么本地大模型推理需要管理显存。','enable_thinking':False,'max_tokens':256},
      {'id':'json','prompt':'只输出一个合法JSON对象，包含name为张三、city为上海、age为28。','enable_thinking':False,'max_tokens':128},
      {'id':'python','prompt':'只输出Python函数factorial(n)的代码，计算非负整数的阶乘，0的阶乘为1。','enable_thinking':False,'max_tokens':256}
    ])
    module.QUALITY_CASES=tuple(cases)
sys.argv=[str(tool),*forward]
raise SystemExit(module.main())
