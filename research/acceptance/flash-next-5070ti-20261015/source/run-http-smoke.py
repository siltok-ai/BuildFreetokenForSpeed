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
p.add_argument('--lifecycle-output',type=Path,required=True)
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
    from silitok_speed import discovery
    # 用户明确授权的临时验收绕过；不修改产品文件和原生授权模块。
    Engine.require_license=lambda self: None
    Engine.is_licensed=property(lambda self: True)
    if hasattr(discovery,'ModelRegistry'):
        discovery.ModelRegistry.require_license=lambda self: None
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
            self.engine=app.state.engine
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
        def exercise_lifecycle(self):
            """实际断开TCP流，再验证同worker复用及完整卸载后按请求重载。"""
            import psutil
            catalog=self.client.get('/v1/models')
            catalog.raise_for_status()
            model_id=catalog.json()['data'][0]['id']
            payload=dict(model=model_id,enable_thinking=False,temperature=0,top_p=1,max_tokens=32,
                         messages=[dict(role='user',content='计算31乘17。只输出一个整数。')])
            def arithmetic():
                response=self.client.post('/v1/chat/completions',json=payload)
                response.raise_for_status()
                body=response.json()
                if body['choices'][0]['message']['content'].strip()!='527':
                    raise RuntimeError('Lifecycle arithmetic changed')
                return body
            before=arithmetic()
            original=module._worker_children(self.engine,model_id)
            observed=[]
            with self.client.stream('POST','/v1/chat/completions',json=payload | dict(stream=True,
                    max_tokens=512,messages=[dict(role='user',content='Write a detailed numbered guide with 50 steps for building a local inference service.')])) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if line.startswith('data: ') and line!='data: [DONE]':
                        chunk=json.loads(line[6:])
                        if any(choice.get('delta',{}).get('content') or
                               choice.get('delta',{}).get('reasoning_content')
                               for choice in chunk.get('choices',[])):
                            observed.append(chunk)
                            break
            if not observed: raise RuntimeError('TCP disconnect probe produced no actual token')
            resumed=arithmetic()
            reused=module._worker_children(self.engine,model_id)
            if reused!=original: raise RuntimeError('TCP cancellation replaced the worker')
            binding=self.engine.model_mgr.resolve(model_id)
            self.engine.backend_pool.evict(binding.spec)
            if any(psutil.pid_exists(pid) for pid in original):
                raise RuntimeError('Full evict left an original worker process')
            reloaded=arithmetic()
            replacement=module._worker_children(self.engine,model_id)
            if not replacement or replacement & original:
                raise RuntimeError('HTTP request did not create a fresh worker after evict')
            self.engine.backend_pool.evict(binding.spec)
            if any(psutil.pid_exists(pid) for pid in replacement):
                raise RuntimeError('Reloaded worker did not exit')
            result=dict(status='passed',transport='tcp',original_worker_pids=sorted(original),
                replacement_worker_pids=sorted(replacement),disconnect_after_content_chunks=len(observed),
                same_worker_after_cancel=True,original_exited_before_reload=True,replacement_exited=True,
                before=before,after_cancel=resumed,after_reload=reloaded)
            if args.lifecycle_output.exists(): raise FileExistsError(args.lifecycle_output)
            args.lifecycle_output.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')

        def __exit__(self,*exc):
            try:
                if exc[0] is None:
                    self.exercise_lifecycle()
            finally:
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
