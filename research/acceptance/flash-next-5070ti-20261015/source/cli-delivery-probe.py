"""运行未注入替身的正式CLI与生成配置，使用真实TCP并验证正常退出。"""
from pathlib import Path
import argparse
import ctypes
import json
import os
import socket
import subprocess
import sys
import time

import httpx
import psutil


def stop_console(pid):
    """仅附加本轮子进程的隐藏控制台，向它发送正常Ctrl+C退出请求。"""
    code = '''import ctypes,sys,time
k=ctypes.WinDLL('kernel32',use_last_error=True)
k.FreeConsole()
if not k.AttachConsole(int(sys.argv[1])): raise ctypes.WinError(ctypes.get_last_error())
if not k.SetConsoleCtrlHandler(None,True): raise ctypes.WinError(ctypes.get_last_error())
if not k.GenerateConsoleCtrlEvent(0,0): raise ctypes.WinError(ctypes.get_last_error())
time.sleep(.5)
k.FreeConsole()
'''
    subprocess.run([sys.executable, '-B', '-c', code, str(pid)], check=True,
                   creationflags=subprocess.CREATE_NO_WINDOW, timeout=10)


def main():
    """只启动单个真实服务，保留配置身份与HTTP结果，失败也收回本轮进程树。"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--deployment', type=Path, required=True)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0)); port = sock.getsockname()[1]
    environment = os.environ.copy()
    environment['SILITOK_CONFIG'] = str(args.deployment/'speed-config')
    runroot = args.output.with_suffix('')
    command = [sys.executable, '-u', '-B', '-m', 'silitok_speed.cli', 'serve',
               '--models-dir', str(args.deployment), '--gpu-resource-probe', 'nvml',
               '--host', '127.0.0.1', '--port', str(port),
               '--log-dir', str(runroot/'logs'), '--result-dir', str(runroot/'results'),
               '--task-runtime-dir', str(runroot/'runtime')]
    info = subprocess.STARTUPINFO()
    info.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    info.wShowWindow = 0
    report = dict(status='running', command=command, cwd=str(args.source),
                  deployment=str(args.deployment), test_bypass=False,
                  config=json.loads((args.deployment/'speed-config/config.yaml').read_text()),
                  checkpoint_target=str((args.deployment/'checkpoints').resolve()))
    owned = []
    with args.output.with_suffix('.cli.log').open('wb') as log:
        proc = subprocess.Popen(command, cwd=args.source, env=environment, stdout=log,
                                stderr=subprocess.STDOUT, startupinfo=info,
                                creationflags=subprocess.CREATE_NEW_CONSOLE)
        report['pid'] = proc.pid
        try:
            with httpx.Client(base_url=f'http://127.0.0.1:{port}', timeout=600) as client:
                deadline = time.monotonic()+90
                while True:
                    if proc.poll() is not None:
                        raise RuntimeError(f'CLI startup exit {proc.returncode}')
                    try:
                        response = client.get('/health', timeout=2)
                    except httpx.TransportError:
                        if time.monotonic() >= deadline:
                            raise RuntimeError('CLI startup timeout')
                        time.sleep(.5)
                        continue
                    response.raise_for_status(); report['health'] = response.json(); break
                response = client.get('/v1/models'); response.raise_for_status()
                report['models'] = response.json()
                model = report['models']['data'][0]['id']
                payload = dict(model=model, temperature=0, top_p=1, enable_thinking=False,
                               max_tokens=32, messages=[dict(role='user', content='计算31乘17。只输出一个整数。')])
                response = client.post('/v1/chat/completions', json=payload)
                response.raise_for_status(); report['arithmetic'] = response.json()
                if report['arithmetic']['choices'][0]['message']['content'].strip() != '527':
                    raise RuntimeError('CLI arithmetic mismatch')
                text=[]; done=0; usage=None
                with client.stream('POST', '/v1/chat/completions', json=payload | dict(stream=True,
                                   stream_options=dict(include_usage=True))) as response:
                    response.raise_for_status()
                    for line in response.iter_lines():
                        if not line.startswith('data: '): continue
                        if line == 'data: [DONE]': done += 1; continue
                        body=json.loads(line[6:])
                        if body.get('usage'): usage=body['usage']
                        for choice in body.get('choices', []):
                            text.append(choice.get('delta', {}).get('content') or '')
                report['sse'] = dict(text=''.join(text), done_events=done, usage=usage)
                if ''.join(text).strip() != '527' or done != 1 or not usage:
                    raise RuntimeError('CLI SSE mismatch')
            owned = psutil.Process(proc.pid).children(recursive=True)
            report['worker_pids'] = [p.pid for p in owned]
            stop_console(proc.pid)
            proc.wait(timeout=45)
            psutil.wait_procs(owned, timeout=10)
            if any(p.is_running() for p in owned):
                raise RuntimeError('CLI normal shutdown left workers')
            report.update(status='passed', returncode=proc.returncode, workers_exited=True,
                          normal_console_shutdown=True)
        except BaseException as error:
            report.update(status='failed', error=repr(error))
            raise
        finally:
            if proc.poll() is None:
                owned = psutil.Process(proc.pid).children(recursive=True)
                for member in reversed(owned):
                    if member.is_running(): member.kill()
                proc.kill(); proc.wait(timeout=10)
                psutil.wait_procs(owned, timeout=10)
            report['remaining_descendants'] = [p.pid for p in owned if p.is_running()]
            args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(dict(status=report['status'], output=str(args.output), pid=proc.pid)), flush=True)


if __name__ == '__main__':
    main()
