"""在已安装解释器中模拟接收者：正常配置启动、共存切换、完整质量样本和重启。"""
from pathlib import Path
import argparse
import json
import os
import subprocess
import sys
import time
import httpx
import psutil

R=Path(__file__).parent
p=argparse.ArgumentParser()
p.add_argument('--app',type=Path,default=Path(r'C:\workspace\build\siltok-speed\i09'))
p.add_argument('--test-root',type=Path,default=R/'install-i09')
p.add_argument('--tag',default='receiver-overall-v1')
p.add_argument('--staged',action='store_true')
p.add_argument('--prior-simple-tag')
args=p.parse_args()
E=args.test_root
APP=args.app
CONFIG=E/'config' if args.staged else Path(r'C:\ProgramData\Siltok\Speed\config')
RUNTIME=E/'runtime' if args.staged else Path(r'C:\ProgramData\Siltok\Speed\runtime')
CACHE_TAG='portable1-receiver-staged' if args.staged else 'installed-i09'
SIMPLE_TAG=args.tag+'-simple' if args.staged else 'installed-v1'
report=dict(status='failed',license_mode='test_stub',clean_os_validated=False,
    device_license_validated=False,tested_python=sys.executable,actual_install_validated=not args.staged,
    rounds=[],quality={},restarts=[])
out=R/(args.tag+'.json')
if out.exists():
    raise FileExistsError(out)
env={k:v for k,v in os.environ.items() if not k.startswith(('SILITOK_','FREETOKEN_','CUDA_','TORCH_'))
    and k not in ('PYTHONPATH','PYTHONHOME','VIRTUAL_ENV','INCLUDE','LIB','LIBPATH','CUDACXX','CMAKE_PREFIX_PATH','CC','CXX','NVCC')}
env.update(PATH=';'.join([str(APP/'python/Scripts'),r'C:\Windows\System32',r'C:\Windows',
    r'C:\Program Files\NVIDIA Corporation\NVSMI']),TEMP=str(E/'tmp'),TMP=str(E/'tmp'),
    PYTHONUTF8='1',PYTHONDONTWRITEBYTECODE='1',
    SILITOK_FREETOKEN_KERNEL_CACHE_DIR=str(APP/'runtime/kernel-cache'),
    FREETOKEN_KERNEL_CACHE_DIR=str(APP/'runtime/kernel-cache'),FREETOKEN_DISABLE_JIT='1',
    TRITON_CACHE_DIR=rf'C:\workspace\cache\triton\{CACHE_TAG}',
    TORCH_EXTENSIONS_DIR=rf'C:\workspace\cache\ext\{CACHE_TAG}',
    HF_HOME=str(E/'empty-hf-cache'),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
for name in ('tmp','empty-hf-cache'):
    (E/name).mkdir(exist_ok=True)

def save():
    out.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

def post(client,mode,messages,limit=128,thinking=False,**extra):
    payload=dict(model='flash-next-strata-q2-'+mode,messages=messages,max_tokens=limit,
        temperature=0,enable_thinking=thinking,**extra)
    started=time.monotonic()
    response=client.post('/v1/chat/completions',json=payload)
    response.raise_for_status()
    return dict(response=response.json(),seconds=time.monotonic()-started)

def run_server(iteration,full):
    """实际 release_profile 读取安装生成的配置，再以 HTTP 驱动同一 Engine 切换。"""
    stop=E/f'receiver-stop-{iteration}'
    identity=R/f'{args.tag}-identity-{iteration}.json'
    log=(E/f'receiver-launch-{iteration}.log').open('wb')
    proc=subprocess.Popen([sys.executable,'-I','-X','utf8','-u','-B',str(R/'receiver-server.py'),
        '--stop-file',str(stop),'--identity',str(identity),
        '--config-root',str(CONFIG),'--runtime-dir',str(RUNTIME)],env=env,cwd=APP,
        stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
    entry=dict(iteration=iteration,pid=proc.pid)
    report['restarts'].append(entry)
    workers=set()
    try:
        with httpx.Client(base_url='http://127.0.0.1:8000',timeout=900,trust_env=False) as client:
            deadline=time.monotonic()+180
            while True:
                if proc.poll() is not None:
                    raise RuntimeError('Receiver server exited before readiness')
                try:
                    health=client.get('/health',timeout=3)
                    if health.status_code==200 and health.json()['status']=='ok':
                        break
                except httpx.TransportError:
                    pass
                if time.monotonic()>deadline:
                    raise TimeoutError('Receiver service readiness timeout')
                time.sleep(.3)
            info=json.loads(identity.read_text())
            assert not any('\\wt\\' in path.lower() or '\\envs\\' in path.lower() for path in info['sys_path']),info
            assert Path(info['models_dir']).resolve()==Path(r'C:\workspace\flash-next-resources')
            assert info['resource']['gpu_probe']=='nvml' and info['resource']['vram_safety_gb']==.25
            assert info['resource']['commit_safety_gb']==4,info
            entry.update(identity=info,health=health.json())
            listed=client.get('/v1/models');listed.raise_for_status()
            assert {m['id'] for m in listed.json()['data']}=={'flash-next-strata-q2-off','flash-next-strata-q2-on'}
            sequence=('off','on','off','on','off','on') if full else ('off','on')
            for number,mode in enumerate(sequence):
                row=post(client,mode,[dict(role='user',content='Calculate 17 + 25. Answer only the integer.')])
                assert row['response']['choices'][0]['message']['content'].strip().strip('.')=='42',row
                children=psutil.Process(proc.pid).children(recursive=True)
                current={(p.pid,p.create_time()) for p in children if p.name().lower()=='python.exe'}
                retired=workers-current
                assert all(not psutil.pid_exists(pid) or psutil.Process(pid).create_time()!=created for pid,created in retired),retired
                workers|=current
                status=client.get('/v1/status');status.raise_for_status()
                report['rounds'].append(dict(iteration=iteration,step=number,mode=mode,workers=sorted(current),
                    result=row,status=status.json(),main_private_bytes=psutil.Process(proc.pid).memory_info().private))
                save()
                if full and number<2:
                    inputs=json.loads((R/'receiver-quality-inputs.json').read_text(encoding='utf-8'))
                    samples=[]
                    for thinking,key in [(False,'quality'),(True,'thinking')]:
                        for case in inputs[key]:
                            result=post(client,mode,case['messages'],case['max_tokens'],thinking)
                            text=result['response']['choices'][0]['message']['content'].strip()
                            try:
                                passed=(json.loads(text)==case['expected']) if case['check_kind']=='json' else text==case['expected']
                            except ValueError:
                                passed=False
                            samples.append(dict(name=case['name'],thinking=thinking,quality_pass=passed,expected=case['expected'],**result))
                            report['quality'][mode]=samples
                            save()
                    sse=[]
                    with client.stream('POST','/v1/chat/completions',json=dict(model='flash-next-strata-q2-'+mode,
                        messages=[dict(role='user',content='将 cat 翻译成中文，只输出译词。')],
                        max_tokens=64,temperature=0,enable_thinking=False,stream=True)) as response:
                        response.raise_for_status()
                        for line in response.iter_lines():
                            if line.startswith('data: '):sse.append(line[6:])
                    assert sse[-1]=='[DONE]',sse
                    text=''.join(c.get('delta',{}).get('content','') for v in sse[:-1] for c in json.loads(v).get('choices',[]))
                    assert '猫' in text,text
                    report.setdefault('sse',{})[mode]=dict(text=text,chunks=sse)
                    rejected=client.post('/v1/chat/completions',json=dict(model='flash-next-strata-q2-'+mode,
                        messages=[dict(role='user',content='hello')],max_tokens=10,response_format=dict(type='json_object')))
                    assert rejected.status_code==400 and rejected.json()['error']['code']=='unsupported_feature',rejected.text
                    report.setdefault('unsupported_request',{})[mode]=rejected.json()
                    save()
    finally:
        stop.touch()
        proc.wait(timeout=90)
        log.close()
        deadline=time.monotonic()+15
        while any(psutil.pid_exists(pid) and psutil.Process(pid).create_time()==created for pid,created in workers) and time.monotonic()<deadline:
            time.sleep(.1)
        gone=all(not psutil.pid_exists(pid) or psutil.Process(pid).create_time()!=created for pid,created in workers)
        entry.update(returncode=proc.returncode,workers_gone=gone)
        save()
        assert proc.returncode==0 and gone,(proc.returncode,workers)

try:
    if args.prior_simple_tag:
        for mode in ('off','on'):
            prior=json.loads((R/f'{args.prior_simple_tag}-{mode}.json').read_text())
            assert prior['status']=='passed' and prior['workers_gone'] and prior['server_stopped']
            assert prior['mtp']['enabled']==(mode=='on') and prior['freetoken_version']=='0.1.3+speed.strataq2portable1'
        report['reused_component_reports']=args.prior_simple_tag
    else:
        with (R/(SIMPLE_TAG+'-models.log')).open('wb') as log:
            subprocess.run([sys.executable,'-I','-X','utf8','-u','-B',str(R/'receiver-simple-probe.py'),
                '--models-dir',r'C:\workspace\flash-next-resources','--tag',SIMPLE_TAG],env=env,cwd=APP,
                stdout=log,stderr=subprocess.STDOUT,check=True,timeout=1800,creationflags=subprocess.CREATE_NO_WINDOW)
    run_server(1,True)
    run_server(2,False)
    report['quality_scores']={mode:dict(no_thinking=sum(row['quality_pass'] for row in rows if not row['thinking']),
        thinking=sum(row['quality_pass'] for row in rows if row['thinking'])) for mode,rows in report['quality'].items()}
    report['status']='passed'
    save()
    (E/'suite-progress.json').write_text(json.dumps(dict(status='passed',
        scope='staged_receiver_not_installer' if args.staged else 'actual_installed_payload_not_clean_os'))+'\n')
except BaseException:
    save()
    (E/'suite-progress.json').write_text(json.dumps(dict(status='failed',
        scope='staged_receiver_not_installer' if args.staged else 'actual_installed_payload_not_clean_os'))+'\n')
    raise
