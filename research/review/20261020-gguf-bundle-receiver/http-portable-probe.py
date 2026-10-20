"""离线消费搬迁后的独立资源包，串行验证真实 HTTP、MTP 与 worker 回收。"""
from pathlib import Path
import argparse
import json
import importlib.metadata
import os
import socket
import sys
import threading
import time

import httpx
import psutil
import uvicorn

parser = argparse.ArgumentParser()
parser.add_argument('--models-dir', type=Path, required=True)
parser.add_argument('--tag', required=True)
parser.add_argument('--source-root', type=Path)
parser.add_argument('--benchmark', action='store_true')
args = parser.parse_args()
if args.source_root:
    sys.path.insert(0, str(args.source_root))
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_HOME'] = str(Path(__file__).parent/'empty-hf-cache')
import silitok_speed
from silitok_speed import Engine, EngineConfig
from silitok_speed.api import create_app
from silitok_speed.discovery import discover_and_register
# 产品包先从被测解释器加载；随后只取独立测试授权工具。
sys.path.append(r'C:\workspace\wt\siltok-speed\flash-next')
from tools.engine_license import engine_license


def run(mode):
    """独立 Engine 完成真实模型请求和退出检查，失败也保留当次完整结果。"""
    output = Path(__file__).parent/f'{args.tag}-{mode}.json'
    if output.exists():
        raise FileExistsError(output)
    report = dict(mode=mode, license_mode='test_stub', product=silitok_speed.__file__,
        speed_version=silitok_speed.__version__, models_dir=str(args.models_dir),
        freetoken_version=importlib.metadata.version('freetoken'),
        hf_offline=True, runs=[], status='failed')
    workers = []
    baseline = {(p.pid,p.create_time()) for p in psutil.Process().children(recursive=True)}
    with engine_license(args.models_dir, test_license=True) as monitor:
        engine = Engine(EngineConfig(models_dir=str(args.models_dir),
            result_store_dir=str(output.with_suffix('')),
            gpu_resource_probe='nvml', vram_safety_margin_bytes=256*2**20,
            commit_safety_margin_bytes=4*2**30), grant_provider=monitor.grant)
        count, catalog = discover_and_register(engine, str(args.models_dir))
        assert count == 2, catalog
        app = create_app(engine, manage_engine_lifecycle=True)
        app.state.model_catalog = catalog
        with socket.socket() as listener:
            listener.bind(('127.0.0.1',0))
            port = listener.getsockname()[1]
            server = uvicorn.Server(uvicorn.Config(app, log_level='warning'))
            thread = threading.Thread(target=server.run, kwargs={'sockets':[listener]}, daemon=True)
            thread.start()
            try:
                deadline = time.monotonic()+15
                while not server.started:
                    if not thread.is_alive() or time.monotonic()>deadline:
                        raise RuntimeError('TCP server did not start')
                    time.sleep(.05)
                model_id = 'flash-next-strata-q2-'+mode
                base = dict(model=model_id, temperature=0, enable_thinking=False, max_tokens=128)
                with httpx.Client(base_url=f'http://127.0.0.1:{port}', timeout=900, trust_env=False) as client:
                    listed = client.get('/v1/models'); listed.raise_for_status()
                    report['models'] = listed.json()
                    started = time.monotonic()
                    response = client.post('/v1/chat/completions',json={**base,
                        'messages':[{'role':'user','content':'Calculate 17 + 25. Answer only the integer.'}]})
                    response.raise_for_status()
                    body = response.json()
                    report['runs'].append(dict(kind='sync', seconds=time.monotonic()-started, response=body))
                    children = psutil.Process().children(recursive=True)
                    workers = [(p.pid,p.create_time()) for p in children
                        if p.name().lower() == 'python.exe' and (p.pid,p.create_time()) not in baseline]
                    report['auxiliary_processes'] = [dict(pid=p.pid,created=p.create_time(),name=p.name(),
                        existed_before_engine=(p.pid,p.create_time()) in baseline)
                        for p in children if p.name().lower() != 'python.exe']
                    assert body['choices'][0]['message']['content'].strip().strip('.') == '42', body
                    binding = engine.model_mgr.resolve(model_id)
                    backend = engine.backend_pool.acquire('acceptance-diagnostics',binding.spec)
                    try:
                        report['mtp'] = backend.mtp_identity()
                        report['cache_geometry'] = backend.cache_geometry()
                    finally:
                        engine.backend_pool.release('acceptance-diagnostics',binding.spec)
                    assert report['mtp']['enabled'] == (mode == 'on'),report['mtp']
                    chunks = []
                    with client.stream('POST','/v1/chat/completions',json={**base,'stream':True,
                        'messages':[{'role':'user','content':'将 cat 翻译成中文，只输出译词。'}]}) as response:
                        response.raise_for_status()
                        for line in response.iter_lines():
                            if line.startswith('data: '): chunks.append(line[6:])
                    assert chunks[-1] == '[DONE]', chunks
                    content = ''.join(c.get('delta',{}).get('content','') for value in chunks[:-1]
                        for c in json.loads(value).get('choices',[]))
                    report['runs'].append(dict(kind='sse',text=content,chunks=chunks))
                    assert '猫' in content, content
                    thinking = client.post('/v1/chat/completions',json={**base,'enable_thinking':True,'max_tokens':512,
                        'messages':[{'role':'user','content':'计算 (12 - 3) * 7，最后只输出结果。'}]})
                    thinking.raise_for_status()
                    body = thinking.json()
                    report['runs'].append(dict(kind='thinking',response=body))
                    assert '63' in body['choices'][0]['message']['content'], body
                    if args.benchmark:
                        from tools.bench.benchmark_freetoken_http import request
                        from transformers import AutoTokenizer
                        tokenizer = AutoTokenizer.from_pretrained(
                            args.models_dir/'checkpoints/flash-next-strata-q2',local_files_only=True)
                        prompts = {
                            'english': 'Write a detailed technical article about how a CPU and a GPU cooperate during local language model inference. Cover memory movement, caching, attention, mixture of experts, scheduling, and practical performance measurement. Continue for at least 1500 words, with concrete examples.',
                            'chinese': '请详细介绍本地大语言模型推理中CPU和GPU如何协作。依次解释权重搬运、缓存、注意力、混合专家、请求调度和性能测量，并给出具体例子。请连续撰写至少三千字的技术文章。',
                        }
                        report['benchmarks'] = []
                        for language,text in prompts.items():
                            messages = [dict(role='user',content=text)]
                            encoded = tokenizer.apply_chat_template(messages,tokenize=True,
                                add_generation_prompt=True,enable_thinking=False)
                            ids = list(encoded['input_ids']) if hasattr(encoded,'keys') else list(encoded)
                            for repetition in range(3):
                                result = request(client,base|dict(messages=messages,max_tokens=512,top_p=1))
                                row = dict(language=language,repetition=repetition,prompt_ids=ids,
                                    cache_state='first_use' if repetition == 0 else 'hot_repeat',**result)
                                report['benchmarks'].append(row)
                                output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
                                assert result['usage']['prompt_tokens'] == len(ids),row
                                assert result['output_tokens'] == 512 and result['finish_reason'] == 'length',row
                                print(json.dumps(dict(tag=args.tag,mode=mode,language=language,
                                    repetition=repetition,decode_tps=result['decode_tps']),ensure_ascii=False),flush=True)
                    report['status'] = 'passed'
            finally:
                server.should_exit = True
                thread.join(60)
                report['server_stopped'] = not thread.is_alive()
                report['workers'] = workers
                deadline = time.monotonic()+15
                while any(psutil.pid_exists(pid) and psutil.Process(pid).create_time()==created
                          for pid,created in workers) and time.monotonic()<deadline:
                    time.sleep(.1)
                report['workers_gone'] = all(not psutil.pid_exists(pid) or psutil.Process(pid).create_time()!=created
                    for pid,created in workers)
                output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
                assert report['server_stopped'] and report['workers_gone'],report
    print(json.dumps(dict(mode=mode,status='passed',output=str(output)),ensure_ascii=False),flush=True)


for mode in ('off','on'):
    run(mode)
