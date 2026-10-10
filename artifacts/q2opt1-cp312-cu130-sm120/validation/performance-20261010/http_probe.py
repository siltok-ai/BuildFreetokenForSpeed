"""真实 localhost SSE 测量，不使用会缓冲响应的 TestClient。"""
import json
import argparse
import os
from pathlib import Path
import socket
import threading
import time

import probe
os.environ['SILITOK_FREETOKEN_KERNEL_CACHE_DIR']=str(probe.CACHE)
os.environ['SILITOK_FREETOKEN_RUNTIME_DIR']=os.environ['Q2_EXPERIMENT_RUNTIME']

def main():
    """以正常 Engine/资源准入/HTTP 路径运行同一组提示并完整关闭服务。"""
    import httpx
    import uvicorn
    from silitok_speed import Engine,EngineConfig
    from silitok_speed.api import create_app
    from silitok_speed.discovery import discover_and_register
    parser=argparse.ArgumentParser();parser.add_argument('--vram-budget',type=float,default=14)
    parser.add_argument('--chunk',type=int,default=4096)
    parser.add_argument('--ipc-timing',action='store_true')
    parser.add_argument('--kv-tokens',type=int,default=8192)
    parser.add_argument('--output',default=str(probe.ROOT/'http-baseline.json'));args=parser.parse_args()
    root=probe.ROOT/'http-models'
    model_id='qwen3.8-flash-next.q2_0-perf'
    extra=dict(attention_backend='qsa_sparse',ple_backend='disk',moe_cache_size=1024,
        max_prefill_length=args.chunk,kv_cache_tokens=args.kv_tokens,cuda_graph_max_bs=1,
        startup_timeout_seconds=600,request_deadline_seconds=1800,shutdown_timeout_seconds=30,
        host_commit_estimate_bytes=36*1024**3,vram_estimate_bytes=int(args.vram_budget*1024**3))
    manifest=dict(schema='silitok.models/1',models=[dict(id=model_id,name=model_id,type='language',
        family='qwen3.8',features=['chat'],engine=dict(pipeline='chat',backend='Qwen36Backend',offload=False,extra=extra),
        artifacts=dict(weights=dict(path='weights',layout='hf_model')))])
    root.mkdir(exist_ok=True)
    if not (root/'weights').exists():
        os.symlink(probe.MODEL,root/'weights',target_is_directory=True)
    (root/'manifest.json').write_text(json.dumps(manifest),encoding='utf-8')
    engine=Engine(EngineConfig(models_dir=str(root),result_store_dir=str(root/'results'),
        vram_safety_margin_bytes=2*1024**3,commit_safety_margin_bytes=1*1024**3))
    registered,catalog=discover_and_register(engine,str(root))
    assert registered==1
    app=create_app(engine,manage_engine_lifecycle=True);app.state.model_catalog=catalog
    listener=socket.socket();listener.bind(('127.0.0.1',0));port=listener.getsockname()[1]
    server=uvicorn.Server(uvicorn.Config(app,host='127.0.0.1',port=port,log_level='warning'))
    thread=threading.Thread(target=lambda:server.run(sockets=[listener]),daemon=True);thread.start()
    while not server.started:
        if not thread.is_alive():raise RuntimeError('HTTP server failed startup')
        time.sleep(.05)
    output=Path(args.output)
    report=dict(mode='http-real-sse',port=port,runs=[],runtime_extra=extra)
    prompts=json.loads((probe.ROOT/'prompts.json').read_text(encoding='utf-8'))
    capture=[]
    with httpx.Client(base_url=f'http://127.0.0.1:{port}',timeout=1800,trust_env=False) as client:
        def run(prompt,tokens,label):
            capture.clear();start=time.perf_counter();started_unix=time.time();times=[];chunks=[];text=[];usage=None;finish_reason=None
            with client.stream('POST','/v1/chat/completions',json=dict(model=model_id,
                    messages=[dict(role='user',content=prompt['text'])],enable_thinking=False,
                    max_tokens=tokens,temperature=0,top_p=1,stream=True)) as response:
                if response.status_code!=200:
                    response.read();raise RuntimeError(response.text)
                for line in response.iter_lines():
                    if line.startswith('data: ') and line!='data: [DONE]':
                        chunk=json.loads(line[6:]);chunks.append(chunk)
                        for choice in chunk.get('choices',[]):
                            content=choice.get('delta',{}).get('content')
                            if content:
                                times.append(time.perf_counter());text.append(content)
                            if choice.get('finish_reason'):finish_reason=choice['finish_reason']
                        if chunk.get('usage'):usage=chunk['usage']
            stop=time.perf_counter()
            if not times or usage is None:
                report['failed_response_chunks']=chunks
                probe.save(output,report)
                raise RuntimeError('missing streamed content or usage: '+json.dumps(chunks,ensure_ascii=False)[:2000])
            gaps=[(b-a)*1000 for a,b in zip(times,times[1:])]
            item=dict(label=label,ttft_ms=(times[0]-start)*1000,elapsed_ms=(stop-start)*1000,
                wall_start_unix=started_unix,
                input_tokens=usage['prompt_tokens'],output_tokens=usage['completion_tokens'],
                decode_tps=(usage['completion_tokens']-1)/(times[-1]-times[0]) if len(times)>1 else None,
                content_events=len(times),event_p95_ms=probe.percentile(gaps,.95),
                arrivals_ms=[(t-start)*1000 for t in times],finish_reason=finish_reason,text=''.join(text))
            frames=list(capture)
            tokens_=[f for f in frames if f['type']=='token']
            content=[f for f in frames if f.get('delta_text')]
            if tokens_:
                item['ipc']=dict(ttft_ms=(tokens_[0]['time']-start)*1000,
                    decode_tps=(len(tokens_)-1)/(tokens_[-1]['time']-tokens_[0]['time']),
                    arrivals_ms=[(f['time']-start)*1000 for f in tokens_],token_ids=[f['token_id'] for f in tokens_],
                    content_matches=[f['delta_text'] for f in content]==text)
                if item['ipc']['content_matches']:
                    delays=[(arrival-f['time'])*1000 for arrival,f in zip(times,content)]
                    item['ipc']['delivery_delay_ms']=delays
                    item['ipc']['delivery_delay_median_ms']=probe.percentile(delays,.5)
                    item['ipc']['delivery_delay_p95_ms']=probe.percentile(delays,.95)
                    item['ipc']['delivery_delay_last_ms']=delays[-1]
            return item
        try:
            report['load_and_first_request']=run(prompts['70-3'],8,'initial-load')
            if args.ipc_timing:
                # 只给原 Queue.put 加时间戳，队列内容和读写协议保持原样。
                registered_model=engine.model_mgr.resolve(model_id)
                backend=engine.backend_pool.acquire('probe-ipc-timing',registered_model.spec)
                engine.backend_pool.release('probe-ipc-timing',registered_model.spec)
                messages=backend._transport._messages
                original_put=messages.put
                def timed_put(message,*a,**kw):
                    if isinstance(message,dict) and message.get('type') in ('token','delta'):
                        capture.append(dict(message,time=time.perf_counter()))
                    return original_put(message,*a,**kw)
                messages.put=timed_put
            probe.save(output,report)
            for length in (70,4096):
                report['runs'].append(run(prompts[f'{length}-3'],8,f'shape-warmup-{length}'))
                for rep in range(3):
                    item=run(prompts[f'{length}-{rep}'],128 if length==70 else 32,f'cold-{length}-{rep}')
                    report['runs'].append(item);probe.save(output,report)
                    print(json.dumps({k:item[k] for k in ('label','ttft_ms','decode_tps','output_tokens')}),flush=True)
                report['runs'].append(run(prompts[f'{length}-2'],32,f'hot-{length}'))
            report['quality']=run(prompts['quality'],16,'quality')
            report['quality_pass']=report['quality']['text'].strip()=='703'
            report['success']=True
        finally:
            server.should_exit=True;thread.join(timeout=60)
            report['server_closed']=not thread.is_alive()
            probe.save(output,report)
    return 0

if __name__=='__main__':raise SystemExit(main())
