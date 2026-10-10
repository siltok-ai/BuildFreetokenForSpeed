"""独立性能探针：只使用已发布运行时，仓库源码和算子保持不变。"""
from __future__ import annotations

import argparse
import contextlib
import ctypes
import hashlib
import json
import os
from pathlib import Path
import random
import re
import statistics
import sys
import threading
import time

ROOT = Path(__file__).resolve().parent
SPEED = Path(os.environ.get('Q2_SPEED_ROOT', 'C:/workspace/worktrees/speed-flash-next-q2-opt'))
MODEL = Path('C:/workspace/models/qwen3.8-flash-next.bf16-q2-experts')
CACHE = Path(os.environ.get('Q2_EXPERIMENT_CACHE', 'C:/workspace/artifacts/freetoken-flash-next-q2exp1-build/kernel-cache'))
sys.path.insert(0, str(SPEED))
sys.path.insert(0, os.environ['Q2_EXPERIMENT_RUNTIME'])
os.environ['FREETOKEN_KERNEL_CACHE_DIR'] = str(CACHE)
os.environ['FREETOKEN_DISABLE_JIT'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['COMMIT_SAFETY_GB'] = '1'
os.environ['VRAM_SAFETY_GB'] = '2'
os.environ.setdefault('TRITON_CACHE_DIR', str(ROOT / 'triton-cache'))


def save(path, value):
    """及时保存原始数据，失败实验也留下可核对的证据。"""
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def percentile(values, p):
    values = sorted(values)
    if not values:
        return None
    x = (len(values) - 1) * p
    lo = int(x)
    hi = min(lo + 1, len(values) - 1)
    return values[lo] + (values[hi] - values[lo]) * (x - lo)


def request_summary(start, finish, times, ids, text):
    """以真实 token 到达时刻计算 TTFT 和生成区间，保留间隔分布。"""
    gaps = [(b - a) * 1000 for a, b in zip(times, times[1:])]
    return dict(ttft_ms=(times[0]-start)*1000 if times else None,
                elapsed_ms=(finish-start)*1000, output_tokens=len(ids),
                decode_tps=(len(ids)-1)/(times[-1]-times[0]) if len(ids)>1 else None,
                itl_median_ms=statistics.median(gaps) if gaps else None,
                itl_p95_ms=percentile(gaps, .95), itl_max_ms=max(gaps) if gaps else None,
                arrivals_ms=[(t-start)*1000 for t in times], token_ids=ids,
                token_sha256=hashlib.sha256(json.dumps(ids).encode()).hexdigest(), text=text)


def build_prompts(tokenizer):
    """构造固定输入集合；前部 nonce 防止冷输入重复命中前缀缓存。"""
    prompts = {}
    unit = ('A local AI runtime coordinates model weights, expert routing, GPU memory, host memory, '
            'KV cache, scheduling, streaming, cancellation, and error handling. ')
    for length in (70, 1024, 4096, 6144):
        for rep in range(4):
            text = f'Case {length:04d}-{rep:04d}. '
            if length == 70:
                text += 'Explain how a local inference engine balances GPU memory, host memory, and token latency. Give a detailed technical answer.'
            else:
                text += 'Read the following engineering notes and write a detailed technical analysis.\n'
            def encode(t):
                return tokenizer.apply_chat_template([dict(role='user',content=t)],
                    add_generation_prompt=True, tokenize=True, enable_thinking=False, return_dict=False)
            ids = encode(text)
            while len(ids) + len(tokenizer.encode(unit)) < length:
                text += unit
                ids = encode(text)
            while len(ids) < length:
                text += ' benchmark'
                ids = encode(text)
            prompts[f'{length}-{rep}'] = dict(text=text, ids=ids, input_tokens=len(ids),
                sha256=hashlib.sha256(json.dumps(ids).encode()).hexdigest())
    text = 'Answer with only one integer. A service handles 37 requests per minute for 19 minutes. How many requests does it handle?'
    ids = tokenizer.apply_chat_template([dict(role='user',content=text)], add_generation_prompt=True,
        tokenize=True, enable_thinking=False, return_dict=False)
    prompts['quality'] = dict(text=text, ids=ids, input_tokens=len(ids))
    for rep in range(3):
        rng=random.Random(12345+rep)
        text=f'Diverse case {rep:04d}. Analyze these varied server records and discuss the resource bottlenecks.\n'
        prefix=lambda t:tokenizer.apply_chat_template([dict(role='user',content=t)],add_generation_prompt=True,
            tokenize=True,enable_thinking=False,return_dict=False)
        index=0
        while True:
            row=f'Record {index}: node {rng.randrange(1000)}, input {rng.randrange(16,8000)}, output {rng.randrange(16,2000)}, cache {rng.randrange(100)}, latency {rng.randrange(100,9000)}, queue {rng.randrange(32)}.\n'
            if len(prefix(text+row))>4096:break
            text+=row;index+=1
        ids=prefix(text)
        while len(ids)<4096:text+=' benchmark';ids=prefix(text)
        prompts[f'diverse-4096-{rep}']=dict(text=text,ids=ids,input_tokens=len(ids))
    text=('A model server has 24 GiB of VRAM. It reserves 2 GiB, uses 3 GiB for runtime state, '
          'and each cache block costs 256 MiB. Determine the maximum whole number of cache blocks '
          'that fit. Explain the reasoning and end with a line formatted exactly as FINAL: <integer>.')
    ids=tokenizer.apply_chat_template([dict(role='user',content=text)],add_generation_prompt=True,
        tokenize=True,enable_thinking=True,return_dict=False)
    prompts['thinking']=dict(text=text,ids=ids,input_tokens=len(ids))
    return prompts


def options(args):
    """复用 Speed 的配置解析和 kwargs 投影，避免原生对照偷偷改变配置。"""
    from silitok_speed.backends.base import BackendConfig
    from silitok_speed.types import ModelSpec
    from silitok_speed.backends.freetoken_qwen36 import _checkpoint_for, _runtime_options, _worker_option_payload
    spec = ModelSpec(model_id='qwen3.8-flash-next.q2_0',weights_path=str(MODEL),quantization='q2_0')
    extra = dict(kernel_cache_dir=str(CACHE), max_prefill_length=args.chunk, moe_cache_size=args.cache,
        moe_cache_auto=False, kv_cache_tokens=8192, cuda_graph_max_bs=args.graph,
        startup_timeout_seconds=600, request_deadline_seconds=1800, shutdown_timeout_seconds=30,
        host_commit_estimate_bytes=int(args.host_budget*1024**3),vram_estimate_bytes=int(args.vram_budget*1024**3))
    cfg = BackendConfig(device='cuda:0',offload=False,extra=extra)
    opts = _worker_option_payload(_runtime_options(_checkpoint_for(spec,cfg),cfg))
    return spec,cfg,opts


class Instrument:
    """诊断专用计时包装；不替换计算，不逐层同步，不用于主性能排名。"""
    def __init__(self, llm, torch, enabled):
        self.llm, self.torch, self.enabled = llm, torch, enabled
        self.steps, self.host, self.gpu = [], [], []
        self.phase = None
        engine = llm.engine
        original = engine.forward_batch
        def forward(batch, sampling):
            phase = 'prefill' if batch.is_prefill else 'decode'
            self.phase = phase
            start = time.perf_counter()
            cached = [r.cached_len for r in batch.reqs]
            lengths = [r.device_len-r.cached_len for r in batch.reqs]
            use_graph = engine.graph_runner.can_use_cuda_graph(batch)
            a = b = None
            if enabled:
                a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                a.record(engine.stream)
            ctx = torch.profiler.record_function(f'probe.forward.{phase}') if enabled else contextlib.nullcontext()
            with ctx:
                result = original(batch,sampling)
            if enabled:
                b.record(engine.stream)
                self.gpu.append((phase,a,b))
            self.steps.append(dict(phase=phase,host_ms=(time.perf_counter()-start)*1000,
                cached=cached,new_tokens=lengths,use_graph=use_graph))
            return result
        engine.forward_batch = forward
        if enabled:
            cache=engine.moe_offload_cache
            invalidate=cache._invalidate_prefill_buffer
            def timed_invalidate(*a,**kw):
                begin=time.perf_counter()
                with torch.profiler.record_function('probe.prefill_invalidate'):
                    result=invalidate(*a,**kw)
                self.host.append(dict(kind='prefill_invalidate',phase=self.phase,ms=(time.perf_counter()-begin)*1000))
                return result
            cache._invalidate_prefill_buffer=timed_invalidate
            ple = engine.model._ple_table
            instrument=self
            class TimedStore:
                """透传原始 RowStore，只记录 hashing/stage 和文件读取/flush 的墙钟时间。"""
                def __init__(self,store):self.store=store
                def __getattr__(self,name):return getattr(self.store,name)
                def stage(self,*a,**kw):return self.timed('ple_stage',self.store.stage,*a,**kw)
                def flush(self,*a,**kw):return self.timed('ple_flush',self.store.flush,*a,**kw)
                def timed(self,kind,func,*a,**kw):
                    begin=time.perf_counter()
                    with torch.profiler.record_function(f'probe.{kind}.{instrument.phase}'):
                        result=func(*a,**kw)
                    instrument.host.append(dict(kind=kind,phase=instrument.phase,ms=(time.perf_counter()-begin)*1000))
                    return result
            ple._store=TimedStore(ple._store)
            fill = ple.fill
            def timed_fill(*a,**kw):
                begin=time.perf_counter()
                with torch.profiler.record_function(f'probe.ple_fill.{self.phase}'):
                    result=fill(*a,**kw)
                self.host.append(dict(kind='ple_fill',phase=self.phase,ms=(time.perf_counter()-begin)*1000))
                return result
            ple.fill=timed_fill

    def clear(self):
        self.steps.clear(); self.host.clear(); self.gpu.clear()

    def result(self):
        return dict(steps=self.steps.copy(),host=self.host.copy(),
            gpu_stream_windows=[dict(phase=p,ms=a.elapsed_time(b)) for p,a,b in self.gpu])


def main():
    """在一个进程中运行一套配置，资源样本由独立 supervisor 获取。"""
    parser=argparse.ArgumentParser()
    parser.add_argument('--mode',choices=['native','backend'],default='native')
    parser.add_argument('--cache',type=int,default=1024)
    parser.add_argument('--chunk',type=int,default=1024)
    parser.add_argument('--graph',type=int,default=1)
    parser.add_argument('--stats',action='store_true')
    parser.add_argument('--profile',action='store_true')
    parser.add_argument('--suite',choices=['full','matrix','profile'],default='matrix')
    parser.add_argument('--host-budget',type=float,default=36)
    parser.add_argument('--vram-budget',type=float,default=14)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    import psutil
    result=dict(args=vars(args), measurement_protocol=2, pid=os.getpid(), speed_root=str(SPEED),model=str(MODEL),runs=[])
    save(args.output,result)
    spec,cfg,opts=options(args)
    result['runtime_options']=opts
    if args.mode=='backend':
        from silitok_speed.backends.freetoken_qwen36 import FreeTokenQwen36Backend
        from transformers import AutoTokenizer
        tokenizer=AutoTokenizer.from_pretrained(MODEL,local_files_only=True)
        prompts=build_prompts(tokenizer)
        backend=FreeTokenQwen36Backend()
        t=time.perf_counter(); backend.load(spec,cfg)
        result['load_seconds']=time.perf_counter()-t
        result['geometry']=backend.cache_geometry()
        result['worker_pid']=backend.worker_pid
        def run(prompt,tokens,label):
            started=time.perf_counter(); times=[]; ids=[]; final=None
            for ev in backend.generate(dict(messages=[dict(role='user',content=prompt['text'])],enable_thinking=False),
                    max_new_tokens=tokens,temperature=0,top_p=1):
                if ev.get('type')=='token':
                    times.append(time.perf_counter()); ids.append(ev['token_id'])
                if ev.get('type')=='final': final=ev
            item=request_summary(started,time.perf_counter(),times,ids,final['text'])
            item.update(label=label,input_tokens=final['usage']['input_tokens'],finish_reason=final['finish_reason'])
            return item
    else:
        from silitok_speed.qwen36.freetoken_kernel_cache import activate_runtime,verify_kernel_cache
        activate_runtime(); verify_kernel_cache(CACHE)
        from silitok_speed.qwen36.freetoken_worker import _install_windows_posix_cache_compat,_llm_kwargs
        _install_windows_posix_cache_compat()
        import torch
        import freetoken.kernel.backend as kernel_backend
        kernel_backend.is_flashinfer_installed=lambda:False
        from freetoken.llm.llm import LLM
        from freetoken.core import SamplingParams
        from freetoken.message import DetokenizeMsg
        class TimedLLM(LLM):
            def offline_send_result(self,reply):
                for msg in reply:
                    if isinstance(msg,DetokenizeMsg) and not(msg.finished and msg.next_token in self.eos_token_ids):
                        self.arrivals.append(time.perf_counter())
                return super().offline_send_result(reply)
        kwargs=_llm_kwargs(opts)
        if args.stats: kwargs['moe_collect_stats']=True
        t=time.perf_counter(); llm=TimedLLM(str(MODEL),**kwargs)
        result['load_seconds']=time.perf_counter()-t
        result['runtime_source']=__import__('freetoken').__file__
        result['versions']=dict(torch=torch.__version__,freetoken=__import__('freetoken').__version__)
        result['geometry']=dict(moe_cache_size=llm.engine.moe_offload_cache.cache_size,num_pages=llm.engine.num_pages)
        result['ple']=dict(io_backend=llm.engine.model._ple_table._store.io_backend(),wait_sync=llm.engine.model._ple_table._wait_sync)
        prompts=build_prompts(llm.tokenizer)
        instrument=Instrument(llm,torch,args.profile)
        def run(prompt,tokens,label):
            torch.cuda.synchronize(); instrument.clear(); llm.arrivals=[]
            io_before=llm.engine.model._ple_table._store.io_stats()
            if args.stats: llm.engine.moe_offload_cache.reset_stats()
            started=time.perf_counter()
            data=llm.generate([prompt['ids']],SamplingParams(temperature=0,top_p=1,max_tokens=tokens))[0]
            finished=time.perf_counter(); torch.cuda.synchronize()
            item=request_summary(started,finished,llm.arrivals,data['token_ids'],data['text'])
            item.update(label=label,input_tokens=prompt['input_tokens'],instrument=instrument.result(),
                cuda_allocated=torch.cuda.memory_allocated(),cuda_reserved=torch.cuda.memory_reserved())
            io_after=llm.engine.model._ple_table._store.io_stats()
            item['ple_io_delta']={key:io_after[key]-io_before[key] for key in
                ('read_requests_completed','aligned_read_bytes_requested','payload_rows_delivered')}
            item['ple_io_after']=io_after
            if args.stats:
                item['moe_stats']=llm.engine.moe_offload_cache.decode_miss_stats()
                item['moe_stats_layers']=llm.engine.moe_offload_cache.decode_miss_stats_per_layer()
                item['routing']=llm.engine.moe_offload_cache.decode_routing_stats()
            return item
    save(ROOT/'prompts.json',prompts)
    result['warmup']=run(prompts['70-3'],16,'warmup')
    save(args.output,result)
    lengths=(70,1024,4096,6144) if args.suite=='full' else (70,4096)
    repetitions=3 if not args.profile else 1
    try:
        for length in lengths:
            # shape warmup 使用不同前部 nonce，测量时仍是冷前缀。
            result['runs'].append(run(prompts[f'{length}-3'],8,f'shape-warmup-{length}'))
            for rep in range(repetitions):
                prompt=prompts[f'{length}-{rep}']
                label=f'cold-{length}-{rep}'
                if args.profile:
                    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA],
                            record_shapes=False,profile_memory=False,with_stack=False) as prof:
                        item=run(prompt,12,label)
                    prof.export_chrome_trace(str(Path(args.output).with_suffix(f'.{length}.trace.json')))
                    item['profiler_cpu_table']=prof.key_averages().table(sort_by='self_cpu_time_total',row_limit=25)
                else:
                    item=run(prompt,128 if length==70 else 32,label)
                result['runs'].append(item)
                save(args.output,result)
                print(json.dumps(dict(label=label,ttft_ms=item['ttft_ms'],decode_tps=item['decode_tps'],output_tokens=item['output_tokens']),ensure_ascii=False),flush=True)
            hot_prompt=prompts[f'{length}-{repetitions-1}']
            result['runs'].append(run(hot_prompt,8,f'hot-shape-warmup-{length}'))
            for hot_rep in range(3):
                if args.profile and length==4096 and hot_rep==0:
                    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA],
                            record_shapes=False,profile_memory=False,with_stack=False) as hot_prof:
                        hot_item=run(hot_prompt,12,f'hot-{length}-{hot_rep}')
                    hot_prof.export_chrome_trace(str(Path(args.output).with_suffix('.hot4096.trace.json')))
                else:
                    hot_item=run(hot_prompt,32,f'hot-{length}-{hot_rep}')
                result['runs'].append(hot_item)
            save(args.output,result)
        result['quality']=run(prompts['quality'],16,'quality')
        result['quality_pass']=result['quality']['text'].strip()=='703'
        if args.mode=='native' and not args.profile and (args.suite=='full' or args.stats):
            result['extended']=[]
            for rep in range(3):
                result['extended'].append(run(prompts[f'diverse-4096-{rep}'],32,f'diverse-4096-{rep}'))
                save(args.output,result)
            thinking=run(prompts['thinking'],4096,'thinking')
            thinking['answer_pass']=bool(re.search(r'FINAL:\s*76\b',thinking['text']))
            result['extended'].append(thinking)
            save(args.output,result)
        if args.mode=='backend':
            result['vram_current']=backend.vram_current()
            result['worker_stderr']=backend._transport.stderr_tail()
        result['success']=True
    finally:
        t=time.perf_counter()
        if args.mode=='backend': backend.unload()
        else: llm.shutdown()
        result['unload_seconds']=time.perf_counter()-t
        save(args.output,result)
    return 0


if __name__=='__main__':
    raise SystemExit(main())
