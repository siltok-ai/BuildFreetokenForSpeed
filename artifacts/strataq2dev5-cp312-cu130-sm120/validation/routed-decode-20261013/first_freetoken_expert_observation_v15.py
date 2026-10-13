"""Standalone long-output FreeToken measurement; no Speed adapter imports."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import time

ROOT=Path(__file__).resolve().parent

def write(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf-8')
def digest(v):return hashlib.sha256(json.dumps(v).encode()).hexdigest()

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--model',required=True);p.add_argument('--output',required=True)
    p.add_argument('--expected-version',default='0.1.3+speed.strataq2dev4')
    p.add_argument('--cache',type=int,default=1024);p.add_argument('--cache-auto',action='store_true')
    p.add_argument('--tokens',type=int,default=512);p.add_argument('--repetitions',type=int,default=3)
    p.add_argument('--graph',type=int,default=1);p.add_argument('--profile',action='store_true')
    p.add_argument('--contexts',nargs='+',choices=['short','diverse4096'],default=['short','diverse4096'])
    a=p.parse_args();out=Path(a.output)
    if out.exists():raise FileExistsError(out)
    acceptance=json.loads((ROOT/'dev4-native-runtime-gate-v1.json').read_text(encoding='utf-8'))
    if not acceptance['passed'] or not all(acceptance['gates'].values()):raise RuntimeError('Correctness gate incomplete')
    import capture_freetoken_reference
    import torch
    import freetoken
    import freetoken.kernel.backend as backend
    backend.is_flashinfer_installed=lambda:False
    from freetoken.core import SamplingParams
    from freetoken.llm.llm import LLM
    from freetoken.mm.config import MultimodalConfig,ENCODER_KINDS
    from freetoken.message import DetokenizeMsg
    if freetoken.__version__!=a.expected_version:raise RuntimeError('Wrong runtime version')
    options=dict(dtype=torch.bfloat16,max_running_req=1,attention_backend='qsa_sparse',
        moe_strategy='offload',ple_backend='disk',expert_load='serial',
        mm=MultimodalConfig(disabled_encoders=frozenset(ENCODER_KINDS)),
        moe_cache_size=a.cache,moe_cache_auto=a.cache_auto,moe_collect_stats=True,
        kv_reserve_tokens=8192,num_token_override=8192,cuda_graph_max_bs=a.graph,
        memory_ratio=.8,max_seq_len_override=262144,max_extend_tokens=4096,
        cache_type='radix',decode_log_interval=1000)
    report=dict(status='running',version=freetoken.__version__,source=freetoken.__file__,model=a.model,
        args=vars(a),no_mtp=True,runs=[],runtime_gate=str(ROOT/'dev4-native-runtime-gate-v1.json'),runtime_gate_sha256=hashlib.sha256((ROOT/'dev4-native-runtime-gate-v1.json').read_bytes()).hexdigest(),native_engine_only=False,speed_acceptance=False,sampling=dict(temperature=0,top_p=1,ignore_eos=False,max_tokens=a.tokens),
        protocol='Standalone actual token delivery timestamps. Decode=(N-1)/(last-first); TTFT excludes tokenizer/model startup. Prefix and GPU expert cold reset; hot repeats immediately. No OS file cache purge. EOS and actual truncation retained.',
        instrumentation='GPU cache counters and CPU PLE timers; no per-layer synchronization or logits copies. Profiling runs are separate diagnostic evidence.')
    write(out,report)
    start=time.perf_counter();llm=LLM(a.model,**options);report['load_seconds']=time.perf_counter()-start
    cache=llm.engine.moe_offload_cache;ple=llm.engine.model._ple_table
    report['geometry']=dict(expert_cache_slots=cache.cache_size,num_pages=llm.engine.num_pages,
        expert_layers=cache.num_layers,expert_counts=cache.num_experts,graph_max_bs=a.graph)
    report['ple']=dict(io_backend=ple._store.io_backend(),wait_sync=ple._wait_sync,format=getattr(ple,'_format','fp8_e4m3'))
    # Freeze the existing historical input IDs instead of regenerating or overwriting them.
    prompts=json.loads((ROOT/'prompts.json').read_text(encoding='utf-8'))
    chosen={k:v for k,v in prompts.items() if k.startswith('70-') or k.startswith('diverse-4096-') or k=='quality'}
    for key,case in chosen.items():
        actual=llm.tokenizer.apply_chat_template([dict(role='user',content=case['text'])],
            add_generation_prompt=True,enable_thinking=False,tokenize=True,return_dict=False)
        if actual!=case['ids']:raise RuntimeError('Historical prompt tokenizer changed: '+key)
    prompts_path=out.with_suffix('.prompts.json');write(prompts_path,chosen)
    report['prompts_sha256']=hashlib.sha256(prompts_path.read_bytes()).hexdigest()
    arrivals=[];terminals=[];prefills=[];timers=[];phase=[None];decode_start_stats=[]
    send=llm.send_result;forward=llm.engine.forward_batch
    def send_hook(msgs):
        for m in msgs:
            if isinstance(m,DetokenizeMsg):
                now=time.perf_counter();token=int(m.next_token)
                if not (m.finished and token in llm.eos_token_ids):arrivals.append((now,token))
                if m.finished:terminals.append(dict(token=token,finish_reason=m.finish_reason,at=now))
        return send(msgs)
    def forward_hook(batch,params):
        phase[0]='decode' if batch.is_decode else 'prefill'
        if batch.is_decode and not decode_start_stats:
            # One stream-ordered device snapshot after prefill, before first decode;
            # no per-token host synchronization. Separates hot prefill counters.
            decode_start_stats.append(cache.lru_stats.clone())
        if not batch.is_decode:prefills.append([dict(cached_len=r.cached_len,device_len=r.device_len) for r in batch.reqs])
        if a.profile:
            with torch.profiler.record_function('probe.forward.'+phase[0]):
                return forward(batch,params)
        return forward(batch,params)
    llm.send_result=send_hook;llm.engine.forward_batch=forward_hook
    class TimedStore:
        def __init__(self,store):self.store=store
        def __getattr__(self,name):return getattr(self.store,name)
        def stage(self,*args,**kwargs):return self.timed('stage',self.store.stage,*args,**kwargs)
        def flush(self,*args,**kwargs):return self.timed('flush',self.store.flush,*args,**kwargs)
        def timed(self,kind,f,*args,**kwargs):
            t=time.perf_counter();result=f(*args,**kwargs)
            timers.append(dict(kind=kind,phase=phase[0],ms=1000*(time.perf_counter()-t)))
            return result
    ple._store=TimedStore(ple._store)
    fill=ple.fill
    def fill_hook(*args,**kwargs):
        t=time.perf_counter();result=fill(*args,**kwargs)
        timers.append(dict(kind='fill_including_host_wait',phase=phase[0],ms=1000*(time.perf_counter()-t)))
        return result
    ple.fill=fill_hook
    def run(case,label,limit,cold):
        if cold:cache.reset()
        torch.cuda.synchronize();cache.reset_stats();torch.cuda.synchronize()
        arrivals.clear();terminals.clear();prefills.clear();timers.clear();decode_start_stats.clear()
        before=ple._store.io_stats()
        started=time.perf_counter()
        output=llm.generate([case['ids']],SamplingParams(temperature=0,top_p=1,max_tokens=limit))[0]
        finished=time.perf_counter();torch.cuda.synchronize()
        if not terminals or [t for _,t in arrivals]!=output['token_ids']:raise RuntimeError('Actual delivered token accounting mismatch')
        ts=[t for t,_ in arrivals];gaps=[1000*(b-a) for a,b in zip(ts,ts[1:])]
        eos=terminals[-1]['token'] in llm.eos_token_ids
        raw_stats=cache.decode_miss_stats();layers=cache.decode_miss_stats_per_layer()
        prefill_stats=decode_start_stats[0] if decode_start_stats else cache.lru_stats.clone()
        decode_stats=cache.lru_stats-prefill_stats
        # For GPU offload every miss is copied; use realized counter sums, not configured cache rate.
        active,missing,calls=[int(v) for v in decode_stats.sum(0).tolist()]
        all_gpu=cache.decode_target=='gpu' and not cache._unpinned_layers
        if not all_gpu:raise RuntimeError('Phase counters here support pure GPU offload; use actual hybrid counters separately')
        stats=dict(layer_calls=calls,active_per_layer=active/calls if calls else 0,
            missing_per_layer=missing/calls if calls else 0,miss_rate=missing/active if active else 0,
            fetched_per_layer=missing/calls if calls else 0,cpu_per_layer=0,fetch_rate=1 if missing else 0)
        phase_layers=[dict(layer=i,active=a,missing=m,calls=c,miss_rate=m/a if a else 0)
            for i,(a,m,c) in enumerate(decode_stats.tolist())]
        expert_slot_bytes=sum(c[0].numel()*c.element_size() for _,c in cache.banks)
        after=ple._store.io_stats()
        cached=[x['cached_len'] for b in prefills for x in b]
        item=dict(label=label,input_tokens=len(case['ids']),input_sha256=digest(case['ids']),
            output=output,output_tokens=len(ts),max_output_tokens=limit,eos=eos,
            truncated=not eos and len(ts)==limit,terminal=terminals[-1],
            ttft_ms=1000*(ts[0]-started) if ts else None,
            decode_tps=(len(ts)-1)/(ts[-1]-ts[0]) if len(ts)>1 else None,
            elapsed_ms=1000*(finished-started),itl_median_ms=statistics.median(gaps) if gaps else None,
            itl_p95_ms=sorted(gaps)[int(.95*(len(gaps)-1))] if gaps else None,
            arrivals_ms=[1000*(t-started) for t in ts],prefills=list(prefills),
            prefix_cached_tokens=max(cached,default=0),expert_cold_reset=cold,
            moe_stats=stats,moe_stats_layers=phase_layers,
            raw_cache_api_stats=raw_stats,raw_cache_api_layers=layers,
            cache_counter_note='Raw cache API combines prefill and decode, and its cpu_per_layer formula is not a CPU-execution measurement on plain GPU offload. Normalized fields use decode-only device snapshots and actual GPU residency.',
            prefill_expert_counters=prefill_stats.tolist(),decode_target=cache.decode_target,
            cpu_expert_layers=sorted(cache._unpinned_layers),
            active_expert_selections=active,missing_expert_selections=missing,decode_layer_calls=calls,
            expert_slot_bytes=expert_slot_bytes,expert_payload_h2d_bytes=missing*expert_slot_bytes,
            transfer_note='Decode-only packed payload bytes from realized misses. Prefill copies and CUDA batch descriptors/alignment excluded; counters include any actually launched unused pipeline forward.',
            ple_io_delta={k:after[k]-before[k] for k in ('read_requests_completed','aligned_read_bytes_requested','payload_rows_delivered')},
            ple_io_after=after,ple_host_timers=list(timers),
            ple_timer_note='fill includes host staging-buffer/readback waits and store flush; stage/flush are nested, do not sum all three. GPU WAIT duration requires diagnostic trace.',
            cuda_allocated=torch.cuda.memory_allocated(),cuda_reserved=torch.cuda.memory_reserved(),
            measurement_sufficient=len(ts)>=256)
        if cold and any(cached):raise RuntimeError('Cold prefix unexpectedly reused')
        return item
    try:
        report['warmup']=run(chosen['70-3'],'warmup',8,True);write(out,report)
        for context in a.contexts:
            for rep in range(a.repetitions):
                case=chosen[f'70-{rep}' if context=='short' else f'diverse-4096-{rep}']
                if rep==0 and context!='short':
                    # Warm the exact prefill shape, then measure with a distinct nonce.
                    warm=chosen['70-3'] if context=='short' else chosen['diverse-4096-2']
                    report.setdefault('shape_warmups',[]).append(run(warm,context+'-shape-warmup',8,True))
                    if context!='short':llm.cache_manager.rebuild(llm.cache_manager.num_pages,llm.cache_manager.page_table)
                for hot in (False,True):
                    label=f'{context}-{rep}-'+('hot' if hot else 'cold')
                    if a.profile:
                        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:
                            item=run(case,label,min(a.tokens,32),not hot)
                        prof.export_chrome_trace(str(out.with_suffix('.'+label+'.trace.json')))
                    else:item=run(case,label,a.tokens,not hot)
                    if hot:
                        item['cold_hot_token_ids_equal']=item['output']['token_ids']==report['runs'][-1]['output']['token_ids']
                        cold_ids=report['runs'][-1]['output']['token_ids'];hot_ids=item['output']['token_ids']
                        item['cold_hot_first_difference']=next((i for i,(x,y) in enumerate(zip(cold_ids,hot_ids)) if x!=y), None)
                        item['cold_hot_length_equal']=len(cold_ids)==len(hot_ids)
                    report['runs'].append(item);write(out,report)
                    print(json.dumps({k:item[k] for k in ('label','input_tokens','output_tokens','eos','truncated','ttft_ms','decode_tps','prefix_cached_tokens')},ensure_ascii=False),flush=True)
        report['quality']=run(chosen['quality'],'quality',16,True)
        report['quality_pass']=report['quality']['output']['text'].strip()=='703'
        report['status']='diagnostic_profile_complete' if a.profile else 'observational_expert_performance_measured'
        report['all_measurements_sufficient']=all(x['measurement_sufficient'] for x in report['runs'])
        if not report['quality_pass']:raise RuntimeError('Post-performance arithmetic regression')
    finally:
        llm.shutdown();write(out,report)
    print(json.dumps(dict(status=report['status'],runs=len(report['runs']),sufficient=report['all_measurements_sufficient']),ensure_ascii=False))

if __name__=='__main__':main()
