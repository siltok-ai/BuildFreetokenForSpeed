"""独立 FreeToken 引擎测试，不导入或启动 Speed backend/worker/HTTP。"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parent
SOURCE = Path(os.environ.get('FREETOKEN_RESEARCH_SOURCE',str(ROOT/"candidate-source/build-source/python")))
if os.environ.get('FREETOKEN_RESEARCH_INSTALLED') != '1':
    sys.path.insert(0,str(SOURCE))
os.environ.setdefault("FREETOKEN_KERNEL_CACHE_DIR",r"C:\workspace\projects\BuildFreetokenForSpeed\artifacts\q2opt2-cp312-cu130-sm120\kernel-cache")
os.environ["FREETOKEN_DISABLE_JIT"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ.setdefault("TRITON_CACHE_DIR",str(ROOT/"candidate-triton-cache"))
if os.name == "nt" and not hasattr(os,"posix_fadvise"):
    os.posix_fadvise = lambda _fd,_offset,_length,_advice: None
    os.POSIX_FADV_DONTNEED = 0


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--output",required=True)
    parser.add_argument("--tokens",type=int,default=16)
    parser.add_argument("--graph",type=int,default=0)
    parser.add_argument("--cache",type=int,default=1024)
    parser.add_argument("--chunk",type=int,default=4096)
    parser.add_argument("--correctness-matrix",action="store_true")
    parser.add_argument("--capture-logits",action="store_true")
    parser.add_argument("--numeric-projections",action="store_true")
    parser.add_argument("--numeric-states",action="store_true")
    parser.add_argument("--numeric-experts",action="store_true")
    parser.add_argument("--long-context",type=int,default=0)
    parser.add_argument("--cancel-reuse",action="store_true")
    parser.add_argument("--diagnose-long-prefix",action="store_true")
    parser.add_argument("--canonical-expert-prefill",action="store_true")
    args=parser.parse_args()
    import torch
    import freetoken.kernel.backend as backend
    backend.is_flashinfer_installed=lambda:False
    from freetoken.core import SamplingParams
    from freetoken.llm.llm import LLM,RequestAllFinished
    from freetoken.message import DetokenizeMsg,AbortBackendMsg,ErrorReplyMsg
    from freetoken.mm.config import ENCODER_KINDS,MultimodalConfig
    if args.numeric_projections:
        from diagnose_runtime_projections import install
        install()
    if args.numeric_states:
        from diagnose_runtime_states import install
        install()
    if args.numeric_experts:
        from diagnose_runtime_experts import install
        install()
    if args.canonical_expert_prefill:
        # 只做因果诊断：短 prefill 重复到 grouped 门槛，保留冻结 wheel 原字节。
        import freetoken.moe.fused_q2_q8 as expert_module
        expert_original=expert_module.fused_experts_q2_q8
        def canonical_experts(x,gu,down,weights,ids,*,is_prefill=False):
            n=x.shape[0]
            if is_prefill and n<128:
                repeats=(128+n-1)//n
                return expert_original(x.repeat(repeats,1),gu,down,weights.repeat(repeats,1),ids.repeat(repeats,1),is_prefill=True)[:n].contiguous()
            return expert_original(x,gu,down,weights,ids,is_prefill=is_prefill)
        expert_module.fused_experts_q2_q8=canonical_experts
    class TimedLLM(LLM):
        def __init__(self,*a,**kw):
            self.abort_queue=[];self.abort_acks=[];self.cancel_after=None;self.cancel_issued=False
            super().__init__(*a,**kw)
        def offline_receive_msg(self,blocking=False):
            if self.abort_queue:
                messages,self.abort_queue=self.abort_queue,[]
                return messages
            return super().offline_receive_msg(blocking)
        def offline_send_result(self,reply):
            normal=[]
            for msg in reply:
                if isinstance(msg,ErrorReplyMsg) and msg.error=='request aborted':
                    self.abort_acks.append(msg.uid)
                    continue
                normal.append(msg)
                if isinstance(msg,DetokenizeMsg):
                    if not(msg.finished and msg.next_token in self.eos_token_ids):
                        self.arrivals.append(time.perf_counter())
                    if msg.finished:
                        self.terminal_token=int(msg.next_token)
                    if self.cancel_after is not None and len(self.arrivals)>=self.cancel_after and not self.cancel_issued:
                        self.abort_queue.append(AbortBackendMsg(uid=msg.uid))
                        self.cancel_issued=True
            return super().offline_send_result(normal)
    result=dict(passed=False,pid=os.getpid(),source=str(__import__('freetoken').__file__),torch=torch.__version__,
        runtime_version=__import__("freetoken").__version__,args=vars(args),runs=[])
    output=Path(args.output)
    def save():
        output.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    save()
    started=time.perf_counter()
    llm=TimedLLM(str(ROOT/"gguf-checkpoint"),dtype=torch.bfloat16,max_running_req=1,
        attention_backend="qsa_sparse",moe_strategy="offload",ple_backend="disk",
        mm=MultimodalConfig(disabled_encoders=frozenset(ENCODER_KINDS)),expert_load="serial",
        moe_q2_resident_experts=9216,moe_q2_dynamic_slots=1024,moe_cache_size=args.cache,moe_cache_auto=False,kv_reserve_tokens=8192,num_token_override=8192,
        cuda_graph_max_bs=args.graph,memory_ratio=.8,max_seq_len_override=262144,
        max_extend_tokens=args.chunk,cache_type="radix",decode_log_interval=1000,moe_collect_stats=True)
    result["load_seconds"]=time.perf_counter()-started
    if result['runtime_version'] != '0.1.3+speed.strataq2dev6':
        raise RuntimeError('Wrong installed FreeToken revision')
    cache = llm.engine.moe_offload_cache
    from freetoken.moe.q2_residency import ProtectedQ2Cache
    from freetoken.engine.engine import CacheRebuildRejected
    if type(cache) is not ProtectedQ2Cache:
        raise RuntimeError('Explicit resident configuration did not select the native owner cache')
    pointers = {role: value.data_ptr() for role, value in cache.bank_caches.items()}
    graph_before = llm.engine.graph_runner
    rejected = False
    try:
        llm.engine.rebuild_runtime_cache(moe_cache_size=cache.cache_size - 1)
    except CacheRebuildRejected:
        rejected = True
    unchanged = graph_before is llm.engine.graph_runner and pointers == {role: value.data_ptr() for role, value in cache.bank_caches.items()}
    if not rejected or not unchanged or getattr(llm.engine, 'rebuild_teardown_started', False):
        raise RuntimeError('Unsafe expert resize did not reject before teardown')
    llm.engine.rebuild_runtime_cache(num_pages=llm.engine.num_pages)
    retained = pointers == {role: value.data_ptr() for role, value in cache.bank_caches.items()}
    if not retained:
        raise RuntimeError('KV-only rebuild changed expert owner addresses')
    result['rebuild_checks'] = dict(expert_resize_rejected_before_teardown=rejected and unchanged,
        kv_only_rebuild_preserves_owner_addresses=retained, fixed_capacity=cache.cache_size)
    result['ownership'] = dict(protected=len(cache.protected_payload.protected_slots),
        dynamic=cache.protected_payload.dynamic, host_bytes=cache.protected_payload.host_bytes,
        gpu_bank_bytes=cache.protected_payload.gpu_bank_bytes)
    result['process_engine_override'] = False
    result['native_engine_only'] = True
    if args.diagnose_long_prefix:
        from diagnose_long_prefix import install
        install(llm.engine,output.with_name(output.stem+'-diagnostic.json'))
    forward = llm.engine.forward_batch
    sample = llm.engine.sampler.sample
    llm.engine_events=[]
    llm.logit_rows=[]
    def forward_checked(batch,sampling):
        event=dict(is_decode=batch.is_decode,input_tokens=batch.input_ids.numel(),
            requests=[dict(cached_len=r.cached_len,device_len=r.device_len,prompt_len=r.prompt_len,
                           position=r.device_len-1) for r in batch.reqs])
        llm.engine_events.append(event)
        return forward(batch,sampling)
    def sample_checked(logits,sampling):
        if args.capture_logits:
            event=llm.engine_events[-1]
            for i,r in enumerate(event['requests']):
                if r['device_len']>=r['prompt_len']:
                    llm.logit_rows.append((r['position'],logits[i].detach().cpu().clone()))
        return sample(logits,sampling)
    llm.engine.forward_batch=forward_checked
    llm.engine.sampler.sample=sample_checked
    pool=llm.engine.linear_state_pool
    result["state_dtypes"]=dict(conv=str(pool.conv_states.dtype),recurrent=str(pool.recurrent_states.dtype),
        kv=str(llm.engine.kv_cache.dtype),index=str(llm.engine.kv_cache.index_dtype))
    result["cache_slots"]=llm.engine.moe_offload_cache.cache_size
    result["ple"]=dict(format=llm.engine.model._ple_table._format,
        backend=llm.engine.model._ple_table._store.io_backend())
    save()
    jobs = (("quality","Answer with only one integer. A service handles 37 requests per minute for 19 minutes. How many requests does it handle?"),
                       ("technical-cold","Explain how a local inference engine balances GPU memory, host memory, and token latency. Give a detailed technical answer."),
                       ("technical-hot","Explain how a local inference engine balances GPU memory, host memory, and token latency. Give a detailed technical answer."))
    if args.correctness_matrix:
        padding = "This is a consistency test of local model execution. Read the following request carefully. "*5
        exact = padding+"Answer with only one integer: what is 27 multiplied by 8?"
        changed = padding+"Answer with only one integer: what is 27 multiplied by 9?"
        jobs += (("prefix-cold",exact),("prefix-hot",exact),("prefix-diverged",changed),
                 ("prefix-restored",exact))
    if args.long_context:
        # 使 QSA 可见 token 超过2048，实际触发 block selection。
        filler="This paragraph is background information about a local service and its routine operation. "
        text=filler*max(1,args.long_context//16)+"\nIgnore the background. Answer with only one integer: what is 27 multiplied by 8?"
        jobs += (("long-cold",text),("long-hot",text))
    for label,text in jobs:
        ids=llm.tokenizer.apply_chat_template([dict(role="user",content=text)],add_generation_prompt=True,
             tokenize=True,enable_thinking=False,return_dict=False)
        llm.arrivals=[];llm.terminal_token=None;llm.engine_events=[];llm.logit_rows=[]
        torch.cuda.synchronize();start=time.perf_counter()
        data=llm.generate([ids],SamplingParams(temperature=0,top_p=1,max_tokens=args.tokens))[0]
        finish=time.perf_counter();times=llm.arrivals
        row=dict(label=label,input_tokens=len(ids),input_sha256=hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
            output_tokens=len(data["token_ids"]),terminal_token=llm.terminal_token,
            eos=llm.terminal_token in llm.eos_token_ids,text=data["text"],token_ids=data["token_ids"],
            ttft_ms=(times[0]-start)*1000 if times else None,
            decode_tps=(len(times)-1)/(times[-1]-times[0]) if len(times)>1 else None,
            elapsed_seconds=finish-start,engine_events=llm.engine_events)
        if args.capture_logits:
            from safetensors.torch import save_file
            tensors={f'position_{position}':logits for position,logits in llm.logit_rows}
            logit_file=output.with_name(output.stem+'-'+label+'-logits.safetensors')
            save_file(tensors,str(logit_file),metadata=dict(input_sha256=row['input_sha256']))
            row['logits_file']=str(logit_file)
        result["runs"].append(row);save();print(json.dumps(row,ensure_ascii=False),flush=True)
    result.update(passed=True,quality_pass=result["runs"][0]["text"].strip()=="703",
        note="首次真实推理 smoke；不代表完整Strata对照或正式性能验收。native研究复用身份另见native-research-reuse.json")
    if args.correctness_matrix:
        runs = {r['label']:r for r in result['runs']}
        result['matrix_checks'] = dict(arithmetic_703=result['quality_pass'],
            arithmetic_216=all(runs[k]['text'].strip()=='216' for k in ('prefix-cold','prefix-hot','prefix-restored')),
            arithmetic_243=runs['prefix-diverged']['text'].strip()=='243',
            prefix_identical=all(runs[k]['token_ids']==runs['prefix-cold']['token_ids'] for k in ('prefix-hot','prefix-restored')),
            technical_identical=runs['technical-cold']['token_ids']==runs['technical-hot']['token_ids'])
        result['correctness_pass'] = all(result['matrix_checks'].values())
    if args.long_context:
        long_runs=[r for r in result['runs'] if r['label'].startswith('long-')]
        result['long_context_checks']=dict(sparse_context_exercised=all(r['input_tokens']>2051 for r in long_runs),
            arithmetic_216=all(r['text'].strip()=='216' for r in long_runs),
            tokens_identical=long_runs[0]['token_ids']==long_runs[1]['token_ids'],
            real_prefix_hit=long_runs[1]['engine_events'][0]['requests'][0]['cached_len']>0)
        result['long_context_pass']=all(result['long_context_checks'].values())
    if args.cancel_reuse:
        def manual_run(ids,sp,cancel_after=None):
            llm.arrivals=[];llm.terminal_token=None;llm.engine_events=[];llm.logit_rows=[]
            llm.cancel_after=cancel_after;llm.cancel_issued=False;llm.abort_acks=[]
            # Offline generate resets UID; manual lifecycle test keeps distinct request IDs.
            uid=llm.counter
            llm.pending_requests=[(ids,sp,None)]
            llm.status_map={}
            try:
                llm.run_forever()
            except RequestAllFinished:
                pass
            torch.cuda.synchronize()
            status=llm.status_map[uid]
            return dict(uid=uid,text=llm.tokenizer.decode(status.output_ids),token_ids=status.output_ids,
                abort_acks=list(llm.abort_acks),cancel_issued=llm.cancel_issued,
                prefill_runnable=bool(llm.prefill_manager.runnable),decode_runnable=bool(llm.decode_manager.runnable),
                mamba_usage=llm._mamba_slot_usage(),page_usage=llm.cache_manager.page_usage())
        long_ids=llm.tokenizer.apply_chat_template([dict(role='user',content='Explain in detail how a local inference engine manages memory and state.')],
            add_generation_prompt=True,tokenize=True,enable_thinking=False,return_dict=False)
        cancelled=manual_run(long_ids,SamplingParams(temperature=0,max_tokens=128,ignore_eos=True),2)
        reuse_ids=llm.tokenizer.apply_chat_template([dict(role='user',content='Answer with only one integer: what is 37 multiplied by 19?')],
            add_generation_prompt=True,tokenize=True,enable_thinking=False,return_dict=False)
        reused=manual_run(reuse_ids,SamplingParams(temperature=0,max_tokens=16))
        result['cancel_reuse']=dict(cancelled=cancelled,reused=reused,
            passed=cancelled['cancel_issued'] and cancelled['abort_acks']==[cancelled['uid']]
                and not cancelled['prefill_runnable'] and not cancelled['decode_runnable']
                and reused['text'].strip()=='703' and not reused['prefill_runnable'] and not reused['decode_runnable'])
    save()
    llm.shutdown()


if __name__ == "__main__":
    main()
