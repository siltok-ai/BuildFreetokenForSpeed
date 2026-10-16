"""本轮原生性能与中段 CUDA trace；每次输出保存完整 token 和实际来源。"""
from pathlib import Path
import argparse
import hashlib
import importlib.metadata as metadata
import json
import runpy
import time

import torch
import freetoken
from freetoken.core import SamplingParams
from freetoken.llm.llm import LLM
from freetoken.message import DetokenizeMsg
from freetoken.mm.config import ENCODER_KINDS, MultimodalConfig

OLD = Path(r'C:\workspace\runs\siltok-speed\flash-accept-20261015')
MODEL = Path(r'C:\workspace\models\siltok-speed\flash-next\checkpoints\flash-next-strata-q2')


class TimedLLM(LLM):
    def __init__(self, *args, **kwargs):
        """在调度器启动前准备交付计数器。"""
        self.arrivals = []
        self.terminals = []
        super().__init__(*args, **kwargs)

    def offline_send_result(self, messages):
        """只记录主模型最终交付 token，保留 EOS 与实际终止原因。"""
        now = time.perf_counter()
        for message in messages:
            if isinstance(message, DetokenizeMsg):
                self.terminals.append(dict(token=int(message.next_token), finished=message.finished,
                                           reason=message.finish_reason))
                if not (message.finished and message.next_token in self.eos_token_ids):
                    self.arrivals.append(now)
        return super().offline_send_result(messages)


def main():
    """先验身份和算术，再运行独立的吞吐或采样诊断，最终释放唯一模型。"""
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--resident', type=int, default=3584)
    p.add_argument('--dynamic', type=int, default=1024)
    p.add_argument('--prefill', type=int, default=4096)
    p.add_argument('--state-cache-slots', type=int)
    p.add_argument('--mtp', action='store_true')
    p.add_argument('--stats', action='store_true')
    p.add_argument('--inventory', action='store_true')
    p.add_argument('--adaptive-owners', action='store_true')
    p.add_argument('--adapt-every', type=int, default=32)
    p.add_argument('--rebuild', action='store_true')
    p.add_argument('--admission-runtime-base',action='store_true')
    p.add_argument('--q2-word16',action='store_true')
    p.add_argument('--grouped-meta',action='store_true')
    p.add_argument('--draft-subset',action='store_true')
    p.add_argument('--draft-subset-mapped',action='store_true')
    p.add_argument('--bf16-window-simt',action='store_true')
    p.add_argument('--host-embedding',action='store_true')
    p.add_argument('--mapped-embedding',action='store_true')
    p.add_argument('--q2-hybrid',action='store_true')
    p.add_argument('--hybrid-gpu-share',type=float,default=.55)
    p.add_argument('--dxgi',action='store_true')
    p.add_argument('--adapt-decay',type=float,default=1.0)
    p.add_argument('--bf16-mode',choices=('down','up','both'),default='down')
    p.add_argument('--bf16-min-tokens',type=int,choices=(2,3,4),default=4)
    p.add_argument('--bf16-down-bn',type=int,default=4)
    p.add_argument('--bf16-down-bk',type=int,default=256)
    p.add_argument('--bf16-down-split',type=int,default=4)
    p.add_argument('--bf16-up-bn',type=int,default=8)
    p.add_argument('--bf16-up-bk',type=int,default=128)
    p.add_argument('--draft-cap',type=int,choices=(1,2,3),default=3)
    p.add_argument('--trace', action='store_true')
    p.add_argument('--tokens', type=int, default=512)
    p.add_argument('--repetitions', type=int, default=3)
    p.add_argument('--keys', nargs='+', default=['70-0', 'diverse-4096-0'])
    p.add_argument('--variant', choices=['base','compact-mtp'], default='base')
    p.add_argument('--suite', choices=['performance','quality'], default='performance')
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    root = Path(freetoken.__file__).parent
    inventory = {f.relative_to(root).as_posix(): hashlib.sha256(f.read_bytes()).hexdigest()
                 for f in sorted(root.rglob('*')) if f.suffix in ('.py', '.pyd', '.json')}
    report = dict(status='loading', args=vars(args) | {'output': str(args.output)},
                  source=str(root), version=metadata.version('freetoken'), files=inventory,
                  runs=[], diagnostic=args.trace or args.stats)
    report['probe_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    report['scope']='diagnostic_hybrid_q2_cpu_timers_and_optional_nsight_not_throughput_acceptance'
    report['variants']={}
    if args.q2_hybrid:
        from hybrid_q2_variant_v3 import enable
        enable(gpu_share=args.hybrid_gpu_share)
        report['variants']['hybrid_q2']=dict(gpu_share=args.hybrid_gpu_share,threads=8)
    if args.admission_runtime_base:
        from admission_runtime_base_variant import enable
        enable()
        report['variants']['runtime_base']=hashlib.sha256(Path('admission_runtime_base_variant.py').read_bytes()).hexdigest()
    if args.q2_word16:
        from q2_word16_variant import enable
        enable()
        report['variants']['word16']=hashlib.sha256(Path('q2_word16_variant.py').read_bytes()).hexdigest()
    if args.grouped_meta:
        from grouped_precomputed_variant import enable
        enable()
        report['variants']['grouped_meta']=hashlib.sha256(Path('grouped_precomputed_variant.py').read_bytes()).hexdigest()
    if args.draft_subset or args.draft_subset_mapped:
        if not args.mtp:raise ValueError('Draft subset requires MTP')
        if args.draft_subset_mapped:
            from draft_mapped_subset_variant import enable, VOCAB_SHA
        else:
            from draft_subset_variant import enable, VOCAB_SHA
        enable()
        report['variants']['draft_subset']=dict(source_sha256=hashlib.sha256(Path('draft_mapped_subset_variant.py' if args.draft_subset_mapped else 'draft_subset_variant.py').read_bytes()).hexdigest(),vocab_sha256=VOCAB_SHA,rows=106299)
    if args.host_embedding:
        from host_embedding_variant import enable
        report['host_embedding']=enable()
        report['variants']['host_embedding']=dict(source_sha256=hashlib.sha256(Path('host_embedding_variant.py').read_bytes()).hexdigest(),alias_sha256=hashlib.sha256(Path('mapped_host_bytes.py').read_bytes()).hexdigest())
    if args.bf16_window_simt:
        from bf16_window_simt_variant import enable
        import freetoken.kernel.gguf_native as native_bf16
        previous_bf16=native_bf16.bf16_gemv
        report['bf16_shape_calls']=enable(down_bn=args.bf16_down_bn,down_bk=args.bf16_down_bk,down_split=args.bf16_down_split,up_bn=args.bf16_up_bn,up_bk=args.bf16_up_bk,mode=args.bf16_mode)
        window_bf16=native_bf16.bf16_gemv
        native_bf16.bf16_gemv=lambda packed,x,rows,**kw:previous_bf16(packed,x,rows,**kw) if len(x)<args.bf16_min_tokens else window_bf16(packed,x,rows,**kw)
        report['variants']['bf16_window_simt']=hashlib.sha256(Path('bf16_window_simt_variant.py').read_bytes()).hexdigest()
    if args.draft_cap<3:
        if not args.mtp:raise ValueError('Draft cap requires MTP')
        import freetoken.models.qwen4_exp.mtp_runtime as runtime
        original_depth=runtime.window_depth
        runtime.window_depth=lambda *a,**k:min(args.draft_cap,original_depth(*a,**k))
    assert Path(freetoken.__file__).is_relative_to(Path(r'C:\workspace\envs\siltok-speed\flash-perf-cand4\Lib\site-packages'))
    assert metadata.version('freetoken')=='0.1.3+speed.strataq2perf4'
    if any((args.admission_runtime_base,args.q2_word16,args.grouped_meta,args.draft_subset,
            args.draft_subset_mapped,args.bf16_window_simt,args.host_embedding)) or args.draft_cap!=3:
        raise ValueError('Frozen perf4 wheel validation must not install prototype hooks')


    def save():
        """逐请求写出证据，失败保留已完成的实际输出。"""
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')

    save()
    start = time.perf_counter()
    llm = TimedLLM(str(MODEL), dtype=torch.bfloat16, max_running_req=1,
        attention_backend='qsa_sparse', moe_strategy='offload', ple_backend='disk', expert_load='serial',
        mm=MultimodalConfig(disabled_encoders=frozenset(ENCODER_KINDS)),
        moe_cache_size=args.resident + args.dynamic + 1024,
        moe_q2_resident_experts=args.resident, moe_q2_dynamic_slots=args.dynamic,
        moe_collect_stats=args.stats, moe_cache_auto=False, kv_reserve_tokens=8192,
        num_token_override=8192, cuda_graph_max_bs=1, memory_ratio=.9,
        max_seq_len_override=262144, max_extend_tokens=args.prefill, cache_type='radix',
        decode_log_interval=1000, mtp_max_context=8192,
        mtp_runtime_path=str(MODEL.parent / 'flash-next-mtp') if args.mtp else None,
        mtp_compact_state=args.variant=='compact-mtp', linear_state_cache_slots=args.state_cache_slots,
        moe_q2_adapt_every=args.adapt_every if args.adaptive_owners else 0,
        gguf_embedding_host=args.mapped_embedding,moe_q2_adapt_decay=args.adapt_decay)
    report['embedding_placement']=getattr(llm.engine.model,'_embedding_placement',None)
    report['host_tables_bytes']=llm.engine._host_tables_bytes
    report['measured_weights_bytes']=llm.engine._weights_bytes
    report['post_weights_free']=llm.engine._post_weights_free
    dxgi=None
    if args.dxgi:
        from dxgi_memory_budget import BudgetReader
        dxgi=BudgetReader()
        report['dxgi_loaded']=dxgi.sample()
    report['source_version']=freetoken.__version__ if hasattr(freetoken,'__version__') else 'source'
    if args.mapped_embedding:
        assert report['embedding_placement']['actual']=='mapped_host',report['embedding_placement']
        assert report['host_tables_bytes']>=273152000

    report['load_seconds'] = time.perf_counter() - start
    if args.draft_subset or args.draft_subset_mapped:
        report['draft_subset_check'] = llm.engine.mtp_runtime.draft.draft_subset_check
    adaptive=llm.engine.q2_adaptive_owners
    cache = llm.engine.moe_offload_cache
    owner = cache.protected_payload
    report['ownership'] = dict(permanent=len(owner.protected_slots), host_bytes=owner.host_bytes,
                                gpu_bank_bytes=owner.gpu_bank_bytes, slots=owner.slots)
    if args.inventory:
        from memory_inventory import inventory
        report['memory_loaded']=inventory(llm.engine)
    prompts = json.loads((OLD / 'frozen-prompts.json').read_text(encoding='utf-8'))
    extra={
        'english':'Explain in detail how a local mixture-of-experts inference engine balances GPU memory, host memory, and token latency. Cover expert placement, cache misses, data movement, speculative decoding, and practical profiling. Use concrete examples and explain the tradeoffs.',
        'chinese':'请详细讲解本地混合专家大模型推理如何平衡显存、系统内存和生成延迟。包括专家驻留、缓存未命中、数据搬运、推测解码和实际性能分析方法。请用具体例子解释各项取舍，给出完整且有条理的技术说明。',
        'code':'Write a complete Python 3 module implementing a thread-safe LRU cache with a configurable TTL. Include type annotations, clear docstrings, get, put, delete, clear and stats methods, deterministic tests using an injectable clock, and a usage example. Explain the design after the code.'}
    for key in args.keys:
        if key in extra:
            ids=llm.tokenizer.apply_chat_template([{'role':'user','content':extra[key]}],tokenize=True,
                add_generation_prompt=True,enable_thinking=False,return_dict=False)
            prompts[key]=dict(text=extra[key],ids=ids)
    report['input_ids']={key:prompts[key]['ids'] for key in args.keys}
    protocol = runpy.run_path(str(OLD / 'native-probe.py'))['archived']('quality-gate-dev7/protocol.json.gz')
    arithmetic = protocol['cases'][0]

    def run(name, ids, tokens):
        """冻结输入并记录真实 token 到达间隔，统计读取放在计时之后。"""
        llm.arrivals.clear()
        llm.terminals.clear()
        if args.stats:
            cache.reset_stats()
        window_start=len(llm.engine.mtp_runtime.rounds) if args.mtp else 0
        before = time.perf_counter()
        out = llm.generate([ids], SamplingParams(temperature=0, top_p=1, max_tokens=tokens))[0]
        elapsed = time.perf_counter() - before
        row = dict(name=name, input_tokens=len(ids), output=out, output_tokens=len(out['token_ids']),
                   arrivals=len(llm.arrivals), arrival_offsets_s=[stamp-before for stamp in llm.arrivals], terminal=llm.terminals[-1], elapsed_seconds=elapsed,
                   ttft_ms=1000*(llm.arrivals[0]-before) if llm.arrivals else None,
                   decode_tps=(len(out['token_ids'])-1)/(llm.arrivals[-1]-llm.arrivals[0])
                       if len(llm.arrivals)>1 else None,
                   allocated=torch.cuda.memory_allocated(), reserved=torch.cuda.memory_reserved())
        if args.mtp:
            windows=list(llm.engine.mtp_runtime.rounds)[window_start:]
            row['mtp']=dict(windows=len(windows),drafted=sum(w['draft_count'] for w in windows),
                accepted=sum(w['accepted_drafts'] for w in windows),committed=sum(w['committed_rows'] for w in windows),
                lookup_windows=sum(w.get('draft_source')=='lookup' for w in windows))
        if args.q2_hybrid:
            from hybrid_q2_variant_v3 import stats
            row['hybrid_cpu']=stats()
        if args.stats:
            row['raw_miss_stats'] = cache.decode_miss_stats()
            # ProtectedQ2Cache直接gather未维护通用hybrid fetched计数；miss在此全由GPU搬运。
            active, misses, calls = (int(v) for v in cache.lru_stats.sum(0))
            row['miss_stats'] = dict(active=active,misses=misses,layer_calls=calls,
                miss_rate=misses/active if active else 0, cpu_experts=0,
                payload_bytes=misses * owner.gpu_bank_bytes // owner.slots)
            row['per_layer'] = cache.decode_miss_stats_per_layer()
        if dxgi is not None:row['dxgi']=dxgi.sample()
        report['runs'].append(row)
        save()
        print(json.dumps({k:v for k,v in row.items() if k not in ('output','per_layer','arrival_offsets_s')},ensure_ascii=False), flush=True)
        return row

    try:
        row = run('warmup', arithmetic['tokens'], 64)
        if row['output']['text'].strip() != '527':
            raise RuntimeError('Arithmetic warmup failed')
        report['status'] = 'running'
        if args.suite == 'performance':
            for key in args.keys:
                for repeat in range(args.repetitions):
                    run(f'{key}-{repeat}', prompts[key]['ids'], args.tokens)
        else:
            archived=runpy.run_path(str(OLD/'native-probe.py'))['archived']
            reference={r['name']:r for r in archived('quality-gate-dev7/freetoken.json.gz')['runs']}
            thinking=archived('quality-thinking-dev3/protocol.json.gz')['cases']
            for case in [*protocol['cases'],*thinking]:
                name=('thinking-' if case in thinking else '')+case['name']
                row=run(name,case['tokens'],case['max_tokens'])
                text=row['output']['text'].rsplit('</think>',1)[-1].strip()
                if case.get('check_kind')=='json':
                    try:row['quality_pass']=json.loads(text)==case['expected']
                    except ValueError:row['quality_pass']=False
                else:row['quality_pass']=text==case['expected']
                if name in reference:
                    row['matches_historical_ids']=row['output']['token_ids']==reference[name]['output']['token_ids']
                save()
            report['score']=sum(r.get('quality_pass',False) for r in report['runs'] if r['name'] in reference)
            report['thinking_score']=sum(r.get('quality_pass',False) for r in report['runs'] if r['name'].startswith('thinking-'))
            report['regressions']=[r['name'] for r in report['runs'] if r['name'] in reference
                and reference[r['name']]['quality_pass'] and not r['quality_pass']]
            if report['score']<15 or report['thinking_score']!=3 or report['regressions']:
                raise RuntimeError('Candidate failed frozen quality gate')
        if args.trace:
            if not args.mtp:raise ValueError('Capture whole MTP cycles')
            owner=llm.engine.mtp_runtime
            original=owner.forward
            calls=0
            def forward(*a,**k):
                """跳过64轮，捕获64轮；保留完整原路径及CPU交接。"""
                nonlocal calls
                selected=64<=calls<128
                if calls==64:torch.cuda.profiler.start()
                if selected:torch.cuda.nvtx.range_push('mtp_whole_window')
                try:return original(*a,**k)
                finally:
                    if selected:torch.cuda.nvtx.range_pop()
                    calls+=1
                    if calls==128:
                        torch.cuda.synchronize(llm.engine.device)
                        torch.cuda.profiler.stop()
            owner.forward=forward
            try:run('trace-'+args.keys[0],prompts[args.keys[0]]['ids'],args.tokens)
            finally:owner.forward=original
            if calls<128:raise RuntimeError('Missing full64-window capture')
            report['capture']=dict(request='trace-'+args.keys[0],steps=64,skipped_decode_steps=64)
        if args.rebuild:
            slots=llm.engine.linear_state_pool.num_slots
            report['rebuild_steps']=[]
            for usable in (slots,slots-1):
                llm.rebuild_cache(num_mamba_slots=usable)
                row=run('rebuild-'+str(usable),arithmetic['tokens'],64)
                if row['output']['text'].strip()!='527':raise RuntimeError('Post-rebuild arithmetic failed')
                llm.cache_manager.check_integrity()
                report['rebuild_steps'].append(dict(usable=usable,
                    logical_slots=llm.engine.linear_state_pool.num_slots,
                    recurrent_slots=llm.engine.linear_state_pool.recurrent_states.shape[1],
                    reserved=list(llm.engine.linear_state_pool.reserved_slot_ids)))
        row = run('after', arithmetic['tokens'], 64)
        if row['output']['text'].strip() != '527':
            raise RuntimeError('Post-performance arithmetic failed')
        if args.mtp:
            windows = llm.engine.mtp_runtime.rounds
            report['mtp_windows'] = dict(count=len(windows), drafted=sum(w['draft_count'] for w in windows),
                accepted=sum(w['accepted_drafts'] for w in windows), committed=sum(w['committed_rows'] for w in windows))
        if args.mtp:
            report['round_details']=list(llm.engine.mtp_runtime.rounds)
        if args.q2_hybrid:
            from hybrid_q2_variant_v3 import health
            health()
        report['status'] = 'passed'
    except BaseException as exc:
        report.update(status='failed', error=repr(exc))
        raise
    finally:
        if adaptive is not None:
            report['adaptive_rounds'] = list(llm.engine.q2_adaptive_owners.rounds)
        llm.shutdown()
        if args.q2_hybrid:
            from hybrid_q2_variant_v3 import close_all
            close_all()
        if dxgi is not None:dxgi.close()
        report['shutdown_complete'] = True
        save()


if __name__ == '__main__':
    main()
