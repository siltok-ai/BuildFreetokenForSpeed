"""对已安装的原版FreeToken做新机推理；支持较小的原生专家驻留池。"""
from pathlib import Path
import argparse
import gzip
import hashlib
import importlib.metadata as metadata
import json
import runpy
import sys
import time

import torch
from freetoken.core import SamplingParams
from freetoken.llm.llm import LLM
from freetoken.message import DetokenizeMsg
from freetoken.mm.config import ENCODER_KINDS, MultimodalConfig

HERE=Path(__file__).parent
RAW=Path(r'C:\workspace\wt\siltok-speed\strata-review\research\strata-q2-mtp2-20261014\raw')


def archived(name):
    """读取已通过104份证据身份复核的历史固定输入。"""
    return json.loads(gzip.decompress((RAW/name).read_bytes()))


class TimedLLM(LLM):
    def __init__(self,*args,**kwargs):
        """在调度器可能发送回调之前初始化交付计数器。"""
        self.arrivals=[]
        self.terminals=[]
        super().__init__(*args,**kwargs)

    def offline_send_result(self, messages):
        """记录实际交付事件及终止决策，MTP草稿不作为输出计数。"""
        now=time.perf_counter()
        for message in messages:
            if isinstance(message,DetokenizeMsg):
                self.terminals.append(dict(token=int(message.next_token),finished=message.finished,
                                           finish_reason=message.finish_reason))
                if not (message.finished and message.next_token in self.eos_token_ids):
                    self.arrivals.append(now)
        return super().offline_send_result(messages)


def main():
    """先记录加载身份，再逐请求保存结果，失败也留下已完成证据。"""
    parser=argparse.ArgumentParser()
    parser.add_argument('--model',type=Path,required=True)
    parser.add_argument('--resident',type=int,default=1024)
    parser.add_argument('--mtp-runtime',type=Path)
    parser.add_argument('--suite',choices=['smoke','quality','performance'],default='smoke')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--tokens',type=int,default=512)
    args=parser.parse_args()
    if args.output.exists(): raise FileExistsError(args.output)
    report=dict(status='loading',args={k:str(v) for k,v in vars(args).items()},
        python=sys.executable,version=metadata.version('freetoken'),runs=[],
        protocol='Arithmetic warmup, then fixed suite; no expert or prefix reset. '
                 'Performance repeats each prompt immediately and counts only delivered tokens.')
    report['identity']=runpy.run_path(str(HERE/'run-identity.py'))['capture']()
    report['probe_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    def save(): args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    save()
    started=time.perf_counter()
    llm=TimedLLM(str(args.model),dtype=torch.bfloat16,max_running_req=1,
        attention_backend='qsa_sparse',moe_strategy='offload',ple_backend='disk',expert_load='serial',
        mm=MultimodalConfig(disabled_encoders=frozenset(ENCODER_KINDS)),
        moe_cache_size=args.resident+2048 if args.resident else 1024,
        moe_q2_resident_experts=args.resident,moe_q2_dynamic_slots=1024,
        moe_cache_auto=False,kv_reserve_tokens=8192,num_token_override=8192,cuda_graph_max_bs=1,
        memory_ratio=.9,max_seq_len_override=262144,max_extend_tokens=4096,
        cache_type='radix',decode_log_interval=1000,
        mtp_runtime_path=str(args.mtp_runtime) if args.mtp_runtime else None,mtp_max_context=8192)
    report['load_seconds']=time.perf_counter()-started
    import freetoken
    report['source']=freetoken.__file__
    cache=llm.engine.moe_offload_cache
    owner=getattr(cache,'protected_payload',None)
    report['ownership']=dict(host_bytes=owner.host_bytes,gpu_bank_bytes=owner.gpu_bank_bytes,
        permanent=len(owner.protected_slots),total_slots=owner.slots) if owner else dict(permanent=0)
    report['status']='running'
    baseline=archived('quality-gate-dev7/freetoken.json.gz')
    baseline={r['name']:r for r in baseline['runs']}
    quality=archived('quality-gate-dev7/protocol.json.gz')['cases']
    perf=archived('mtp2-native-performance-v2.prompts.json.gz')
    prefill=[]
    original_forward=llm.engine.forward_batch
    def forward(batch,params):
        if not batch.is_decode:
            prefill.append(dict(type=str(batch.forward_mode) if hasattr(batch,'forward_mode') else 'prefill',
                requests=[dict(cached_len=r.cached_len,device_len=r.device_len) for r in batch.reqs]))
        return original_forward(batch,params)
    llm.engine.forward_batch=forward
    def run(name,ids,limit,case=None,temperature=0):
        llm.arrivals=[];llm.terminals=[];prefill.clear()
        start=time.perf_counter()
        output=llm.generate([ids],SamplingParams(temperature=temperature,top_p=1,max_tokens=limit))[0]
        elapsed=time.perf_counter()-start
        count=len(output['token_ids'])
        row=dict(name=name,input_tokens=len(ids),output=output,output_tokens=count,
                 elapsed_seconds=elapsed,terminal=llm.terminals[-1] if llm.terminals else None,
                 prefill_events=list(prefill),torch_allocated=torch.cuda.memory_allocated(),
                 torch_reserved=torch.cuda.memory_reserved(),arrivals=len(llm.arrivals),temperature=temperature)
        if len(llm.arrivals)>=2:
            row.update(ttft_ms=(llm.arrivals[0]-start)*1000,
                       decode_tps=(count-1)/(llm.arrivals[-1]-llm.arrivals[0]))
        if case is not None:
            text=output['text'].rsplit('</think>',1)[-1].strip()
            if case.get('check_kind')=='json':
                try: row['quality_pass']=json.loads(text)==case['expected']
                except ValueError: row['quality_pass']=False
            else: row['quality_pass']=text==case['expected']
            if name in baseline:
                row['matches_historical_ids']=output['token_ids']==baseline[name]['output']['token_ids']
        report['runs'].append(row);save()
        print(json.dumps({k:v for k,v in row.items() if k not in ('output','prefill_events')},ensure_ascii=False),flush=True)
        return row
    try:
        warm=run('warmup',quality[0]['tokens'],64,quality[0])
        if warm['output']['text'].strip()!='527': raise RuntimeError('First real-model arithmetic failed')
        if args.mtp_runtime:
            rounds=len(llm.engine.mtp_runtime.rounds)
            run('non_greedy_fallback',quality[0]['tokens'],32,temperature=.7)
            report['non_greedy_mtp_windows']=len(llm.engine.mtp_runtime.rounds)-rounds
            if report['non_greedy_mtp_windows']!=0: raise RuntimeError('Non-greedy request entered MTP verification')
        if args.suite=='smoke':
            a=run('repeat',quality[0]['tokens'],64,quality[0])
            if a['output']!=warm['output']: raise RuntimeError('Repeated request changed arithmetic')
            run('context4096',perf['diverse-4096-0']['ids'],32)
            run('after_long',quality[0]['tokens'],64,quality[0])
        elif args.suite=='quality':
            for case in quality: run(case['name'],case['tokens'],case['max_tokens'],case)
            for case in archived('quality-thinking-dev3/protocol.json.gz')['cases']:
                run('thinking-'+case['name'],case['tokens'],case['max_tokens'],case)
            report['score']=sum(r.get('quality_pass',False) for r in report['runs'] if r['name'] in baseline)
            report['thinking_score']=sum(r.get('quality_pass',False) for r in report['runs'] if r['name'].startswith('thinking-'))
            report['regressions']=[r['name'] for r in report['runs'] if r['name'] in baseline
                and baseline[r['name']]['quality_pass'] and not r['quality_pass']]
            if report['score']<15 or report['thinking_score']!=3 or report['regressions']:
                raise RuntimeError('Quality fell below the frozen Q2 baseline (15/20 and thinking 3/3)')
        else:
            for key in ('70-0','diverse-4096-0'):
                for repeat in range(3): run(key+'-'+str(repeat),perf[key]['ids'],args.tokens)
        report['status']='passed'
        if args.mtp_runtime:
            windows=list(llm.engine.mtp_runtime.rounds)
            report['mtp_windows']=dict(count=len(windows),drafted=sum(w['draft_count'] for w in windows),
                accepted=sum(w['accepted_drafts'] for w in windows),committed_rows=sum(w['committed_rows'] for w in windows))
    except BaseException as error:
        report['status']='failed'
        report['error']=repr(error)
        raise
    finally:
        llm.shutdown()
        report['shutdown_complete']=True
        save()
    print(json.dumps(dict(status=report['status'],output=str(args.output))),flush=True)


if __name__=='__main__': main()
