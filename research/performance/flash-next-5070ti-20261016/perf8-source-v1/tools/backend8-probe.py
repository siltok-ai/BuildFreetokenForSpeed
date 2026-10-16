"""通过Speed正式backend和独立worker进行本机质量、吞吐及取消验收。"""
from pathlib import Path
import argparse
import gzip
import hashlib
import importlib.metadata as metadata
import json
import runpy
import statistics
import sys
import time

import psutil
sys.path.insert(0,r'C:\workspace\wt\siltok-speed\flash-perf')
import silitok_speed
_tools_parser=argparse.ArgumentParser(add_help=False)
_tools_parser.add_argument('--source',type=Path,default=Path(r'C:\workspace\wt\siltok-speed\flash-accept'))
_tool_args,_=_tools_parser.parse_known_args()
# 产品先从实际解释器导入；源码目录仅供应仓外验收工具，编译包不能回退到源码产品。
sys.path.insert(0,str(_tool_args.source.resolve(strict=True)))
from silitok_speed.backends.base import BackendConfig
from silitok_speed.backends.freetoken_qwen36 import FreeTokenQwen36Backend
from silitok_speed.types import ModelSpec
from tools.bench.benchmark_freetoken_qwen36 import _cancel_once
from tools.smoke.freetoken_qwen36 import _worker_process_tree

RAW=Path(r'C:\workspace\wt\siltok-speed\strata-review\research\strata-q2-mtp2-20261014\raw')


def archived(name):
    """复用已核验的历史输入及质量门槛，不重新挑选题目。"""
    return json.loads(gzip.decompress((RAW/name).read_bytes()))


def main():
    """正式worker加载后记录实际所有权，逐请求留证，最后检查整个进程树退出。"""
    parser=argparse.ArgumentParser()
    parser.add_argument('--source',type=Path,default=_tool_args.source)
    parser.add_argument('--model',type=Path,required=True)
    parser.add_argument('--resident',type=int,required=True)
    parser.add_argument('--vram-gib',type=float,required=True)
    parser.add_argument('--host-gib',type=float,required=True)
    parser.add_argument('--mtp-runtime',type=Path)
    parser.add_argument('--compact-mtp',action='store_true')
    parser.add_argument('--mtp-draft-policy',choices=('full','paired_cjk'),default='full')
    parser.add_argument('--prefill',type=int,default=1024)
    parser.add_argument('--state-cache-slots',type=int,default=1)
    parser.add_argument('--adapt-every',type=int,default=32)
    parser.add_argument('--suite',choices=['smoke','quality','performance'],default='quality')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--tokens',type=int,default=512)
    args=parser.parse_args()
    if args.output.exists(): raise FileExistsError(args.output)
    extra=dict(moe_q2_resident_experts=args.resident,moe_cache_size=args.resident+2048,
        moe_q2_dynamic_slots=1024,memory_ratio=.9,kv_cache_tokens=8192,max_prefill_length=args.prefill,
        mtp_compact_state=args.compact_mtp,mtp_draft_policy=args.mtp_draft_policy,linear_state_cache_slots=args.state_cache_slots,moe_q2_adapt_every=args.adapt_every,
        moe_q2_adapt_decay=0.7,gguf_embedding_host=True,ple_row_cache_rows=1048576,
        moe_q2_prefill_workspace=True,fixed_q2_rebuild_reserve_mib=0,
        vram_estimate_bytes=int(args.vram_gib*2**30),host_commit_estimate_bytes=int(args.host_gib*2**30),
        startup_timeout_seconds=600,request_deadline_seconds=1800,shutdown_timeout_seconds=20)
    if args.mtp_runtime: extra.update(mtp_enabled=True,mtp_runtime_path=str(args.mtp_runtime))
    backend=FreeTokenQwen36Backend()
    spec=ModelSpec(model_id='flash-next-5070ti',weights_path=str(args.model),quantization='q2_0_strata')
    report=dict(scope='speed_source_with_ordinary_freetoken_wheel_not_installer',status='loading',args={k:str(v) for k,v in vars(args).items()},runs=[],
        python=sys.executable,speed_origin=silitok_speed.__spec__.origin,freetoken=metadata.version('freetoken'),
        protocol='Arithmetic warmup, then fixed suite; no expert or prefix reset. '
                 'Performance repeats each prompt immediately and counts only delivered tokens.',
        options=extra)
    report['identity']=runpy.run_path(str(Path(__file__).with_name('run-identity.py')))['capture'](silitok_speed.__spec__.origin)
    report['probe_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    def save(): args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    save();started=time.perf_counter();owned=set()
    baseline={r['name']:r for r in archived('quality-gate-dev7/freetoken.json.gz')['runs']}
    cases=archived('quality-gate-dev7/protocol.json.gz')['cases']
    prompts=archived('mtp2-native-performance-v2.prompts.json.gz')
    def run(name,messages,limit,thinking=False,case=None):
        start=time.perf_counter();arrivals=[];final=None
        for event in backend.generate(dict(messages=messages,enable_thinking=thinking),
                max_new_tokens=limit,temperature=0,top_p=1):
            if event.get('type')=='token': arrivals.append(time.perf_counter())
            if event.get('type')=='final': final=event
        if final is None or not arrivals: raise RuntimeError('Missing actual output or terminal response')
        count=final['usage']['output_tokens']
        if count!=len(final['tokens']): raise RuntimeError('Output token accounting mismatch')
        row=dict(name=name,final=final,elapsed_s=time.perf_counter()-start,ttft_ms=1000*(arrivals[0]-start),
            output_tokens=count,decode_tps=(count-1)/(arrivals[-1]-arrivals[0]) if count>1 else None,
            events=len(arrivals))
        if case:
            if final['usage']['input_tokens']!=case['input_tokens']: raise RuntimeError('Frozen input token count changed')
            if case['check_kind']=='json':
                try: row['quality_pass']=json.loads(final['text'])==case['expected']
                except ValueError: row['quality_pass']=False
            else: row['quality_pass']=final['text'].strip()==case['expected']
            if name in baseline: row['matches_historical_ids']=final['tokens']==baseline[name]['output']['token_ids']
        report['runs'].append(row);save()
        print(json.dumps({k:v for k,v in row.items() if k!='final'},ensure_ascii=False),flush=True)
        return row
    try:
        backend.load(spec,BackendConfig(device='cuda:0',offload=False,extra=extra))
        report.update(load_s=time.perf_counter()-started,worker_pid=backend.worker_pid,
            expert_storage=backend.expert_storage(),cache_geometry=backend.cache_geometry(),
            mtp_identity=backend.mtp_identity(),status='running')
        owned=_worker_process_tree(backend.worker_pid);save()
        warm=run('warmup',cases[0]['messages'],64,case=cases[0])
        if warm['final']['text'].strip()!='527': raise RuntimeError('First arithmetic failed')
        if args.suite=='quality':
            for case in cases: run(case['name'],case['messages'],case['max_tokens'],case=case)
            for case in archived('quality-thinking-dev3/protocol.json.gz')['cases']:
                run('thinking-'+case['name'],case['messages'],case['max_tokens'],True,case)
            report['score']=sum(r.get('quality_pass',False) for r in report['runs'] if r['name'] in baseline)
            report['thinking_score']=sum(r.get('quality_pass',False) for r in report['runs'] if r['name'].startswith('thinking-'))
            report['regressions']=[r['name'] for r in report['runs'] if r['name'] in baseline
                and baseline[r['name']]['quality_pass'] and not r['quality_pass']]
            if report['score']<15 or report['thinking_score']!=3 or report['regressions']:
                raise RuntimeError('Quality fell below the frozen Q2 baseline (15/20 and thinking 3/3)')
        elif args.suite=='performance':
            for key in ('70-0','diverse-4096-0'):
                for repeat in range(3):
                    run(key+'-'+str(repeat),[dict(role='user',content=prompts[key]['text'])],args.tokens)
            report['medians']={key:statistics.median(r['decode_tps'] for r in report['runs'] if r['name'].startswith(key))
                               for key in ('70-0','diverse-4096-0')}
        else:
            run('context4096',[dict(role='user',content=prompts['diverse-4096-0']['text'])],32)
        cancel_started=time.perf_counter()
        report['cancel']=_cancel_once(backend)
        report['cancel_total_s']=time.perf_counter()-cancel_started
        after=run('after_cancel',cases[0]['messages'],64,case=cases[0])
        if after['final']['tokens']!=warm['final']['tokens']: raise RuntimeError('Cancel reuse changed output')
        report['status']='passed'
        report['worker_stderr_tail']=backend._transport.stderr_tail()
    except BaseException as error:
        report['status']='failed'
        report['error']=repr(error)
        if backend._transport is not None:
            report['worker_stderr_tail']=backend._transport.stderr_tail()
        raise
    finally:
        backend.unload()
        report['worker_pids']=sorted(owned)
        report['worker_pids_gone']=bool(owned) and all(not psutil.pid_exists(pid) for pid in owned)
        save()
    if not report['worker_pids_gone']: raise RuntimeError('Owned worker tree did not exit')
    print(json.dumps(dict(status=report['status'],output=str(args.output),worker_pids_gone=True)))


if __name__=='__main__': main()
