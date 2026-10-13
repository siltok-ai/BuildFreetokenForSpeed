"""Run frozen twenty-item local quality protocol; one big model at a time."""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'quality-gate-dev7'

def write(p,value):p.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')
def grade(case,text):
    if case['check_kind']=='json':
        try:return json.loads(text)==case['expected']
        except (ValueError,TypeError):return False
    return text.strip()==case['expected']

def run_freetoken(mode):
    import capture_freetoken_reference
    import torch
    import freetoken
    import freetoken.kernel.backend as backend
    backend.is_flashinfer_installed=lambda:False
    from freetoken.core import SamplingParams
    from freetoken.llm.llm import LLM
    from freetoken.mm.config import MultimodalConfig,ENCODER_KINDS
    from freetoken.message import DetokenizeMsg
    target=OUT/(mode+'.json')
    if target.exists():raise FileExistsError(target)
    if freetoken.__version__!='0.1.3+speed.strataq2dev7':raise RuntimeError('wrong installed wheel')
    model=ROOT/'gguf-checkpoint' if mode=='freetoken' else Path('C:/workspace/artifacts/freetoken-q2-review-20261011/checkpoint-finalize')
    llm=LLM(str(model),dtype=torch.bfloat16,max_running_req=1,
        attention_backend='qsa_sparse',moe_strategy='offload',ple_backend='disk',expert_load='serial',
        mm=MultimodalConfig(disabled_encoders=frozenset(ENCODER_KINDS)),moe_cache_size=11264,moe_q2_resident_experts=9216,moe_q2_dynamic_slots=1024,
        moe_cache_auto=False,kv_reserve_tokens=8192,num_token_override=8192,cuda_graph_max_bs=1,
        memory_ratio=.9,max_seq_len_override=262144,max_extend_tokens=4096,cache_type='radix',decode_log_interval=1000)
    protocol=json.loads((OUT/'protocol.json').read_text(encoding='utf-8'))
    report=dict(version=freetoken.__version__,source=freetoken.__file__,model=str(model),runs=[],scope=protocol['scope'])
    original=llm.send_result;terminal=[];events=[];fwd=llm.engine.forward_batch
    def send(msgs):
        terminal.extend(dict(token=int(m.next_token),finished=m.finished,finish_reason=m.finish_reason)
            for m in msgs if isinstance(m,DetokenizeMsg))
        return original(msgs)
    def forward(batch,params):
        if not batch.is_decode:events.append(dict(decode=False,requests=[dict(cached_len=r.cached_len,device_len=r.device_len) for r in batch.reqs]))
        return fwd(batch,params)
    llm.send_result=send;llm.engine.forward_batch=forward
    for case in protocol['cases']:
        terminal.clear();events.clear()
        output=llm.generate([case['tokens']],SamplingParams(temperature=0,top_p=1,max_tokens=case['max_tokens']))[0]
        if not terminal or not terminal[-1]['finished']:raise RuntimeError('missing actual terminal decision')
        eos=terminal[-1]['token'] in llm.eos_token_ids
        row=dict(name=case['name'],input_tokens=case['input_tokens'],input_sha256=case['sha256'],
            output=output,output_tokens=len(output['token_ids']),eos=eos,terminal=terminal[-1],
            truncated=not eos and len(output['token_ids'])==case['max_tokens'],prefill_events=list(events),
            quality_pass=grade(case,output['text']))
        report['runs'].append(row);write(target,report)
        print(json.dumps(row,ensure_ascii=False),flush=True)
    report['score']=sum(r['quality_pass'] for r in report['runs']);report['total']=len(report['runs']);write(target,report)
    llm.shutdown()

def run_strata():
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(str(ROOT/'gguf-checkpoint'),local_files_only=True)
    target=OUT/'strata.json'
    if target.exists():raise FileExistsError(target)
    cases=json.loads((OUT/'protocol.json').read_text(encoding='utf-8'))['cases']
    base=json.loads((ROOT/'strata-s2-quality-reference-v4-resource.json').read_text())['command']
    for flag in ('--dump-logits','--dump-layers','--dump-routing'):
        i=base.index(flag);del base[i:i+2]
    base.append('--stop-eos')
    env=os.environ.copy();env['PATH']='C:/workspace/tools/cuda-13.0/bin/x64;'+env['PATH']
    report=dict(runs=[],scope='Same-weight frozen Strata local quality reference; not speed measurement.')
    for case in cases:
        command=list(base)
        for flag,val in [('--tokens-file',str(OUT/(case['name']+'.tokens'))),('--max-new',str(case['max_tokens'])),
            ('--max-context',str(max(256,2**((case['input_tokens']+case['max_tokens']+64-1).bit_length()))))]:
            command[command.index(flag)+1]=val
        resource=OUT/(case['name']+'-strata-resource.json')
        if resource.exists():raise FileExistsError(resource)
        with (OUT/'strata-supervisor.log').open('a',encoding='utf-8') as log:
            rc=subprocess.call(['C:/workspace/envs/speed-q2-review/Scripts/python.exe',str(ROOT/'supervise.py'),
                '--output',str(resource),'--projected-gib','43','--',*command],env=env,stdout=log,stderr=log)
        if rc:raise RuntimeError(case['name']+' Strata job failed; see resource logs')
        stdout=resource.with_suffix('.stdout.log').read_text(encoding='utf-8')
        m=re.search(r'^output\s*:\s*([^\r\n]*)',stdout,re.M)
        if m is None:raise RuntimeError('missing reference output')
        ids=[int(t) for t in m.group(1).split()];visible=[t for t in ids if t not in (248044,248046)]
        eos=bool(ids and ids[-1] in (248044,248046));text=tok.decode(visible)
        row=dict(name=case['name'],input_tokens=case['input_tokens'],input_sha256=case['sha256'],
            output=dict(text=text,token_ids=visible),output_tokens=len(visible),eos=eos,
            truncated=not eos and len(ids)==case['max_tokens'],quality_pass=grade(case,text),resource=str(resource))
        report['runs'].append(row);write(target,report);print(json.dumps(row,ensure_ascii=False),flush=True)
    report['score']=sum(r['quality_pass'] for r in report['runs']);report['total']=len(report['runs']);write(target,report)

def compare():
    reports={name:json.loads((OUT/(name+'.json')).read_text(encoding='utf-8')) for name in ('freetoken','strata','mixed')}
    rows=[]
    for case in json.loads((OUT/'protocol.json').read_text(encoding='utf-8'))['cases']:
        actual={name:next(r for r in report['runs'] if r['name']==case['name']) for name,report in reports.items()}
        if any(r['input_sha256']!=case['sha256'] for r in actual.values()):raise RuntimeError('different input IDs')
        rows.append(dict(name=case['name'],scores={name:r['quality_pass'] for name,r in actual.items()},
            outputs={name:r['output']['text'] for name,r in actual.items()},
            output_tokens={name:r['output_tokens'] for name,r in actual.items()},
            eos={name:r['eos'] for name,r in actual.items()},truncated={name:r['truncated'] for name,r in actual.items()}))
    scores={name:report['score'] for name,report in reports.items()}
    report=dict(status='quality_gate_recorded_review_pending',scores=scores,total=20,runs=rows,
        candidate_only_failures=[r['name'] for r in rows if not r['scores']['freetoken'] and r['scores']['strata']],
        scope='Twenty local predeclared tasks. Mixed is a historical mixed-weight reference, not pure NVFP4.')
    write(OUT/'comparison.json',report);print(json.dumps({k:v for k,v in report.items() if k!='runs'},ensure_ascii=False))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['freetoken']);a=p.parse_args()
    if a.mode in ('freetoken','mixed'):run_freetoken(a.mode)
    elif a.mode=='strata':run_strata()
    else:compare()
