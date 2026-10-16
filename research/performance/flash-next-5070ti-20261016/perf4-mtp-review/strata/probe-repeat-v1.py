"""对固定Strata二进制做stdin流式实测；不启动网络服务，保留真实token和资源侧车。"""
from pathlib import Path
import argparse,hashlib,json,os,queue,subprocess,sys,threading,time,gzip

def main():
    """每次仅拥有一个Strata进程；预热后按冻结IDs发请求，最后QUIT并验证退出。"""
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--mtp',action='store_true');p.add_argument('--profile',type=Path);p.add_argument('--repetitions',type=int,default=1);p.add_argument('--keys',nargs='+',default=['english','chinese','code']);p.add_argument('--slots',default='auto');p.add_argument('--pcie',type=float);p.add_argument('--adapt-every',type=int,default=4);p.add_argument('--suffix',type=int,default=3);p.add_argument('--exe',type=Path,default=Path(r'C:\workspace\mats\strata\flash-perf-01\strata.exe'))
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    binary_sha=hashlib.sha256(a.exe.read_bytes()).hexdigest()
    if binary_sha!='9832fcaffd5fc389963bf391ad2df5700d43399da7a85f9fcf42d65d82c58f76':raise RuntimeError('Expected fixed unmodified Strata baseline')
    if a.profile:raise ValueError('Baseline repetition probe does not instrument profiling')
    root=Path(r'C:\workspace');src=root/'wt/strata/flash-perf';model=root/'models/siltok-speed/flash-next/sources/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF/ed59f92082b1e93c0e96d60a8b11aab089b52f09/Q2_0/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf'
    cmd=[str(a.exe),'--pack',str(root/'models/strata/flash-next-q2-reference'),'--native',str(model),'--serve','--max-context','8192','--expert-cache',a.slots,'--expert-profile',str(src/'data/expert-profile.bin'),'--resident-experts','--prefill','4096','--vram-reserve-mib','1024','--spec','4','--suffix-draft',str(a.suffix if a.mtp else 0),'--adapt-every',str(a.adapt_every)]
    if a.mtp:cmd+=['--mtp',str(root/'models/strata/flash-next-mtp-reference')]
    if a.pcie is not None:cmd+=['--pcie-frac',str(a.pcie)]
    if a.profile:
        os.environ['STRATA_PERF_TRACE']='1'
        cmd=[r'C:\workspace\tools\n51\ProgramFiles64Folder\NVIDIA Corporation\Nsight Systems 2026.5.1\target-windows-x64\nsys.exe','profile','--trace=cuda,nvtx','--sample=none','--cpuctxsw=none','--cuda-graph-trace=node','--show-output=true','--capture-range=cudaProfilerApi','--capture-range-end=stop','--output',str(a.profile),*cmd]
    prompts=json.loads((root/'runs/siltok-speed/flash-perf-20261016/baseline-mtp-workloads-v3.json').read_text())['input_ids']
    std=json.loads(gzip.decompress((root/'wt/siltok-speed/strata-review/research/strata-q2-mtp2-20261014/raw/quality-gate-dev7/protocol.json.gz').read_bytes()))['cases'][0]
    sys.path.insert(0,str(src/'tools'))
    from strata_tokenizer import Tokenizer
    tokenizer=Tokenizer.from_gguf(model)
    report=dict(status='loading',command=cmd,binary_sha256=hashlib.sha256(a.exe.read_bytes()).hexdigest(),source_commit='6674a0065fb96bacde33e3eb10f91a1df86f95f2',probe_worktree_commit=subprocess.check_output(['git','-C',str(src),'rev-parse','HEAD'],text=True).strip(),input_ids=prompts,runs=[],model=str(model),scope='same_full_GSQ_Q2_weights; runtime arithmetic/cache/KV/protocol differ; actual stdin token delivery')
    def save():a.output.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    save();q=queue.Queue()
    with a.output.with_suffix('.engine.log').open('w',encoding='utf-8') as log:
        proc=subprocess.Popen(cmd,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=log,text=True,encoding='utf-8',bufsize=1,creationflags=subprocess.CREATE_NO_WINDOW)
        def reader():
            """在独立读线程标记实际到达时刻，避免主线程处理报告造成时间偏移。"""
            for line in proc.stdout:q.put((time.perf_counter(),line.rstrip()))
            q.put((time.perf_counter(),None))
        threading.Thread(target=reader,daemon=True).start()
        def line(timeout=600):
            stamp,s=q.get(timeout=timeout)
            if s is None:
                proc.wait(timeout=5)
                raise RuntimeError(f'Strata exited {proc.returncode}')
            if s.startswith('ERR'):raise RuntimeError(s)
            return stamp,s
        try:
            start=time.perf_counter()
            while True:
                stamp,s=line()
                if s.startswith('READY '):break
            report.update(status='running',load_s=stamp-start,engine_pid=proc.pid);save()
            def request(name,ids,limit):
                """实际token数、终止字段和逐token时序都保留；不把提前EOS换成请求长度。"""
                times=[];tokens=[];meta=[];start=time.perf_counter();proc.stdin.write(f'GEN {limit} '+','.join(map(str,ids))+'\n');proc.stdin.flush()
                while True:
                    stamp,s=line()
                    if s.startswith('T '):tokens.append(int(s[2:]));times.append(stamp-start)
                    elif s.startswith('DONE '):
                        done=s.split();break
                    else:meta.append(s)
                if len(tokens)!=int(done[1]):raise RuntimeError('Strata terminal count mismatch')
                row=dict(name=name,text=tokenizer.decode(tokens),input_tokens=len(ids),output_tokens=len(tokens),token_ids=tokens,arrival_offsets=times,done=done,metadata=meta,ttft_ms=1000*times[0],decode_tps=(len(tokens)-1)/(times[-1]-times[0]) if len(tokens)>1 else None,elapsed_s=stamp-start)
                report['runs'].append(row);save();print(json.dumps({k:v for k,v in row.items() if k not in ('token_ids','arrival_offsets','metadata','text')},ensure_ascii=False),flush=True)
                if name!='warmup' and len(tokens)!=limit:raise RuntimeError('Strata did not deliver all requested tokens')
                return row
            warm=request('warmup',std['tokens'],64)
            report['warmup_expected']='527';report['warmup_token_ids']=warm['token_ids']
            text=tokenizer.decode([t for t in warm['token_ids'] if t!=248046]).strip();report['warmup_text']=text
            if text!='527':raise RuntimeError(f'Arithmetic failed: {text!r}')
            for key in a.keys:
                for repeat in range(a.repetitions):request(f'{key}-{repeat}',prompts[key],512)
            report['status']='passed'
        except BaseException as e:report.update(status='failed',error=repr(e));raise
        finally:
            if proc.poll() is None:
                try:
                    proc.stdin.write('QUIT\n');proc.stdin.flush()
                except (BrokenPipeError,OSError):
                    pass
                try:proc.wait(timeout=30)
                except subprocess.TimeoutExpired:proc.kill();proc.wait();report['forced_shutdown']=True
            report.update(returncode=proc.returncode,shutdown_complete=proc.poll() is not None);save()
if __name__=='__main__':main()
