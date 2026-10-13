"""Frozen Strata persistent-engine benchmark, timestamping actual stdout token delivery."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import queue
import statistics
import subprocess
import threading
import time

ROOT = Path(__file__).resolve().parent

def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', required=True)
    p.add_argument('--cache', default='1024')
    p.add_argument('--pcie-frac', default='1')
    p.add_argument('--resident', action='store_true')
    p.add_argument('--prefill', default='auto')
    p.add_argument('--tokens', type=int, default=512)
    p.add_argument('--repetitions', type=int, default=3)
    a = p.parse_args()
    out = Path(a.output)
    if out.exists():
        raise FileExistsError(out)
    base = json.loads((ROOT/'strata-s2-quality-reference-v4-resource.json').read_text(encoding='utf-8'))['command']
    command = base[:base.index('--tokens-file')]
    # Frozen serve requires a >=2-row verify buffer even when no drafter is enabled.
    # Actual DONE draft counts below must remain zero; every emitted token is verified.
    command += ['--serve', '--spec', '2', '--suffix-draft', '0', '--max-context', '8192', '--expert-cache', a.cache,
                '--expert-cache-per-layer', '--prefill', a.prefill, '--prompt-cache', '6', '--stats',
                '--expert-profile', str(ROOT/'strata-official-expert-profile.bin')]
    command += ['--resident-experts' if a.resident else '--mmap-experts']
    if a.pcie_frac != 'default':
        command += ['--pcie-frac', a.pcie_frac]
    env = os.environ.copy()
    env['STRATA_DECODE_TIMING'] = '1'
    env['STRATA_VERIFY_PROFILE'] = '1'
    env['PATH'] = 'C:/workspace/tools/cuda-13.0/bin/x64;' + env['PATH']
    report = dict(status='running', args=vars(a), command=command, runs=[], no_mtp=True,
                  executable_sha256=hashlib.sha256(Path(command[0]).read_bytes()).hexdigest(),
                  protocol='Actual streamed T records; decode=(visible N-1)/(last-first), EOS excluded, startup/prefill excluded. Greedy, fixed historical prompt IDs. Expert cache persists between requests; cold labels mean distinct prompt, not an explicit expert-cache reset. OS file cache is not purged. Native CUDA-event stage profiling enabled. Diagnostic only; instrumented throughput is not a sustained speed benchmark.')
    write(out, report)
    events = queue.Queue()
    with out.with_suffix('.native.stdout.log').open('w', encoding='utf-8') as stdout, out.with_suffix('.native.stderr.log').open('w', encoding='utf-8') as stderr:
        child = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=stderr,
                                 env=env, text=True, encoding='utf-8', errors='strict', bufsize=1)
        def reader():
            try:
                for line in child.stdout:
                    now = time.perf_counter()
                    stdout.write(line)
                    stdout.flush()
                    events.put((now, line.strip()))
            finally:
                events.put((time.perf_counter(), None))
        worker = threading.Thread(target=reader, daemon=True)
        worker.start()
        def get():
            now, line = events.get(timeout=600)
            if line is None or line.startswith(('ERR ', 'FATAL ')):
                raise RuntimeError('Strata engine stopped or rejected request: '+str(line))
            return now, line
        try:
            start = time.perf_counter()
            startup = []
            while True:
                now, line = get()
                startup.append(line)
                if line.startswith('READY '):
                    break
            report['startup_seconds'] = now-start
            report['startup_protocol'] = startup
            prompts = json.loads((ROOT/'prompts.json').read_text(encoding='utf-8'))
            from transformers import AutoTokenizer
            tok = AutoTokenizer.from_pretrained(str(ROOT/'gguf-checkpoint'), local_files_only=True)
            def run(case, label, limit):
                ids = case['ids']
                started = time.perf_counter()
                child.stdin.write('GEN '+str(limit)+' '+','.join(map(str, ids))+'\n')
                child.stdin.flush()
                tokens, arrivals, lines = [], [], []
                while True:
                    now, line = get()
                    lines.append(dict(ms=1000*(now-started), line=line))
                    if line.startswith('T '):
                        tokens.append(int(line.split()[1]))
                        arrivals.append(now)
                    if line.startswith('DONE '):
                        done = line.split()[1:]
                        break
                if int(done[0]) != len(tokens) or int(done[1]) != len(ids):
                    raise RuntimeError('Native token accounting mismatch')
                if int(done[5]) or int(done[6]):
                    raise RuntimeError('Unexpected speculative decode')
                eos = bool(tokens and tokens[-1] in (248044, 248046))
                visible = tokens[:-1] if eos else tokens
                ts = arrivals[:-1] if eos else arrivals
                gaps = [1000*(b-a) for a,b in zip(ts, ts[1:])]
                return dict(label=label, input_tokens=len(ids),
                    input_sha256=hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
                    output=dict(token_ids=visible, text=tok.decode(visible)), output_tokens=len(visible),
                    native_generated_tokens=len(tokens), eos=eos, truncated=not eos and len(tokens)==limit,
                    finish_reason=done[4], decode_tps=(len(ts)-1)/(ts[-1]-ts[0]) if len(ts)>1 else None,
                    ttft_ms=1000*(ts[0]-started) if ts else None,
                    native_prompt_ms=float(done[2]), native_decode_ms=float(done[3]),
                    prefix_cached_tokens=int(done[7]), expert_hits=int(done[8]), expert_lookups=int(done[9]),
                    expert_offloaded=int(done[14]),
                    itl_median_ms=statistics.median(gaps) if gaps else None,
                    arrivals_ms=[1000*(t-started) for t in ts], protocol_records=lines,
                    measurement_sufficient=len(visible)>=256)
            report['warmup'] = run(prompts['70-3'], 'warmup', 8)
            write(out, report)
            for context in ('short','diverse4096'):
                for rep in range(a.repetitions):
                    case = prompts[f'70-{rep}' if context=='short' else f'diverse-4096-{rep}']
                    for hot in (False, True):
                        row = run(case, f'{context}-{rep}-'+('hot' if hot else 'cold'), a.tokens)
                        if hot:
                            row['cold_hot_token_ids_equal'] = row['output']['token_ids']==report['runs'][-1]['output']['token_ids']
                        report['runs'].append(row)
                        write(out, report)
                        print(json.dumps({k:v for k,v in row.items() if k not in ('output','protocol_records','arrivals_ms')}, ensure_ascii=False), flush=True)
            report['quality_smoke'] = run(prompts['quality'], 'quality', 64)
            report['quality_pass'] = report['quality_smoke']['output']['text'].strip()=='703'
            report['thinking_smoke'] = []
            report['thinking_skipped_reason'] = 'This is stage attribution only; existing frozen thinking quality evidence remains separate'
            report['all_measurements_sufficient'] = all(r['measurement_sufficient'] for r in report['runs'])
            child.stdin.write('QUIT\n'); child.stdin.flush()
            report['returncode'] = child.wait(timeout=60)
            if report['returncode']:
                raise RuntimeError('Native engine failed shutdown')
            report['status'] = 'diagnostic_gpu_stages_complete'
            report['profiling_env'] = dict(STRATA_DECODE_TIMING='1', STRATA_VERIFY_PROFILE='1')
            report['speed_acceptance'] = False
            write(out, report)
        except BaseException as exc:
            report['status']='failed';report['error']=repr(exc);write(out, report)
            child.terminate();child.wait(timeout=30)
            raise

if __name__=='__main__':
    main()
