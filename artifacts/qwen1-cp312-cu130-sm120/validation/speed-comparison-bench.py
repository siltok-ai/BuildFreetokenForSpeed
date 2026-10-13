"""用独立 FreeToken 测试的同一个 prompt 和相同 token 上限测 Speed。"""
import argparse
import importlib.util
import json
from pathlib import Path
import statistics
import sys
import time

p=argparse.ArgumentParser()
p.add_argument('--source',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--model',type=Path,required=True)
p.add_argument('--packaged',action='store_true')
p.add_argument('--kernel-cache',type=Path,required=True)
a=p.parse_args()
if not a.packaged: sys.path.insert(0,str(a.source))
import silitok_speed
if a.packaged and not silitok_speed.__spec__.origin.endswith('.pyd'): raise RuntimeError('not compiled Speed')
spec=importlib.util.spec_from_file_location('bench',a.source/'tools/bench/benchmark_freetoken_qwen36.py')
bench=importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)
backend=bench.FreeTokenQwen38Backend()
model=bench.ModelSpec(model_id='qwen3.8-27b.nvfp4',weights_path=str(a.model),quantization='nvfp4')
config=bench.BackendConfig(device='cuda:0',offload=False,extra={'kernel_cache_dir':str(a.kernel_cache),'startup_timeout_seconds':300,'request_deadline_seconds':1800,'shutdown_timeout_seconds':20})
prompt='Write a detailed technical article about how a local inference engine balances GPU memory, host memory, and token latency. Include concrete examples and continue until the article is complete.'
result={'speed_origin':silitok_speed.__spec__.origin,'runs':[]}
try:
    t=time.perf_counter()
    backend.load(model,config)
    result['load_seconds']=time.perf_counter()-t
    result['vram_bytes']=backend.vram_current()
    for count in [16,128,128,128,512,512,512]:
        run=bench._generate_once(backend,prompt,max_tokens=count)
        run['max_tokens']=count
        result['runs'].append(run)
        print(json.dumps({k:v for k,v in run.items() if k not in {'text','reasoning_content'}},ensure_ascii=True),flush=True)
    for count in [128,512]:
        result[f'median_{count}_decode_tps']=statistics.median(x['decode_tokens_per_second'] for x in result['runs'] if x['max_tokens']==count)
finally:
    backend.unload()
a.output.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
