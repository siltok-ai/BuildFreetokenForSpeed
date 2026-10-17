from pathlib import Path
import json
import os
import runpy
import subprocess
R=Path(__file__).parent
ns=runpy.run_path(str(R/'run-source-acceptance.py'))
P,S,T,C=map(ns.get,('P','S','T','C'))
os.environ.update(FREETOKEN_KERNEL_CACHE_DIR=str(C),SILITOK_FREETOKEN_KERNEL_CACHE_DIR=str(C),FREETOKEN_DISABLE_JIT='1',TRITON_CACHE_DIR=r'C:\workspace\cache\triton\flash-perf-mtp8',PYTHONUTF8='1',PYTHONDONTWRITEBYTECODE='1')
stem='review1-source-qwen36-backend-v2'
ns['run'](stem,[P,'-B',T/'qwen36-legacy-probe.py','--mode','backend','--source',S,'--model',r'C:\workspace\models\siltok-speed\languages\qwen3.6-35b-a3b.nvfp4-modelscope','--profile','qwen36','--quant','nvfp4','--kernel-cache',C,'--repeats',1,'--long-repeats',1,'--tokens',64,'--long-prompt-tokens',256,'--long-output-tokens',16,'--quality','--output',R/(stem+'.json')])
subprocess.run([str(P),'-B',str(T/'audit-qwen-regression.py'),'--report',str(R/(stem+'.json')),'--baseline',str(T/'qwen36-ft2-backend-v1.json'),'--output',str(R/'review1-source-qwen36-audit-v2.json')],check=True)
(R/'source-acceptance-complete.json').write_text(json.dumps({'status':'passed','old35B':'review1-source-qwen36-backend-v2.json'})+'\n')
