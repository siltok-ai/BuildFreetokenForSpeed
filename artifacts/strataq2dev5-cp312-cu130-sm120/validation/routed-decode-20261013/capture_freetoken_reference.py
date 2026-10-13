"""Capture installed GGUF engine's actual residuals and logits for fixed token IDs."""
import hashlib
import json
import os
from pathlib import Path
ROOT=Path(__file__).resolve().parent
os.environ['FREETOKEN_DISABLE_JIT']='1'
os.environ['TOKENIZERS_PARALLELISM']='false'
os.environ.setdefault('FREETOKEN_KERNEL_CACHE_DIR',
    'C:/workspace/worktrees/build-freetoken-flash-next-strata-q2/artifacts/strataq2dev2-cp312-cu130-sm120/kernel-cache')
os.environ.setdefault('TRITON_CACHE_DIR',str(ROOT/'candidate-triton-cache'))
if os.name=='nt' and not hasattr(os,'posix_fadvise'):
    os.posix_fadvise=lambda *_:None
    os.POSIX_FADV_DONTNEED=0

def main():
    import torch
    import freetoken
    import freetoken.kernel.backend as backend
    backend.is_flashinfer_installed=lambda:False
    from freetoken.core import SamplingParams
    from freetoken.llm.llm import LLM
    from freetoken.mm.config import MultimodalConfig,ENCODER_KINDS
    from safetensors.torch import save_file
    tokens=[int(t) for t in (ROOT/'strata-reference-quality.tokens').read_text().split(',')]
    digest=hashlib.sha256(json.dumps(tokens).encode()).hexdigest()
    existing=json.loads((ROOT/'candidate-dev2-wheel-matrix.json').read_text(encoding='utf-8'))['runs'][0]
    if existing['input_sha256']!=digest:raise RuntimeError('fixed tokens differ from previous quality probe')
    llm=LLM(str(ROOT/'gguf-checkpoint'),dtype=torch.bfloat16,max_running_req=1,
        attention_backend='qsa_sparse',moe_strategy='offload',ple_backend='disk',expert_load='serial',
        mm=MultimodalConfig(disabled_encoders=frozenset(ENCODER_KINDS)),moe_cache_size=1024,
        moe_cache_auto=False,kv_reserve_tokens=8192,num_token_override=8192,cuda_graph_max_bs=0,
        memory_ratio=.8,max_seq_len_override=262144,max_extend_tokens=4096,cache_type='radix',decode_log_interval=1000)
    captured={}
    for layer in llm.engine.model.model.layers.op_list:
        original=layer.forward
        def wrap(x,*args,_original=original,_id=layer._layer_id,**kwargs):
            if _id==0:captured['residual_input']=x.detach().cpu().clone()
            y=_original(x,*args,**kwargs)
            captured[f'layer_{_id:02d}']=y.detach().cpu().clone()
            return y
        layer.forward=wrap
    sample=llm.engine.sampler.sample
    def sampling(logits,params):
        captured['logits']=logits.detach().cpu().clone()
        return sample(logits,params)
    llm.engine.sampler.sample=sampling
    output=llm.generate([tokens],SamplingParams(temperature=0,top_p=1,max_tokens=1))[0]
    target=ROOT/'freetoken-dev2-quality-reference.safetensors'
    save_file(captured,str(target))
    report=dict(version=freetoken.__version__,source=freetoken.__file__,input_sha256=digest,
        input_tokens=len(tokens),capture=str(target),capture_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
        shapes={k:list(v.shape) for k,v in captured.items()},output=output,
        scope='Installed FreeToken actual fixed-token layer residuals and logits; comparison pending')
    (ROOT/'freetoken-dev2-quality-reference.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k!='shapes'},ensure_ascii=False))

if __name__=='__main__':main()
