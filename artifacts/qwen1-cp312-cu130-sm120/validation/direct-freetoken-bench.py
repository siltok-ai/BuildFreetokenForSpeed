"""直接测固定 FreeToken 的生成路径，不经过 Speed worker 或 IPC。"""
import argparse
import collections
import json
import os
from pathlib import Path
import statistics
import time

p=argparse.ArgumentParser()
p.add_argument('--output', type=Path, required=True)
p.add_argument('--model', type=Path, required=True)
p.add_argument('--kernel-cache', type=Path, required=True)
a=p.parse_args()
os.environ['FREETOKEN_KERNEL_CACHE_DIR']=str(a.kernel_cache)
os.environ['FREETOKEN_DISABLE_JIT']='1'
import torch
from freetoken.llm import LLM
from freetoken.core import SamplingParams
from freetoken.message import DetokenizeMsg
from freetoken.mm.config import ENCODER_KINDS, MultimodalConfig

class TimedLLM(LLM):
    """仅在官方输出回调记录到达时间，计算与 Speed 相同的 decode 口径。"""
    def offline_send_result(self, reply):
        for msg in reply:
            if isinstance(msg, DetokenizeMsg):
                self.arrivals.append(time.perf_counter())
        super().offline_send_result(reply)

kwargs=dict(max_running_req=1,attention_backend='triton',moe_strategy='fused',
    mm=MultimodalConfig(disabled_encoders=frozenset(ENCODER_KINDS)),
    quant_backend='moe.nvfp4=triton',expert_load='auto',moe_cache_size=0,
    moe_cache_auto=False,kv_reserve_tokens=0,num_token_override=4096,
    cuda_graph_max_bs=1,memory_ratio=0.90,max_seq_len_override=262144,
    max_extend_tokens=4096,cache_type='radix',decode_log_interval=1000)
t=time.perf_counter()
llm=TimedLLM(str(a.model),**kwargs)
result={'load_seconds':time.perf_counter()-t,'engine_kwargs':{k:str(v) for k,v in kwargs.items()},'runs':[]}
model=llm.engine.model
inventory=[]
for name,tensor in model.state_dict().items():
    inventory.append({'name':name,'device':str(tensor.device),'dtype':str(tensor.dtype),'bytes':tensor.numel()*tensor.element_size()})
result['tensor_devices']=dict(collections.Counter(item['device'] for item in inventory))
result['tensor_bytes_by_device']={d:sum(x['bytes'] for x in inventory if x['device']==d) for d in result['tensor_devices']}
result['cpu_tensors']=[x for x in inventory if not x['device'].startswith('cuda')]
from freetoken.layers.base import BaseOP
ops=[]
def visit(op):
    """遍历官方 BaseOP，记录实际量化方法和后端。"""
    ops.append(op)
    for child in vars(op).values():
        if isinstance(child,BaseOP): visit(child)
        elif isinstance(child,list):
            for item in child:
                if isinstance(item,BaseOP): visit(item)
visit(model)
result['linear_methods']=dict(collections.Counter(type(m.quant_method).__module__+'.'+type(m.quant_method).__name__ for m in ops if hasattr(m,'quant_method')))
result['linear_backend_attributes']=[{'class':type(m.quant_method).__name__,'attributes':{k:str(v) for k,v in vars(m.quant_method).items()}} for m in ops if hasattr(m,'quant_method')][:8]
result['cuda_allocated_bytes']=torch.cuda.memory_allocated()
result['cuda_reserved_bytes']=torch.cuda.memory_reserved()
prompt='Write a detailed technical article about how a local inference engine balances GPU memory, host memory, and token latency. Include concrete examples and continue until the article is complete.'
tokens=llm.tokenizer.apply_chat_template([{'role':'user','content':prompt}],add_generation_prompt=True,tokenize=True,enable_thinking=False,return_dict=False)
for count in [16,128,128,128,512,512,512]:
    llm.arrivals=[]
    started=time.perf_counter()
    generated=llm.generate([tokens],SamplingParams(temperature=0,top_k=1,top_p=1,max_tokens=count,ignore_eos=False))[0]
    ended=time.perf_counter()
    arrivals=llm.arrivals
    run={'max_tokens':count,'input_tokens':len(tokens),'output_tokens':len(generated['token_ids']),
         'events':len(arrivals),'ttft_ms':1000*(arrivals[0]-started),
         'decode_tps':(len(arrivals)-1)/(arrivals[-1]-arrivals[0]),
         'wall_seconds':ended-started,'text_prefix':generated['text'][:120]}
    result['runs'].append(run)
    print(json.dumps(run),flush=True)
for count in [128,512]:
    result[f'median_{count}_decode_tps']=statistics.median(x['decode_tps'] for x in result['runs'] if x['max_tokens']==count)
a.output.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
print(json.dumps({k:v for k,v in result.items() if k!='runs'},indent=2),flush=True)
