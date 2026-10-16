"""新缓存中对照48层首次编译与热图成本，逐字节核对全部准入状态。"""
from pathlib import Path
import json
import os
import time
import torch
from freetoken.moe.q2_residency import _ensure as original
from admission_runtime_base_variant import _ensure_runtime_base as candidate

RUN=Path(__file__).parent
cache=Path(os.environ['TRITON_CACHE_DIR'])
assert not list(cache.rglob('_ensure.json')) and not list(cache.rglob('_ensure_runtime_base.json'))
seed=(torch.arange(60,device='cuda')%10).int()


def state():
    """仅分配元数据，使用真实48层/512专家/2048动态槽和60残余路由几何。"""
    return dict(ids=seed.repeat(48,1),mapping=torch.full((48*512,),-1,device='cuda',dtype=torch.int32),
        dynamic=torch.full((2048,),-1,device='cuda',dtype=torch.int32),
        usage=torch.zeros(2048,device='cuda',dtype=torch.int64),step=torch.zeros(1,device='cuda',dtype=torch.int64),
        src=torch.full((256,),-1,device='cuda',dtype=torch.int32),dst=torch.full((256,),-1,device='cuda',dtype=torch.int32),
        count=torch.zeros(1,device='cuda',dtype=torch.int64),stats=torch.zeros(4,device='cuda',dtype=torch.int64))


def invoke(fn,values,reset=False):
    """所有动态参数与整数归约一致，唯一差异是层偏移的编译期/运行时选择。"""
    for layer in range(48):
        if reset:values['ids'][layer].copy_(seed)
        fn[(1,)](values['ids'][layer],values['mapping'],values['dynamic'],values['usage'],
            values['step'],values['src'],values['dst'],values['count'],values['stats'],
            layer*512,0,2048,2048,60,64,2048,True,num_warps=4)


def main():
    """冷首次调用计wall，热态只计CUDA图；两者不混称整模型性能。"""
    report=dict(scope='metadata_only_cold_JIT_and_warm_graph_microbench_not_model_tps',cases={})
    states={}
    for name,fn in [('original',original),('runtime_base',candidate)]:
        values=state();torch.cuda.synchronize()
        start=time.perf_counter();invoke(fn,values);torch.cuda.synchronize()
        cold=time.perf_counter()-start
        states[name]={key:value.clone() for key,value in values.items()}
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):invoke(fn,values,True)
        for _ in range(10):graph.replay()
        a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        a.record()
        for _ in range(200):graph.replay()
        b.record();b.synchronize()
        report['cases'][name]=dict(cold_48_layers_seconds=cold,warm_graph_ms_per48_layers=a.elapsed_time(b)/200)
        graph.reset()
    for key in states['original']:
        assert torch.equal(states['original'][key],states['runtime_base'][key]),key
    report['all_metadata_bytes_equal']=True
    report['compiled_counts']=dict(original=len(list(cache.rglob('_ensure.json'))),
                                  runtime_base=len(list(cache.rglob('_ensure_runtime_base.json'))))
    assert report['compiled_counts']==dict(original=48,runtime_base=1)
    report['status']='passed'
    path=RUN/'admission-runtime-base-cold-v1.json'
    if path.exists():raise FileExistsError(path)
    path.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report))


if __name__=='__main__':main()
