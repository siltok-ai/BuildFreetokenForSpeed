"""诊断实际长上下文冷/热路径的首个差异；不修改已冻结的 engine/wheel。"""
import json
from pathlib import Path
import torch

ROOT=Path(__file__).resolve().parent

def install(engine, output):
    cold={};rows=[];selections=[]
    def active():
        from freetoken.core import get_global_ctx
        batch=get_global_ctx().batch
        if batch is None or batch.is_decode or len(batch.reqs)!=1:
            return None
        req=batch.reqs[0]
        if req.prompt_len<=2051 or req.device_len!=req.prompt_len:
            return None
        return 'cold' if req.cached_len==0 else 'hot'
    def capture(label,value):
        phase=active()
        if phase is None or value is None:return
        x=value[-33:].detach().float().cpu().clone()
        if phase=='cold':
            cold[label]=x
        elif label in cold:
            ref=cold[label];delta=x-ref
            per_row=(delta.square().sum(-1).sqrt()/ref.square().sum(-1).sqrt().clamp_min(1e-10))
            rows.append(dict(label=label,shape=list(x.shape),max_error=delta.abs().max().item(),
                relative_l2=(delta.norm()/ref.norm().clamp_min(1e-10)).item(),row_relative_l2=per_row.tolist()))
        Path(output).write_text(json.dumps(dict(rows=rows,selections=selections),indent=2),encoding='utf-8')
    def wrap_forward(op,label):
        original=op.forward
        def forward(x,*args,**kwargs):
            capture(label+'.in',x)
            result=original(x,*args,**kwargs)
            capture(label+'.out',result)
            return result
        op.forward=forward
    def wrap_mix(op,label):
        original=op.mix
        def mix(x):
            capture(label+'.in',x)
            result=original(x)
            capture(label+'.out',result[0]);capture(label+'.inject',result[1])
            return result
        op.mix=mix
    for layer in engine.model.model.layers.op_list:
        label=f'layer{layer._layer_id}'
        if layer.ple is not None:wrap_forward(layer.ple,label+'.ple')
        wrap_mix(layer.attn_hyper_connection,label+'.attn_hc')
        wrap_forward(layer.linear_attn if layer._is_linear else layer.self_attn,label+'.attention')
        wrap_mix(layer.mlp_hyper_connection,label+'.mlp_hc')
        wrap_forward(layer.mlp,label+'.moe')
        wrap_forward(layer,label)
    backend=engine.attn_backend
    qsa_forward=backend.qsa_forward
    def attention(q,k,v,index,layer,batch):
        capture(f'layer{layer}.qsa.q',q.flatten(1))
        capture(f'layer{layer}.qsa.k',k)
        capture(f'layer{layer}.qsa.v',v)
        result=qsa_forward(q,k,v,index,layer,batch)
        capture(f'layer{layer}.qsa.attn_out',result.flatten(1))
        return result
    backend.qsa_forward=attention
    select=backend._select
    def selected(index,md,slot):
        capture(f'qsa{slot}.raw_index_q',index.q.flatten(1))
        capture(f'qsa{slot}.raw_index_k',index.k.flatten(1))
        result=select(index,md,slot)
        phase=active()
        if phase:
            name=f'qsa{slot}.indices'
            x=result[-33:].detach().cpu().clone()
            if phase=='cold':cold[name]=x
            else:
                ref=cold[name]
                selections.append(dict(slot=slot,equal=torch.equal(ref,x),
                    differing_rows=(ref!=x).any(-1).nonzero().flatten().tolist(),
                    set_differences=[len(set(a.tolist())^set(b.tolist())) for a,b in zip(ref,x)]))
            pages=backend._cmp_pages(slot)
            logical=pages[md.block_table[0,:43].long()].flatten(0,1)[:680].flatten(1)
            cache_label=f'qsa{slot}.cached_index_rows'
            cached=logical[:672].detach().cpu().clone()
            if phase=='cold':cold[cache_label]=cached
            else:
                ref=cold[cache_label]
                delta=cached-ref
                rows.append(dict(label=cache_label,shape=list(cached.shape),max_error=delta.abs().max().item(),
                    relative_l2=(delta.norm()/ref.norm().clamp_min(1e-10)).item()))
        return result
    backend._select=selected

