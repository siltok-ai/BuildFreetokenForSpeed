"""按真实CUDA storage归属核对显存，不调用模型属性或创建新GPU张量。"""
import torch

def inventory(engine):
    """只遍历已有字段，去重视图与模块别名；未归属部分保留为图池/其他分配。"""
    seen=set();storages={}
    def visit(obj,path):
        if id(obj) in seen:return
        seen.add(id(obj))
        if isinstance(obj,torch.Tensor):
            if obj.is_cuda:
                storage=obj.untyped_storage();key=storage.data_ptr()
                storages.setdefault(key,dict(path=path,bytes=storage.nbytes(),shape=list(obj.shape),dtype=str(obj.dtype)))
            return
        if isinstance(obj,dict):
            for k,v in obj.items():visit(v,path+'.'+str(k))
        elif isinstance(obj,(list,tuple)):
            for k,v in enumerate(obj):visit(v,path+'.'+str(k))
        elif type(obj).__module__.startswith(('freetoken','torch.nn','torch.cuda')) and hasattr(obj,'__dict__'):
            for k,v in vars(obj).items():visit(v,path+'.'+k)
    for attr in ['moe_offload_cache','linear_state_pool','kv_cache','model','graph_runner','mtp_runtime','ctx']:
        visit(getattr(engine,attr,None),attr)
    by_root={}
    for row in storages.values():
        root=row['path'].split('.')[0];by_root[root]=by_root.get(root,0)+row['bytes']
    allocated=torch.cuda.memory_allocated();reserved=torch.cuda.memory_reserved()
    return dict(allocated=allocated,reserved=reserved,storage_bytes=sum(by_root.values()),by_root=by_root,
                unattributed_allocated=allocated-sum(by_root.values()),largest=sorted(storages.values(),key=lambda r:-r['bytes'])[:100])
