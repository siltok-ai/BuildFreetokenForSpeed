"""进程内混合Q2原型：CPU只读真实host冷专家，GPU命中/拷贝路径不改精度。"""
import atexit
from pathlib import Path
import weakref
import numpy as np
import torch
import triton
from hybrid_q2_bridge_v2 import Bridge
from hybrid_q2_admission_v1 import _ensure
from hybrid_q2_routed_v1 import _hybrid_routed_vector

_BRIDGES=[]
_PENDING={}


def host_pointers(cache):
    """按CPU虚拟地址生成紧凑host所有权表，GPU owner没有伪造host副本。"""
    p=cache.protected_payload
    indices=np.asarray(p.host_indices,dtype=np.int64)
    tables=[]
    for role in cache.bank_schema:
        banks=p.sources[role]
        bases=np.asarray([bank.data_ptr() for bank in banks],dtype=np.int64)
        row_bytes=p.caches[role][0].numel()*p.caches[role].element_size()
        tables.append(np.where(indices>=0,bases[:,None]+indices*row_bytes,0))
    return torch.from_numpy(np.stack(tables,axis=1).copy())


def close_all():
    """模型停用后关闭本进程创建的线程；IO owner在关闭完成前保持存活。"""
    for bridge in _BRIDGES:bridge.close()
    _BRIDGES.clear();_PENDING.clear()


def health():
    """主机已读取窗口结果时检查CPU状态，错误不能被当作有效输出。"""
    for bridge in _BRIDGES:bridge.check()


def enable(gpu_share=.55,threads=8):
    """只更换本实验进程的decode/verify，prefill及草稿全驻留专家仍走原路径。"""
    from freetoken.layers.moe import OffloadMoELayer,fused_topk
    from freetoken.moe import fused_q2_q8 as projection
    from freetoken.moe.q2_adaptive import AdaptiveQ2Owners
    original_forward=OffloadMoELayer.decode_with_shared
    original_project=projection._routed_project
    original_adapt=AdaptiveQ2Owners.adapt

    def decode(self,hidden_states,router_logits,shared_forward):
        """copy流先启动CPU工作再搬GPU冷专家，主流共享专家与它们同时执行。"""
        cache=self.offload_cache
        if not hasattr(cache,'protected_payload'):
            return original_forward(self,hidden_states,router_logits,shared_forward)
        if not hasattr(cache,'_hybrid_bridge'):
            assert not torch.cuda.is_current_stream_capturing()
            cache._hybrid_bridge=Bridge(host_pointers(cache),threads)
            _BRIDGES.append(cache._hybrid_bridge)
        bridge=cache._hybrid_bridge
        weights,ids=fused_topk(hidden_states=hidden_states,gating_output=router_logits,
            topk=self.top_k,renormalize=self.renormalize)
        main=torch.cuda.current_stream(hidden_states.device);copy=cache.decode_copy_stream
        copy.wait_stream(main)
        with torch.cuda.stream(copy):
            cache._pending_src_layer=self.layer_id;cache._pending_whole_layer=False
            n=ids.numel();p=cache.protected_payload
            if cache.collect_decode_freq:
                cache.decode_freq[self.layer_id].scatter_add_(0,ids.view(-1).long(),
                    torch.ones(n,dtype=torch.int64,device=ids.device))
            _ensure[(1,)](ids,cache.slot_for_id.view(-1),cache.id_of_slot[:p.protected_begin],
                cache.usage[:p.protected_begin],cache.step,cache.src_indices,cache.evict_slots,
                cache.num_indices,cache.lru_stats[self.layer_id],self.layer_id*cache.num_experts,
                0,p.protected_begin,p.protected_begin,n,triton.next_power_of_2(n),
                triton.next_power_of_2(p.protected_begin),cache.collect_stats,gpu_share,num_warps=4)
            q,scales=projection.quantize_expert_activation(hidden_states)
            bridge.publish(q,scales,ids,self.layer_id)
            cache.copy_missing()
        _PENDING[id(ids)]=bridge
        shared=shared_forward(hidden_states)
        main.wait_stream(copy)
        routed=self._expert_gemm(cache,hidden_states,weights,ids,views=cache.bank_views(),
            n=None,alphas=cache.alphas_for_slots(self.layer_id),is_prefill=False)
        assert id(ids) not in _PENDING
        return self._maybe_all_reduce(routed),shared

    def project(value,bank,result,ids,input_topk):
        """CPU route不读取GPU payload；CPU输出在既有路由加权归约之前合入。"""
        bridge=_PENDING.get(id(ids))
        if bridge is None:return original_project(value,bank,result,ids,input_topk)
        q,scales=projection.quantize_expert_activation(value)
        k,n=value.shape[1],result.shape[1]
        _hybrid_routed_vector[(ids.numel(),triton.cdiv(n,8))](q.view(torch.int32),scales,bank,ids,result,
            k,n,input_topk,8,triton.next_power_of_2(k//32),num_warps=4,enable_fp_fusion=False)
        if input_topk==1:
            bridge.collect(result)
            del _PENDING[id(ids)]

    def adapt(self):
        """原有交换先排空全部读者，随后同步CPU地址表以跟随真实owner变化。"""
        original_adapt(self)
        bridge=getattr(self.cache,'_hybrid_bridge',None)
        if bridge is not None:bridge.update(host_pointers(self.cache));bridge.check()

    OffloadMoELayer.decode_with_shared=decode
    projection._routed_project=project
    AdaptiveQ2Owners.adapt=adapt
    atexit.register(close_all)
