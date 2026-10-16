"""只替换已证明逐值一致的GU分组投影，down保持冻结原路径。"""
import torch
import triton
from grouped_precomputed_variant import _build_groups
from native_grouped_q2_variant_v2 import module


def enable():
    """2..4token共享GU权重解码，按原Triton四组顺序累加。"""
    from freetoken.moe import fused_q2_q8 as target
    original=target._routed_project
    def project(value,bank,result,ids,input_topk):
        tokens,topk=ids.shape
        if not (2<=tokens<=4 and input_topk==topk):
            return original(value,bank,result,ids,input_topk)
        bt=triton.next_power_of_2(tokens)
        meta=torch.empty((ids.numel(),bt+1),device=ids.device,dtype=torch.int32)
        _build_groups[(ids.numel(),)](ids,meta,tokens,topk,bt,triton.next_power_of_2(topk),num_warps=4)
        q,scales=target.quantize_expert_activation(value)
        module().project(q.view(torch.int32),scales,bank,ids,meta,result,input_topk,4,
            torch.cuda.current_stream(value.device).cuda_stream)
    target._routed_project=project
    return original
