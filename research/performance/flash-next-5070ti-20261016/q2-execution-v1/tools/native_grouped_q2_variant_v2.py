"""独立CUDA分组原型；只处理2..4token，不改冻结材料和激活量化。"""
from functools import lru_cache
from pathlib import Path
import os
import torch
import triton
from grouped_precomputed_variant import _build_groups


@lru_cache(maxsize=1)
def module():
    """构建独立模块；沿用已验证的Windows编译参数修复。"""
    from freetoken.kernel.utils import KERNEL_PATH,load_jit
    from tvm_ffi.cpp import extension
    generate=extension._generate_ninja_build
    def windows_build(**kwargs):
        source=generate(**kwargs).replace('-Xcompiler /std:c++17 /O2','-Xcompiler=/std:c++20,/O2')
        lines=source.splitlines()
        for i,line in enumerate(lines):
            if line.startswith('cxxflags = '):
                lines[i]=line.replace('/std:c++17','/std:c++20').replace(' -std=c++20','').replace(' -O3','')
        return '\n'.join(lines)+'\n'
    extension._generate_ninja_build=windows_build
    try:
        return load_jit('native_grouped_q2_proto2',cuda_files=[str(Path(__file__).with_name('native_grouped_q2_proto2.cuh'))],
            cuda_wrappers=[('project','NativeGroupedQ2Proto2::project')],
            extra_cuda_cflags=['--fmad=false','-D__always_inline=__forceinline__'],
            extra_include_paths=[r'C:\workspace\build\freetoken-mats\flash-perf-materials-04\candidate-source\build-source\tools\windows-compat'],
            extra_ldflags=[f'/LIBPATH:"{Path(os.environ["CUDA_PATH"])/"lib/x64"}"','cudart.lib'],
            build_directory=r'C:\workspace\build\freetoken-mats\flash-perf-native-grouped-q2-02')
    finally:
        extension._generate_ninja_build=generate


def make_projector(original,lanes=4):
    """GU和down共享图内路由表，所有权限于同一次专家调用。"""
    pending={}
    def project(value,bank,result,ids,input_topk):
        tokens,topk=ids.shape
        if not 2<=tokens<=4:return original(value,bank,result,ids,input_topk)
        bt=triton.next_power_of_2(tokens)
        if input_topk==topk:
            meta=torch.empty((ids.numel(),bt+1),device=ids.device,dtype=torch.int32)
            _build_groups[(ids.numel(),)](ids,meta,tokens,topk,bt,triton.next_power_of_2(topk),num_warps=4)
            pending.update(ids=ids,meta=meta)
        else:
            assert input_topk==1 and pending['ids'] is ids
            meta=pending.pop('meta');pending.pop('ids')
        from freetoken.moe.fused_q2_q8 import quantize_expert_activation
        q,scales=quantize_expert_activation(value)
        module().project(q.view(torch.int32),scales,bank,ids,meta,result,input_topk,lanes,
            torch.cuda.current_stream(value.device).cuda_stream)
    return project


def enable(lanes=4):
    """只修改当前实验进程；正式方案还需版本化重建与完整回归。"""
    from freetoken.moe import fused_q2_q8 as target
    original=target._routed_project
    target._routed_project=make_projector(original,lanes)
    return original
