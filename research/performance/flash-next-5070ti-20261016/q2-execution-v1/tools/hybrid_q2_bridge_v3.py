"""Strata原始CPU Q2内核+独立mapped交接；尚非Speed交付实现。"""
from functools import lru_cache
from pathlib import Path
import os
import torch

ROOT=Path(__file__).parent


@lru_cache(maxsize=1)
def module():
    """冻结外部CPU来源，在独立DLL中构建AVX512实验。"""
    import hashlib
    from freetoken.kernel.utils import load_jit
    from tvm_ffi.cpp import extension
    source=ROOT/'strata-cpu-reference-v1'
    for name,sha in [('expert.cpp','b60c0a6b03f01e6e390cf801253ec73c9c7dd480069af59e9a42549e6d2755fc'),
        ('include/strata/kernels/cpu/expert.hpp','c1c4ac822465b16fd9adbfe85aeb819265f508fa2c48c3cda3837315efa7c0b1')]:
        assert hashlib.sha256((source/name).read_bytes()).hexdigest()==sha
    generate=extension._generate_ninja_build
    def windows_build(**kwargs):
        text=generate(**kwargs).replace('-Xcompiler /std:c++17 /O2','-Xcompiler=/std:c++20,/O2')
        lines=text.splitlines()
        for i,line in enumerate(lines):
            if line.startswith('cxxflags = '):
                lines[i]=line.replace('/std:c++17','/std:c++20').replace(' -std=c++20','').replace(' -O3','')
        return '\n'.join(lines)+'\n'
    extension._generate_ninja_build=windows_build
    try:
        return load_jit('hybrid_q2_proto3',
            cpp_files=[str(source/'expert.cpp'),str(ROOT/'hybrid_q2_host_v2.cpp')],
            cuda_files=[str(ROOT/'hybrid_q2_device_v1.cuh')],
            cpp_wrappers=[(name,'HybridQ2HostV1::'+name) for name in ('io_bytes','create','update','health','batches','close','stat')],
            cuda_wrappers=[(name,'HybridQ2DeviceV1::'+name) for name in ('publish','collect')],
            extra_cflags=['/O2','/arch:AVX512'],
            extra_cuda_cflags=['-D__always_inline=__forceinline__','-Xcompiler=/MD'],
            extra_include_paths=[str(source/'include'),str(ROOT),
                r'C:\workspace\build\freetoken-mats\flash-perf-materials-04\candidate-source\build-source\tools\windows-compat'],
            extra_ldflags=[f'/LIBPATH:"{Path(os.environ["CUDA_PATH"])/"lib/x64"}"','cudart.lib'],
            build_directory=r'C:\workspace\build\freetoken-mats\flash-perf-hybrid-q2-03')
    finally:
        extension._generate_ninja_build=generate


class Bridge:
    def __init__(self,pointers,threads=8):
        """pointers为CPU虚拟地址，GPU alias使用独立的device可见地址。"""
        from freetoken.kernel.pinned import MappedHostBytes,alloc_pinned_tensor
        self.native=module()
        self.owner=MappedHostBytes(alloc_pinned_tensor(self.native.io_bytes(),dtype=torch.uint8))
        self.storage=self.owner.cuda_view(torch.device('cuda'))
        self.handle=self.native.create(self.owner.owner,pointers,threads)

    def publish(self,q,scales,ids,layer):
        """调用流先完成路由/量化，发布后CPU异步计算；无需Python轮询。"""
        self.native.publish(self.storage,q,scales,ids,layer,torch.cuda.current_stream().cuda_stream)

    def collect(self,parts):
        """保持每个route独立的原始输出，再由既有GPU顺序进行路由加权归约。"""
        self.native.collect(self.storage,parts,torch.cuda.current_stream().cuda_stream)

    def check(self):
        """任何CPU错误必须显式上报，不能接纳残缺expert结果。"""
        if self.native.health(self.handle):raise RuntimeError('Hybrid CPU Q2 worker failed')

    def stats(self):
        """返回累计CPU阶段计时，GPU等待由Nsight单独量化。"""
        return dict(batches=self.native.batches(self.handle),raw=[self.native.stat(self.handle,i) for i in range(8)])

    def update(self,pointers):
        """调用者已排空GPU/CPU读者后发布真实host所有权表。"""
        self.native.update(self.handle,pointers)

    def close(self):
        """先排空图读者，再停止线程并释放mapped IO owner。"""
        if self.handle:
            torch.cuda.synchronize()
            self.native.close(self.handle)
            self.handle=0
