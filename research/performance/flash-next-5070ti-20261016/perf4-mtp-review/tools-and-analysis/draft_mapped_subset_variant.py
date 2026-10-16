"""Draft-only Q5 row indirection, without a duplicate packed head."""
from functools import lru_cache
import hashlib
import inspect
from pathlib import Path
import textwrap
import os

import torch
from draft_subset_variant import VOCAB, VOCAB_SHA


@lru_cache(maxsize=1)
def module():
    """Build a separate native module; never replace a frozen AOT input."""
    from freetoken.kernel.utils import KERNEL_PATH, load_jit
    from tvm_ffi.cpp import extension
    generate = extension._generate_ninja_build

    def windows_build(**kwargs):
        """Apply the same local MSVC flag repair used by the accepted native producer."""
        source = generate(**kwargs).replace('-Xcompiler /std:c++17 /O2','-Xcompiler=/std:c++20,/O2')
        lines = source.splitlines()
        for i,line in enumerate(lines):
            if line.startswith('cxxflags = '):
                lines[i] = line.replace('/std:c++17','/std:c++20').replace(' -std=c++20','').replace(' -O3','')
        return '\n'.join(lines)+'\n'

    extension._generate_ninja_build = windows_build
    try:
        return load_jit('gguf_q5_subset_proto4_r1',cuda_files=[str(Path(__file__).with_name('mapped_q5_proto4_r1.cuh'))],
            cuda_wrappers=[('project','DraftMappedQ5::project')],extra_cuda_cflags=['--use_fast_math','-D__always_inline=__forceinline__'],
            extra_include_paths=[str(KERNEL_PATH/'third_party/strata_mmvq/include'),
                r'C:\workspace\build\freetoken-mats\flash-perf-materials-03\candidate-source\build-source\tools\windows-compat'],
            extra_ldflags=[f'/LIBPATH:"{Path(os.environ["CUDA_PATH"])/"lib/x64"}"','cudart.lib'],
            build_directory=r'C:\workspace\build\freetoken-mats\flash-perf-mapped-head-01')
    finally:
        extension._generate_ninja_build = generate


def mapped_project(packed, x, ids):
    """Keep the native activation quantizer and project only indexed original rows."""
    from freetoken.kernel.gguf_native import quantize_q8_1
    q8 = quantize_q8_1(x)
    out = torch.empty((1,len(ids)),device=x.device,dtype=torch.float32)
    module().project(packed,q8,ids,out,torch.cuda.current_stream(x.device).cuda_stream)
    return out


def enable():
    """Bind validated draft ids before graph capture; the main head stays complete."""
    import freetoken.models.qwen4_exp.mtp_drafter as drafter
    raw = VOCAB.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=VOCAB_SHA:raise ValueError('Draft vocabulary identity changed')
    ids = torch.frombuffer(bytearray(raw),dtype=torch.int32).long().sort().values
    if len(ids)!=106299 or ids.unique().numel()!=len(ids) or int(ids[0])<0 or int(ids[-1])>=248320:
        raise ValueError('Draft vocabulary has invalid rows')
    old_init = drafter.MtpDrafter.__init__

    def init(self,*args,**kwargs):
        """Own only the row map, and verify indexed logits against the original head."""
        old_init(self,*args,**kwargs)
        module()
        self.draft_vocab = ids.to(self.device)
        self.owned_bytes += self.draft_vocab.numel()*self.draft_vocab.element_size()
        generator = torch.Generator(device=self.device).manual_seed(731)
        exact = []
        for _ in range(3):
            x = torch.randn((1,2560),device=self.device,dtype=torch.float32,generator=generator)
            full = self.head.quant_method.apply(self.head,x)
            subset = mapped_project(self.head.packed_0,x,self.draft_vocab)
            same = torch.equal(full.index_select(1,self.draft_vocab),subset)
            exact.append(same)
            if not same:raise RuntimeError('Indexed Q5 subset changed logits')
        self.draft_subset_check = dict(rows=len(ids),packed_bytes=0,index_bytes=self.draft_vocab.numel()*8,
            exact_full_head_subset_checks=exact,target_full_rows=248320,vocab_sha256=VOCAB_SHA,mode='indexed_original_weights')

    original = textwrap.dedent(inspect.getsource(drafter.MtpDrafter.step))
    source = original.replace('logits = self.head.quant_method.apply(self.head, mixed)',
        'logits = mapped_project(self.head.packed_0, mixed, self.draft_vocab)').replace(
        'return (logits.argmax(-1).to(torch.int32), r)',
        'return (self.draft_vocab[logits.argmax(-1)].to(torch.int32), r)')
    if source==original or source.count('mapped_project(')!=1:raise RuntimeError('Unexpected draft step source')
    namespace = dict(drafter.__dict__,mapped_project=mapped_project)
    exec(compile(source,str(Path(__file__)),'exec'),namespace)
    drafter.MtpDrafter.__init__ = init
    drafter.MtpDrafter.step = namespace['step']

