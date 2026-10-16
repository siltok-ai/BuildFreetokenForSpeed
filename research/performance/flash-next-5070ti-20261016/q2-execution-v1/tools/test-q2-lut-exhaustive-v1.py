"""穷举全部16-bit编码，检查byte-permute的有符号展开和零点。"""
from pathlib import Path
import json
import torch
import triton
import triton.language as tl
from q2_lut_variant_v1 import signed_codes


@triton.jit
def unpack(C,Y,B:tl.constexpr):
    index=tl.program_id(0)*B+tl.arange(0,B)
    code=tl.load(C+index).to(tl.uint32)
    lo=signed_codes(code,tl.full((B,),False,tl.int1))
    hi=signed_codes(code,tl.full((B,),True,tl.int1))
    tl.store(Y+index*2,lo);tl.store(Y+index*2+1,hi)


def main():
    """独立定义八个2-bit量化值为code-1，与实际汇编helper逐字节对照。"""
    out=Path(__file__).with_name('q2-lut-exhaustive-v1.json')
    if out.exists():raise FileExistsError(out)
    values=torch.arange(65536,device='cuda',dtype=torch.int32)
    encoded=values.to(torch.int16)
    packed=torch.empty((65536,2),device='cuda',dtype=torch.int32)
    unpack[(256,)](encoded,packed,256,num_warps=4)
    reference=(((values[:,None]>>(torch.arange(8,device='cuda')*2)[None,:])&3)-1).to(torch.int8)
    torch.testing.assert_close(packed.view(torch.int8),reference,atol=0,rtol=0)
    result=dict(status='passed',inputs=65536,decoded_values=65536*8,exact=True,
        scope='Original Q2 code domain, no model or precision change')
    out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))


if __name__=='__main__':main()
