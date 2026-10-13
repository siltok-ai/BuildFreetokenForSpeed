"""Installed FreeToken native binding versus frozen Strata component oracle.

Uses captured actual HC inputs, all first-layer matrix formats, and final head.
No engine overrides. Numerical differences are bounded, not bitwise gates.
"""
import ctypes
import hashlib
import json
import os
from pathlib import Path

ROOT=Path(__file__).resolve().parent
os.environ['FREETOKEN_DISABLE_JIT']='1'
import capture_freetoken_reference
import torch
from safetensors import safe_open
import freetoken
from freetoken.models.qwen4_exp.gguf_source import GGUFModelSource
from freetoken.models.qwen4_exp.hc import GroupedPlusOneRMSNorm
from freetoken.kernel.triton.hc import hc_silu,hc_gate_mix
from freetoken.kernel import gguf_quant as q

def metric(x,y):
    delta=x-y
    return dict(finite=bool(torch.isfinite(x).all() and torch.isfinite(y).all()),
        max_error=delta.abs().max().item(),relative_l2=(delta.norm()/y.norm().clamp_min(1e-10)).item())

def main():
    target=ROOT/'dev4-native-component-v1.json'
    if target.exists():raise FileExistsError(target)
    if freetoken.__version__!='0.1.3+speed.strataq2dev4':raise RuntimeError('Wrong engine')
    source=GGUFModelSource([s['path'] for s in json.loads((ROOT/'complete-gguf-inventory.json').read_text())['shards']])
    def raw(t):return torch.frombuffer(bytearray(t.read_rows(0,t.nbytes//t.row_bytes)),dtype=torch.uint8).cuda()
    ft=json.loads((ROOT/'expanded-correctness-dev3-v3/freetoken.json').read_text())['runs'][-1]
    with safe_open(ft['capture'],framework='pt',device='cpu') as f:
        r=f.get_tensor('input_0').cuda();final=f.get_tensor('layer_47_0').cuda()
    def mix(r,prefix):
        norm=GroupedPlusOneRMSNorm(10240,1e-6,4)
        norm.weight=raw(source.tensors[prefix+'norm.weight']).view(torch.float32)-1
        xn=norm.forward(r)
        lo=hc_silu(q.compressed_linear(raw(source.tensors[prefix+'down.weight']),xn,320,30),4)
        hi=q.compressed_linear(raw(source.tensors[prefix+'up.weight']),lo,10240,30)
        return hc_gate_mix(xn,hi,4),lo,xn
    mixed,lo,xn=mix(r,'blk.0.hc_attn_');head,_,_=mix(final,'output_hc_')
    inputs={10240:xn,320:lo,2560:mixed}
    lib=ctypes.CDLL(str(ROOT/'strata-cuda-oracle-v2.dll'))
    lib.oracle_bf16.argtypes=[ctypes.c_void_p]*3+[ctypes.c_int]*3+[ctypes.c_void_p];lib.oracle_bf16.restype=ctypes.c_int
    lib.oracle_mmvq.argtypes=[ctypes.c_int]+[ctypes.c_void_p]*4+[ctypes.c_int]*3+[ctypes.c_void_p];lib.oracle_mmvq.restype=ctypes.c_int
    selected=[t for t in source.tensors.values() if t.name.startswith('blk.0.') and len(t.shape)==2 and t.type_id in (30,2,6,8,11,12,13,14,23,42)]
    selected.append(source.tensors['output.weight'])
    report=dict(passed=False,version=freetoken.__version__,source=freetoken.__file__,checks=[],
        scope='Actual HC-derived first-layer and head inputs; repeated mixed inputs for other widths. Frozen independent native oracle, no override. Not full cross-engine quality acceptance.',
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),relative_l2_limit=3e-4)
    for t in selected:
        k=t.shape[0];n=t.nbytes//t.row_bytes;p=raw(t)
        x=head[-3:].contiguous() if t.name=='output.weight' else inputs.get(k)
        if x is None:x=mixed.repeat(1,(k+2559)//2560)[:,:k].contiguous()
        got=q.compressed_linear(p,x,n,t.type_id);ref=torch.empty_like(got)
        for first in range(0,len(x),8):
            a=x[first:first+8].contiguous();y=ref[first:first+8]
            if t.type_id==30:
                rc=lib.oracle_bf16(a.data_ptr(),p.data_ptr(),y.data_ptr(),k,n,len(a),torch.cuda.current_stream().cuda_stream)
            else:
                scratch=torch.empty((len(a),k//32,36),device='cuda',dtype=torch.uint8)
                rc=lib.oracle_mmvq(t.type_id,p.data_ptr(),a.data_ptr(),scratch.data_ptr(),y.data_ptr(),k,n,len(a),torch.cuda.current_stream().cuda_stream)
            if rc:raise RuntimeError('Native oracle rejected '+t.name)
        checks=[]
        for size in [1,3,17,28]:
            chunks=torch.cat([q.compressed_linear(p,x[s:s+size],n,t.type_id) for s in range(0,len(x),size)])
            checks.append(dict(chunk=size,**metric(chunks,got)))
        drift=metric(got,ref)
        row=dict(name=t.name,type=t.type_id,shape=[n,k],native_error=drift,batch=checks)
        row['passed']=drift['finite'] and drift['relative_l2']<=3e-4 and all(c['finite'] and c['relative_l2']<=3e-4 for c in checks)
        report['checks'].append(row)
        target.write_text(json.dumps(report,indent=2),encoding='utf-8')
        print(json.dumps(row),flush=True)
    report['passed']=all(c['passed'] for c in report['checks'])
    target.write_text(json.dumps(report,indent=2),encoding='utf-8')
    if not report['passed']:raise RuntimeError('Component validation failed')

if __name__=='__main__':
    with torch.cuda.stream(torch.cuda.Stream()):main()
