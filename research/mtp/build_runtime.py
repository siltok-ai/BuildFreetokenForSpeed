"""Create FreeToken-layout MTP bytes from independently verified BF16 tensors."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import time
import sys

import numpy as np

HERE=Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]/'tools'))
from flash_next_mtp import finalize_runtime, canonical_manifest
Q8_NAMES={'fc_embedding.weight','fc_hidden.weight','self_attn.q_proj.weight',
    'self_attn.k_proj.weight','self_attn.v_proj.weight','self_attn.o_proj.weight',
    'self_attn.indexer.index_qk_proj.weight','mlp.shared_expert.gate_proj.weight',
    'mlp.shared_expert.up_proj.weight','mlp.shared_expert.down_proj.weight'}


def sha(path):
    with path.open('rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()


def reference_function(filename,name):
    """仅加载已冻结参考的单个量化函数，不加载Strata运行引擎或GGUF依赖。"""
    origins=json.loads((HERE/'references/origins.json').read_text(encoding='utf-8'))
    identity=next(r for r in origins['files'] if r['path']==filename)
    path=HERE/'references'/filename
    if sha(path)!=identity['sha256']: raise ValueError('Frozen quantizer source changed')
    tree=ast.parse(path.read_text(encoding='utf-8'))
    node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name)
    namespace={'np':np}
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),namespace)
    return namespace[name]


def bf16_float(raw):
    return (np.asarray(raw).astype(np.uint32)<<16).view(np.float32)


def short_name(name):
    result=name.removeprefix('mtp.')
    return result.removeprefix('layers.0.')


def decode_q2(packed,shape):
    """独立展开FreeToken原始18-byte块，用来核验FP16 scale及相邻四码布局。"""
    blocks=np.frombuffer(packed,dtype=np.uint8).reshape(-1,18)
    scales=blocks[:,:2].copy().view('<f2').astype(np.float32)
    codes=((blocks[:,2:,None]>>np.array([0,2,4,6],dtype=np.uint8))&3).astype(np.int8)-1
    return (codes.reshape(-1,64).astype(np.float32)*scales).reshape(shape)


def main():
    """全量核验固定来源后逐专家转换，不构造额外BF16完整bank或S2重排副本。"""
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    _, frozen = canonical_manifest()
    verified=json.loads((args.source/'complete-source-verification.json').read_text(encoding='utf-8'))
    lock=json.loads((args.source/'source-lock.json').read_text(encoding='utf-8'))
    if not verified['passed'] or verified['complete_tensor_count']!=31 or len(verified['tensors'])!=31:
        raise ValueError('Incomplete MTP source verification')
    if lock['revision']!='de4b8e4d43b917e7706784d8bb445c9af86a3540':
        raise ValueError('MTP revision differs from the validated Strata recipe')
    if lock != frozen['source'] or {r['name'] for r in verified['tensors']} != set(lock['tensor_sha256']):
        raise ValueError('MTP source differs from the complete frozen tensor set')
    for row in verified['tensors']:
        path=args.source/'tensors'/(row['name']+'.bin')
        if path.stat().st_size!=row['bytes'] or sha(path)!=lock['tensor_sha256'][row['name']]:
            raise ValueError('MTP source bytes changed: '+row['name'])
    args.output.mkdir(parents=True,exist_ok=False)
    q2=reference_function('mtp_pack.py','q2_0')
    q8=reference_function('mtp_rt.py','q8_0')
    started=time.perf_counter()
    report=dict(format=1,status='building', source=lock,
        source_verification_sha256=sha(args.source/'complete-source-verification.json'), tensors=[],
        layout='FreeToken raw Q2_0: FP16 scale + 16 adjacent 2-bit-code bytes, code-1; gate half then up half.',
        norm_semantics='Raw FP32 zero-centered Gemma weight; runtime applies (1+w) once.',
        draft_only=True, engine_integrated=False)
    def save():
        (args.output/'conversion-progress.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\n')
    save()
    for row in verified['tensors']:
        name=short_name(row['name'])
        path=args.source/'tensors'/(row['name']+'.bin')
        raw=np.memmap(path,dtype='<u2',mode='r',shape=tuple(row['shape']))
        target=args.output/(name+'.bin')
        sanity=[]
        if name in ('mlp.experts.gate_up_proj','mlp.experts.down_proj'):
            kind,type_id='q2_0',42
            with target.open('xb') as f:
                for expert in range(512):
                    values=bf16_float(raw[expert])
                    data=q2(values).tobytes()
                    f.write(data)
                    if expert in (0,1,255,511):
                        reconstructed=decode_q2(data,values.shape)
                        relative=float(np.sqrt(np.mean((reconstructed-values)**2)/np.mean(values**2)))
                        if not np.isfinite(relative) or relative>.6:
                            raise RuntimeError('MTP quantization sanity failed')
                        sanity.append(dict(expert=expert,relative_rms_error=relative))
                    if expert%64==0: print(name,expert,flush=True)
                    del values,data
            shape=row['shape']
        else:
            values=bf16_float(raw)
            shape=row['shape']
            if name in Q8_NAMES:
                kind,type_id='q8_0',8
                data=q8(values)
            elif len(shape)==1:
                kind,type_id='f32_raw_plus_one_norm',0
                data=values.astype('<f4').tobytes()
            else:
                kind,type_id='bf16',30
                data=np.asarray(raw).tobytes()
            target.write_bytes(data)
            del values,data
        entry=dict(name=name,shape=shape,kind=kind,type_id=type_id,path=target.name,
            bytes=target.stat().st_size,sha256=sha(target),source_sha256=row['sha256'])
        if sanity: entry['sanity']=sanity
        report['tensors'].append(entry)
        save()
        del raw
    report.update(status='packed_unvalidated_runtime',
        total_bytes=sum(t['bytes'] for t in report['tensors']),
        conversion_seconds=time.perf_counter()-started)
    save()
    finalize_runtime(args.output, report)
    print(json.dumps({k:v for k,v in report.items() if k not in {'source','tensors'}},ensure_ascii=False))


if __name__=='__main__': main()
