"""以真实Q2/Q8布局构造已知分块贡献，反推冻结Triton的FP32累加顺序。"""
from pathlib import Path
import itertools
import json
import numpy as np
import torch
from freetoken.moe.fused_q2_q8 import _routed_vector

ROOT=Path(__file__).parent
OUT=ROOT/'q2-reduction-order-v1.json'
if OUT.exists():raise FileExistsError(OUT)


def main():
    """输出每组已知贡献的和，浮点候选顺序在CPU用float32精确模拟。"""
    count=512;k=640;n=2560
    rng=np.random.default_rng(561)
    coefficients=(rng.normal(size=(count,20))*np.power(10.,rng.integers(-3,4,size=(count,20)))).astype(np.float32)
    q=torch.zeros((count,k),device='cuda',dtype=torch.int8)
    q[:,::32]=torch.from_numpy(np.where(coefficients>=0,1,-1).astype(np.int8)).cuda()
    scales=torch.from_numpy(np.abs(coefficients)).cuda()
    weights=torch.full((1,n,k//64,18),0xAA,device='cuda',dtype=torch.uint8)
    weights[...,:2]=torch.tensor([1.],dtype=torch.float16,device='cuda').view(torch.uint8)
    weights=weights.reshape(1,n,k//64*18)
    ids=torch.zeros(count,device='cuda',dtype=torch.int32)
    result=torch.empty((count,n),device='cuda')
    _routed_vector[(count,n//8)](q.view(torch.int32),scales,weights,ids,result,
        k,n,1,8,32,num_warps=4,enable_fp_fusion=False)
    observed=result[:,0].cpu().numpy()
    rows=[]
    for perm in itertools.permutations(range(5)):
        partial=[]
        for lane in range(4):
            accumulator=coefficients[:,lane+4*perm[0]].copy()
            for group in perm[1:]:accumulator=np.float32(accumulator+coefficients[:,lane+4*group])
            partial.append(accumulator)
        for pairing in ((0,2,1,3),(0,1,2,3),(0,3,1,2)):
            a,b,c,d=pairing
            value=np.float32(np.float32(partial[a]+partial[b])+np.float32(partial[c]+partial[d]))
            equal=int(np.count_nonzero(value==observed))
            rows.append(dict(permutation=perm,pairing=pairing,equal=equal,max_abs=float(np.max(np.abs(value-observed)))))
    rows.sort(key=lambda r:(-r['equal'],r['max_abs']))
    report=dict(status='passed',cases=count,groups=20,best=rows[:10],fully_matching=[r for r in rows if r['equal']==count])
    OUT.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
