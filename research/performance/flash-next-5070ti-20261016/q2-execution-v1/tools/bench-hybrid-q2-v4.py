"""先验证同一Q8输入下CPU原始Q2路径、mapped交接与图重放，再考虑接入模型。"""
from pathlib import Path
import json
import time
import torch
import triton
from freetoken.kernel.pinned import alloc_pinned_tensor
from freetoken.moe.fused_q2_q8 import _routed_project,quantize_expert_activation
from freetoken.moe.fused_q2_0 import _grouped_gate_act
from hybrid_q2_bridge_v4 import Bridge,module

ROOT=Path(__file__).parent
OUT=ROOT/'hybrid-q2-components-v4.json'
if OUT.exists():raise FileExistsError(OUT)
torch.manual_seed(174)
report=dict(status='running',scope='External hybrid prototype, synthetic original Q2 experts, not model acceptance',cases=[])


def bank(n,k):
    """真实18-byte块，有限半精度scale与全部2-bit编码。"""
    w=torch.randint(0,256,(10,n,k//64,18),device='cuda',dtype=torch.uint8)
    s=(torch.rand((10,n,k//64),device='cuda')*.02+.0001).half().contiguous().view(torch.uint8).reshape(10,n,k//64,2)
    w[...,:2]=s
    packed=w.reshape(10,n,k//64*18)
    host=alloc_pinned_tensor(*packed.shape,dtype=torch.uint8)
    host.copy_(packed.cpu())
    return packed,host


def main():
    """每种window长度测试零CPU与混合路由，修改捕获输入，并明确销毁线程。"""
    module()
    gu,hostgu=bank(1280,2560);down,hostdown=bank(2560,640)
    pointers=torch.zeros((48,2,512),dtype=torch.int64,device='cpu')
    for role,host in enumerate((hostgu,hostdown)):
        for expert in range(10):pointers[0,role,expert]=host[expert].data_ptr()
    bridge=Bridge(pointers)
    try:
        for tokens in (1,2,3,4):
            ids=torch.arange(10,device='cuda',dtype=torch.int32)[None].repeat(tokens,1)
            encoded=ids.clone();encoded[:,:3]=-(encoded[:,:3]+1)
            x=torch.randn((tokens,2560),device='cuda')
            gu_values=torch.empty((tokens*10,1280),device='cuda')
            active=torch.empty((tokens*10,640),device='cuda')
            expected=torch.empty((tokens*10,2560),device='cuda')
            def reference():
                _routed_project(x,gu,gu_values,ids,10)
                _grouped_gate_act[(triton.cdiv(tokens*10*640,256),)](gu_values,active,640,tokens*10*640,256,enable_fp_fusion=False)
                _routed_project(active,down,expected,ids,1)
            result=torch.full_like(expected,-99)
            def run():
                q,scales=quantize_expert_activation(x)
                bridge.publish(q,scales,encoded,0)
                bridge.collect(result,encoded)
            reference();run();torch.cuda.synchronize();bridge.check()
            cpu=encoded.flatten()<0
            err=float((result[cpu]-expected[cpu]).abs().max())
            rel=float(torch.linalg.vector_norm(result[cpu]-expected[cpu])/torch.linalg.vector_norm(expected[cpu]))
            row=dict(tokens=tokens,max_abs=err,relative_l2=rel)
            report['cases'].append(row);OUT.write_text(json.dumps(report,indent=2)+'\n')
            assert rel<1e-5,(row,'CPU/GPU Q8 path diverged')
            assert torch.all(result[~cpu]==-99)
            graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):run()
            x.mul_(.9).add_(.01)
            encoded[:,:3]=-(torch.tensor([1,2,3],device='cuda',dtype=torch.int32)+1)
            graph.replay();reference();torch.cuda.synchronize();bridge.check()
            # Route IDs changed: compare the corresponding original expert rows.
            ids[:,:3]=torch.tensor([1,2,3],device='cuda',dtype=torch.int32)
            reference()
            torch.cuda.synchronize()
            rel=float(torch.linalg.vector_norm(result[cpu]-expected[cpu])/torch.linalg.vector_norm(expected[cpu]))
            assert rel<1e-5,('changed graph',rel)
            graph.reset()
            start=time.perf_counter()
            for _ in range(20):run()
            torch.cuda.synchronize()
            row.update(changed_graph_relative_l2=rel,us=(time.perf_counter()-start)*1e6/20)
            bridge.check()
            print(json.dumps(row),flush=True)
        # Move canonical owners to new pinned allocations, then poison obsolete rows.
        backups=[]
        for role,host in enumerate((hostgu,hostdown)):
            backup=alloc_pinned_tensor(*host.shape,dtype=torch.uint8)
            backup.copy_(host)
            backups.append(backup)
            for expert in range(10):pointers[0,role,expert]=backup[expert].data_ptr()
        bridge.update(pointers)
        hostgu.zero_();hostdown.zero_()
        run();reference();torch.cuda.synchronize();bridge.check()
        rel=float(torch.linalg.vector_norm(result[cpu]-expected[cpu])/torch.linalg.vector_norm(expected[cpu]))
        assert rel<1e-5,('owner update',rel)
        report['new_owner_old_bytes_poisoned_relative_l2']=rel
        encoded.copy_(ids);result.fill_(-99)
        run();torch.cuda.synchronize();bridge.check()
        assert torch.all(result==-99)
        report['zero_cpu_keeps_gpu_routes']=True
        report['cpu_stats']=bridge.stats()
        report['batches']=bridge.native.batches(bridge.handle)
        report['status']='passed'
    finally:
        bridge.close()
        report['closed']=bridge.handle==0
        OUT.write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
