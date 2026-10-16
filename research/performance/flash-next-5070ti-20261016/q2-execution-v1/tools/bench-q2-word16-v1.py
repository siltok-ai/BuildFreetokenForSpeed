"""真实专家几何比较原BN8与16位读取，覆盖共享/部分/不共享路由和变输入图。"""
from pathlib import Path
import json
import torch
import triton
from freetoken.moe import fused_q2_q8 as module
from test_grouped_expert import make_bank
from q2_word16_variant import project

RUN=Path(__file__).parent


def main():
    """投影用紧数值界，完整两层Q8路径沿用已公开的舍入阈值边界。"""
    torch.manual_seed(197)
    gate,down=make_bank(48,1280,2560),make_bank(48,2560,640)
    original=module._routed_project
    report=dict(status='running',scope='expert_microbench_not_whole_model',cases=[])
    target=RUN/'q2-word16-components-v1.json'
    if target.exists():raise FileExistsError(target)
    for tokens in (1,2,3,4):
        for mode in ('shared','partial','disjoint'):
            ids=torch.stack([torch.arange(10,device='cuda')+i*(0 if mode=='shared' else 3 if mode=='partial' else 10)
                             for i in range(tokens)]).int()
            x=torch.randn(tokens,2560,device='cuda')
            weights=torch.softmax(torch.randn(tokens,10,device='cuda'),dim=-1)
            old_gate=torch.empty((tokens*10,1280),device='cuda')
            new_gate=torch.empty_like(old_gate)
            original(x,gate,old_gate,ids,10)
            project(x,gate,new_gate,ids,10)
            torch.testing.assert_close(new_gate,old_gate,rtol=1e-6,atol=2e-6)
            seed_x,seed_ids=x.clone(),ids.clone()
            fn=lambda:module.fused_experts_q2_q8(x,gate,down,weights,ids)
            module._routed_project=original
            reference=fn()
            paths=[]
            for name,projection in [('baseline8',original),('word16',project)]:
                x.copy_(seed_x);ids.copy_(seed_ids)
                module._routed_project=projection
                output=fn()
                torch.testing.assert_close(output,reference,rtol=1e-4,atol=2e-5)
                graph=torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):captured=fn()
                x.mul_(.9).add_(.01);ids.copy_(ids.roll(2,dims=1))
                graph.replay()
                module._routed_project=original
                expected=fn()
                module._routed_project=projection
                error=float((captured-expected).abs().max())
                torch.testing.assert_close(captured,expected,rtol=1e-4,atol=2e-5)
                x.copy_(seed_x);ids.copy_(seed_ids)
                paths.append(dict(kind=name,us=triton.testing.do_bench_cudagraph(fn,rep=100)*1000,
                    bitwise_equal=torch.equal(output,reference),changed_graph_max_abs=error))
                graph.reset()
            row=dict(tokens=tokens,mode=mode,projection_max_abs=float((old_gate-new_gate).abs().max()),paths=paths)
            report['cases'].append(row)
            target.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
            print(json.dumps(row),flush=True)
    module._routed_project=original
    report['status']='passed'
    target.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')


if __name__=='__main__':main()
