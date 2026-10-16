"""验证真实Q2投影形状、分组复用、图重放输入更新，并测完整两投影开销。"""
from pathlib import Path
import hashlib
import json
import torch
import triton
from freetoken.moe.fused_q2_q8 import _routed_project,quantize_expert_activation
from native_grouped_q2_variant_v1 import make_projector,module

ROOT=Path(__file__).parent
OUT=ROOT/'native-grouped-q2-components-v2.json'
if OUT.exists():raise FileExistsError(OUT)
torch.manual_seed(419)
report=dict(status='running',scope='Synthetic full original Q2 bytes, exact FP32 scales, not model throughput',cases=[],
    source_sha256=hashlib.sha256((ROOT/'native_grouped_q2_proto1.cuh').read_bytes()).hexdigest())


def bank(n,k):
    """构造所有二位码及有限半精度scale，保留18字节块布局。"""
    value=torch.randint(0,256,(48,n,k//64,18),device='cuda',dtype=torch.uint8)
    scales=(torch.rand((48,n,k//64),device='cuda')*.02+.0001).half().contiguous().view(torch.uint8).reshape(48,n,k//64,2)
    value[...,:2]=scales
    return value.reshape(48,n,k//64*18)


def run():
    """覆盖共享/不共享专家及2/3/4token，改变图内路由后重新对照。"""
    module()
    gu=bank(1280,2560);down=bank(2560,640)
    for tokens in (2,3,4):
        for pattern in ('shared','mixed','unique'):
            ids=torch.arange(tokens*10,device='cuda',dtype=torch.int32).reshape(tokens,10)
            if pattern=='shared':ids[:]=ids[0]
            if pattern=='mixed':ids[:,::2]=ids[0,::2]
            x=torch.randn((tokens,2560),device='cuda')
            active=torch.randn((tokens*10,640),device='cuda')
            baseline_gu=torch.empty((tokens*10,1280),device='cuda')
            baseline_down=torch.empty((tokens*10,2560),device='cuda')
            gu_out=torch.empty_like(baseline_gu);down_out=torch.empty_like(baseline_down)
            def old():
                _routed_project(x,gu,baseline_gu,ids,10)
                _routed_project(active,down,baseline_down,ids,1)
            old()
            old_us=triton.testing.do_bench_cudagraph(old,rep=40)*1000
            choices=[]
            for lanes in (16,32):
                projector=make_projector(_routed_project,lanes)
                def new():
                    projector(x,gu,gu_out,ids,10)
                    projector(active,down,down_out,ids,1)
                new()
                torch.testing.assert_close(gu_out,baseline_gu,atol=2e-5,rtol=2e-5)
                torch.testing.assert_close(down_out,baseline_down,atol=2e-5,rtol=2e-5)
                error=max(float((gu_out-baseline_gu).abs().max()),float((down_out-baseline_down).abs().max()))
                graph=torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):new()
                x.mul_(.93).add_(.01);active.mul_(.91).add_(.02)
                ids.add_(1).remainder_(48)
                graph.replay();old()
                torch.testing.assert_close(gu_out,baseline_gu,atol=2e-5,rtol=2e-5)
                torch.testing.assert_close(down_out,baseline_down,atol=2e-5,rtol=2e-5)
                graph.reset()
                us=triton.testing.do_bench_cudagraph(new,rep=40)*1000
                choices.append(dict(lanes=lanes,us=us,baseline_us=old_us,max_abs_error=error,changed_graph=True))
            row=dict(tokens=tokens,pattern=pattern,choices=choices)
            report['cases'].append(row)
            OUT.write_text(json.dumps(report,indent=2)+'\n')
            print(json.dumps(row),flush=True)
    report['status']='passed'
    OUT.write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':run()
