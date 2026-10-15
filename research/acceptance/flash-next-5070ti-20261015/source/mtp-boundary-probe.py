"""在新机真实worker的MTP最终提交窗口注入取消，检查同worker后续正确输出。"""
from pathlib import Path
import hashlib
import json
import runpy
import sys
import time

ROOT=Path(__file__).resolve().parent
SPEED=Path(r'C:\workspace\wt\siltok-speed\flash-accept')
sys.path.insert(0,str(SPEED))
import psutil
import freetoken
from silitok_speed.backends.base import BackendConfig
import silitok_speed.backends.freetoken_qwen36 as module
from silitok_speed.pipelines.base import PipelineCancelledError as PipelineCancellation
from silitok_speed.types import ModelSpec

OUT=ROOT/'mtp-boundary-r1024-v1.json'
MARKERS=ROOT/'mtp-boundary-r1024-v1.markers.jsonl'
if OUT.exists() or MARKERS.exists():
    raise FileExistsError('Use independent cancellation outputs')
if freetoken.__version__!='0.1.3+speed.strataq2mtp2':
    raise RuntimeError('Fixed independent MTP2 wheel required')

original_command=module._worker_command
HOOK='''
from pathlib import Path
import json
from silitok_speed.qwen36 import freetoken_worker as worker
original_factory=worker._build_streaming_llm_class
def factory(writer):
    """使用正式streaming类，仅包裹最终窗口的可观察取消边界。"""
    base=original_factory(writer)
    class FinalWindowProbe(base):
        def _forward(self, data):
            """只在真实最终窗口完成GPU拷贝后注入取消，不改模型数学或提交逻辑。"""
            result=super()._forward(data)
            batch=data.batch
            if batch.mtp_token_ids is not None:
                req=batch.reqs[0]
                if req.output_len in (2,4) and not req.can_decode:
                    result.copy_done_event.synchronize()
                    record=dict(uid=req.uid,cached_len=req.cached_len,device_len=req.device_len,
                                cpu_length=req.input_ids.numel(),bundle=batch.mtp_token_ids.tolist(),
                                requested_tokens=req.output_len)
                    with Path(MARKER_PATH).open('a',encoding='utf-8') as out:
                        out.write(json.dumps(record)+'\\n')
                    self._control.cancelled.set()
            return result
    return FinalWindowProbe
worker._build_streaming_llm_class=factory
raise SystemExit(worker.main())
'''


def observed_command():
    """只替换本测试worker的启动代码，保留正式解释器与隔离导入路径。"""
    command=original_command()
    index=command.index('-c')+1
    prefix=command[index].split('from silitok_speed.qwen36.freetoken_worker import main;')[0]
    command[index]=prefix+'MARKER_PATH='+repr(str(MARKERS))+';\n'+HOOK
    return command


def main():
    """执行四次边界取消及续写，校验真实设备状态长度和进程退出。"""
    module._worker_command=observed_command
    report=dict(status='running',scope='Actual installed MTP2 kernels and Speed source worker with a test-only final-window cancellation hook',
        forced_cancel_hook=True,performance_acceptance=False,runs=[],version=freetoken.__version__,
        source_sha256={p:hashlib.sha256((SPEED/p).read_bytes()).hexdigest() for p in
            ('silitok_speed/backends/freetoken_qwen36.py','silitok_speed/qwen36/freetoken_worker.py',
             'silitok_speed/qwen36/freetoken_mtp.py','silitok_speed/qwen36/release.json')})
    import silitok_speed
    report['identity']=runpy.run_path(str(ROOT/'run-identity.py'))['capture'](silitok_speed.__spec__.origin)
    report['probe_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    def save():
        OUT.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    save()
    backend=module.FreeTokenQwen36Backend()
    pid=None
    try:
        backend.load(ModelSpec(model_id='flash-next.q2_0_strata',weights_path=r'C:\workspace\models\siltok-speed\flash-next\checkpoints\flash-next-strata-q2',quantization='q2_0_strata'),
            BackendConfig(device='cuda:0',extra=dict(startup_timeout_seconds=600,mtp_enabled=True,
                moe_q2_resident_experts=1024,moe_cache_size=3072,moe_q2_dynamic_slots=1024,
                vram_estimate_bytes=14*2**30,host_commit_estimate_bytes=36*2**30,
                kernel_cache_dir=r'C:\workspace\wt\freetoken-mats\strata-review\artifacts\strataq2mtp2-cp312-cu130-sm120\kernel-cache',
                mtp_runtime_path=r'C:\workspace\models\siltok-speed\flash-next\checkpoints\flash-next-mtp')))
        pid=backend.worker_pid
        report.update(worker_pid=pid,mtp_identity=backend.mtp_identity(),expert_storage=backend.expert_storage())
        prompt=json.loads((ROOT/'frozen-prompts.json').read_text(encoding='utf-8'))['70-0']
        cold_reference=list(backend.generate(dict(messages=[dict(role='user',content=prompt['text'])],enable_thinking=False),
            max_new_tokens=8,temperature=0,top_p=1))[-1]
        # 后续取消及续写都复用64-token前缀，参考请求也采用同一前缀条件。
        reference=list(backend.generate(dict(messages=[dict(role='user',content=prompt['text'])],enable_thinking=False),
            max_new_tokens=8,temperature=0,top_p=1))[-1]
        assert reference['type']=='final' and len(reference['tokens'])==8
        report['uninterrupted_reference']=reference
        report['cold_reference']=cold_reference
        for cap in (2,4,2,4):
            events=[]
            started=time.perf_counter()
            try:
                events.extend(backend.generate(dict(messages=[dict(role='user',content=prompt['text'])],enable_thinking=False),
                    max_new_tokens=cap,temperature=0,top_p=1))
            except PipelineCancellation:
                pass
            else:
                raise AssertionError('Final-window cancellation did not reach the client')
            assert not any(e['type']=='final' for e in events)
            assert backend.worker_pid==pid
            report['runs'].append(dict(label='cancel-'+str(cap),events=events,seconds=time.perf_counter()-started,
                same_worker=True,delivered_tokens=sum(e['type']=='token' for e in events)))
            continued=list(backend.generate(dict(messages=[dict(role='user',content=prompt['text'])],enable_thinking=False),
                max_new_tokens=8,temperature=0,top_p=1))
            assert continued[-1]['type']=='final' and len(continued[-1]['tokens'])==8 and backend.worker_pid==pid
            assert continued[-1]['tokens']==reference['tokens']
            report['runs'].append(dict(label='continued-8',final=continued[-1],same_worker=True))
            save()
        markers=[json.loads(line) for line in MARKERS.read_text(encoding='utf-8').splitlines()]
        assert len(markers)==4
        for marker in markers:
            assert marker['device_len']==70+marker['requested_tokens']
            assert marker['cached_len']==marker['cpu_length']-1+len(marker['bundle'])
        report.update(status='passed',markers=markers)
        report['worker_stderr_tail']=backend._transport.stderr_tail()
    finally:
        backend.unload()
        report['worker_gone']=pid is not None and not psutil.pid_exists(pid)
        save()
    if not report['worker_gone']:
        raise RuntimeError('Actual cancellation worker was not reaped')
    print(json.dumps(dict(status=report['status'],forced_boundary_cancels=4,same_worker_reuse=4,worker_gone=True)))


if __name__=='__main__':
    main()
