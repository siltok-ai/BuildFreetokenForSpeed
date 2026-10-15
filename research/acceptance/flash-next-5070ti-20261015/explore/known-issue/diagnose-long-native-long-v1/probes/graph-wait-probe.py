"""小规模CUDA图同步诊断；验证NaN填充探针卡住是否只是额外图节点所致。"""
from pathlib import Path
import argparse
import hashlib
import importlib.metadata
import json
import threading
import time

import torch
from freetoken.kernel.pinned import alloc_pinned_tensor
from freetoken.kernel.row_store import probe_wait_sync, signal, wait_reset


def main():
    """延迟由独立CPU线程释放WAIT，记录graph.replay是否先于信号返回。"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    import freetoken
    report = dict(scope='synthetic_graph_sync_diagnostic_not_model_acceptance', rows=[],
        freetoken_version=importlib.metadata.version('freetoken'), torch_version=torch.__version__,
        code_origin=freetoken.__file__, probe_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    stream = torch.cuda.Stream()
    report['wait_supported'] = probe_wait_sync('auto', torch.device('cuda:0'))
    if not report['wait_supported']:
        raise RuntimeError('Stream wait is unavailable')
    flag = alloc_pinned_tensor(1, dtype=torch.int64)
    for count in (0, 64, 512, 2048):
        graph = torch.cuda.CUDAGraph()
        flag.zero_()
        with torch.cuda.stream(stream):
            result = torch.zeros(1024, device='cuda')
            torch.cuda.synchronize()
            retained = []
            with torch.cuda.graph(graph, stream=stream):
                wait_reset(stream, flag)
                for _ in range(count):
                    value = torch.empty(1024, device='cuda')
                    value.fill_(float('nan'))
                    retained.append(value)
                    result.add_(value.nan_to_num())
                result.add_(1)
            timeline = {}
            ready = threading.Event()

            def release():
                """延迟信号避免把同步返回误认为普通异步launch。"""
                ready.wait()
                time.sleep(.25)
                timeline['signal_time'] = time.perf_counter()
                signal(flag)

            thread = threading.Thread(target=release, daemon=True)
            thread.start()
            start = time.perf_counter()
            ready.set()
            graph.replay()
            returned = time.perf_counter()
            stream.synchronize()
            thread.join(2)
            row = dict(extra_fill_groups=count, replay_return_seconds=returned-start,
                signal_seconds=timeline['signal_time']-start,
                returned_before_signal=returned < timeline['signal_time'],
                result_finite=bool(torch.isfinite(result).all()),
                result_correct=bool((result==1).all()), flag_after=int(flag[0]))
            report['rows'].append(row)
            args.output.write_text(json.dumps(report, indent=2), encoding='utf-8')
            print(json.dumps(row), flush=True)
        del graph, retained, result
        torch.cuda.empty_cache()
    from freetoken.kernel.gguf_native import quantize_q8_1, mmvq
    with torch.cuda.stream(stream):
        magnitudes = torch.tensor([0., 1., 100., 1e4, 1e6, 1e8, 1e12], device='cuda')
        value = magnitudes[:, None].expand(-1, 256).contiguous()
        quantized = quantize_q8_1(value)
        scales = quantized[:, :, :4].contiguous().view(torch.float16)
        weights = torch.zeros((17, 4, 18), device='cuda', dtype=torch.uint8)
        weights[:, :, :2] = torch.full((17, 4), .125, device='cuda').half().view(torch.uint8).reshape(17, 4, 2)
        projected = mmvq(weights.flatten(), quantized, 17, 42)
        report['native_q8_extreme_finite_inputs'] = dict(magnitudes=magnitudes.cpu().tolist(),
            scales_finite=bool(torch.isfinite(scales).all()),
            projection_finite=bool(torch.isfinite(projected).all()))
    report['status'] = 'complete'
    args.output.write_text(json.dumps(report, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
