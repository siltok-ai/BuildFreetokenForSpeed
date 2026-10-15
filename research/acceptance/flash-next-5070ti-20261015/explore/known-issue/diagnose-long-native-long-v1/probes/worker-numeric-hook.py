"""仅在独立诊断worker中记录logits数值；不改动采样结果或已安装文件。"""
import atexit
import contextlib
import json
import os
import sys
from pathlib import Path

with contextlib.redirect_stdout(sys.stderr):
    import torch
    from freetoken.engine.engine import Engine
    from freetoken.engine.sample import Sampler

original_sample = Sampler.sample
original_forward = Engine.forward_batch
records = []
metadata = []


def observed_forward(self, batch, args):
    """只读取已有CPU请求元数据，保持正式forward及CUDA图调度。"""
    self.sampler._diag_meta = dict(decode=batch.is_decode,
        requests=[dict(uid=req.uid, cached_len=req.cached_len, device_len=req.device_len,
                       requested_output=req.output_len) for req in batch.reqs])
    return original_forward(self, batch, args)


def observed_sample(self, logits, args):
    """在正式采样后异步记录GPU归约值，不在每步增加CPU同步。"""
    tokens = original_sample(self, logits, args)
    if len(records) >= 65536:
        raise RuntimeError('Diagnostic step capacity exceeded')
    summary = torch.stack((torch.isnan(logits).sum().float(), torch.isinf(logits).sum().float(),
                           logits[0, 0].float(), logits.nan_to_num().abs().amax().float(),
                           tokens[0].float()))
    records.append(summary)
    metadata.append(getattr(self, '_diag_meta', {}))
    return tokens


def save():
    """worker正常退出时统一回读，保留全部采样步和首次非有限值的位置。"""
    values = torch.stack(records).cpu().tolist() if records else []
    rows = [dict(**meta, nan_count=int(value[0]), inf_count=int(value[1]),
                 zero_logit=value[2], max_finite_abs=value[3], token=int(value[4]))
            for meta, value in zip(metadata, values)]
    output = Path(OUTPUT) / ('numeric-' + str(os.getpid()) + '.json')
    output.write_text(json.dumps(dict(scope='diagnostic_async_reductions_added',
        steps=rows, first_nonfinite=next((i for i, row in enumerate(rows)
            if row['nan_count'] or row['inf_count']), None)), ensure_ascii=False, indent=2), encoding='utf-8')


Sampler.sample = observed_sample
Engine.forward_batch = observed_forward
atexit.register(save)
