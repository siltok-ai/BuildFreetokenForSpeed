"""Use the actual sampler to verify accepted top-k integers reach a safe device shape."""
from pathlib import Path
from types import SimpleNamespace
import argparse
import json
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--source', type=Path)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
if args.source:
    sys.path.insert(0, str(args.source / 'python'))

import torch
from freetoken.core import SamplingParams
from freetoken.engine.sample import Sampler
import freetoken

sampler = Sampler(torch.device('cuda:0'), vocab_size=16)
rows = []
for k in (0, 1, 16, 17, 2**31 - 1, 2**31, 2**80):
    params = SamplingParams(temperature=0.7, top_p=1.0, top_k=k)
    row = dict(top_k=k)
    try:
        prepared = sampler.prepare(SimpleNamespace(reqs=[SimpleNamespace(sampling_params=params)]))
        row.update(passed=True, temperatures=None if prepared.temperatures is None else prepared.temperatures.tolist(),
                   device_top_k=None if prepared.top_k is None else prepared.top_k.tolist())
    except Exception as exc:
        row.update(passed=False, error_type=type(exc).__name__, error=str(exc))
    rows.append(row)
report = dict(source=freetoken.__file__, results=rows, all_passed=all(row['passed'] for row in rows))
args.output.write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
print(json.dumps(report, indent=2), flush=True)
