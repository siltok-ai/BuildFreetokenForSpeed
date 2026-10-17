"""Exercise stop recognition and streaming with the actual local Qwen tokenizer."""
from pathlib import Path
import argparse
import json
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--source', type=Path)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
if args.source:
    sys.path.insert(0, str(args.source / 'python'))

from transformers import AutoTokenizer
from freetoken.scheduler.scheduler import _match_stop_tail
from freetoken.tokenizer.detokenize import DetokenizeManager
from freetoken.message import DetokenizeMsg
import freetoken

tokenizer = AutoTokenizer.from_pretrained(
    r'C:\workspace\models\siltok-speed\flash-next\checkpoints\flash-next-strata-q2',
    local_files_only=True,
)
cases = [
    ('unicode_pair', 'prefix ', '🦄🦄', ['🦄🦄']),
    ('unicode_single', 'prefix ', '🦄', ['🦄']),
    ('unicode_cjk', 'prefix ', '𠮷𠮷', ['𠮷𠮷']),
    ('earliest_in_token', '', 'prefix', ['fix', 'pre']),
    ('ordinary_ascii', 'prefix ', 'STOP', ['STOP']),
]
results = []
for name, prefix, output, stops in cases:
    prompt = tokenizer.encode('prompt with ' + ' '.join(stops), add_special_tokens=False)
    generated = tokenizer.encode(prefix + output, add_special_tokens=False)
    detokenizer = DetokenizeManager(tokenizer)
    history = prompt.copy()
    deltas = []
    matched = None
    for i, token in enumerate(generated):
        history.append(token)
        matched = _match_stop_tail(tokenizer, stops, history, len(prompt))
        message = DetokenizeMsg(
            uid=1, next_token=token, finished=matched is not None or i == len(generated) - 1,
            finish_reason='stop' if matched is not None else ('length' if i == len(generated)-1 else None),
            matched_stop=matched, stop_strs=stops,
        )
        deltas.extend(detokenizer.detokenize([message]))
        if matched:
            break
    text = ''.join(deltas)
    results.append(dict(name=name, stops=stops, output=output, generated_tokens=generated,
                        matched=matched, streamed=text, expected=prefix,
                        passed=matched is not None and text == prefix))
report = dict(source=freetoken.__file__, results=results,
              all_passed=all(row['passed'] for row in results))
args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
