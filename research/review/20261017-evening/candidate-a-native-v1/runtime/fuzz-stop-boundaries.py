"""Compare real incremental decoding with an output-prefix oracle over reproducible Unicode cases."""
from pathlib import Path
import hashlib
import json
import random

from transformers import AutoTokenizer
from freetoken.message import DetokenizeMsg
from freetoken.scheduler.scheduler import _match_stop_tail
from freetoken.tokenizer.detokenize import DetokenizeManager
import freetoken

RUN = Path(__file__).parent
tokenizer = AutoTokenizer.from_pretrained(
    r'C:\workspace\models\siltok-speed\flash-next\checkpoints\flash-next-strata-q2', local_files_only=True)
rng = random.Random(20261017)
alphabet = list('abc XYZ_你𠮷🦄é𝄞\n')
failed = []
cases = 2500
steps = 0
for number in range(cases):
    text = ''.join(rng.choice(alphabet) for _ in range(rng.randrange(2, 48)))
    stops = []
    for _ in range(rng.randrange(1, 5)):
        start = rng.randrange(len(text))
        stops.append(text[start:start+rng.randrange(1, min(9, len(text)-start)+1)])
    rng.shuffle(stops)
    prompt = tokenizer.encode('prompt ' + ' '.join(stops), add_special_tokens=False)
    generated = tokenizer.encode(text, add_special_tokens=False)
    manager = DetokenizeManager(tokenizer, frozenset())
    deltas = []
    for index, token in enumerate(generated):
        prefix = tokenizer.decode(generated[:index+1])
        positions = [(prefix.find(stop), order, stop) for order, stop in enumerate(stops) if stop in prefix]
        expected = min(positions)[2] if positions else None
        observed = _match_stop_tail(tokenizer, stops, prompt+generated[:index+1], len(prompt))
        steps += 1
        if observed != expected:
            failed.append(dict(case=number, text=text, stops=stops, prefix=prefix,
                               expected=expected, observed=observed, tokens=generated, index=index))
            break
        finished = observed is not None or index == len(generated)-1
        deltas.extend(manager.detokenize([DetokenizeMsg(uid=number, next_token=token,
            finished=finished, finish_reason='stop' if observed else ('length' if finished else None),
            matched_stop=observed, stop_strs=stops)]))
        if finished:
            expected_text = prefix[:min(positions)[0]] if positions else prefix
            if ''.join(deltas) != expected_text:
                failed.append(dict(case=number, text=text, stops=stops, prefix=prefix,
                                   expected_text=expected_text, observed_text=''.join(deltas), tokens=generated))
            break
report = dict(status='passed' if not failed else 'failed', seed=20261017, cases=cases, steps=steps,
              source=freetoken.__file__, failures=failed, probe_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
(RUN/'stop-unicode-fuzz-v1.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
print(json.dumps(dict(status=report['status'], cases=cases, steps=steps, failures=len(failed))), flush=True)
raise SystemExit(bool(failed))
