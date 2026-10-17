"""Exercise request-boundary fixes with the full real checkpoint, without changing its weights."""
from pathlib import Path
import argparse
import json
import time

import torch
from freetoken.core import SamplingParams
from freetoken.llm import LLM
from freetoken.message import DetokenizeMsg
from freetoken.mm.config import ENCODER_KINDS, MultimodalConfig
from freetoken.tokenizer.detokenize import DetokenizeManager
import freetoken

MODEL = Path(r'C:\workspace\models\siltok-speed\flash-next\checkpoints\flash-next-strata-q2')


class BoundaryLLM(LLM):
    """Keep actual scheduler terminals and detokenized deltas for semantic assertions."""
    def offline_send_result(self, messages):
        for message in messages:
            if isinstance(message, DetokenizeMsg):
                self.deliveries.append(dict(uid=message.uid, token=int(message.next_token),
                    finished=message.finished, reason=message.finish_reason, matched_stop=message.matched_stop))
                self.text.setdefault(message.uid, []).extend(self.detokenizer.detokenize([message]))
        return super().offline_send_result(messages)

    def generate(self, *args, **kwargs):
        self.deliveries, self.text = [], {}
        self.detokenizer = DetokenizeManager(self.tokenizer, frozenset(self.eos_token_ids))
        return super().generate(*args, **kwargs)


def main():
    """Verify full-window clipping, a huge valid top-k, and real Unicode/overlapping stop delivery."""
    p = argparse.ArgumentParser()
    p.add_argument('--mtp', action='store_true')
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    report = dict(status='loading', python_source=freetoken.__file__, version=freetoken.__version__,
                  mtp=args.mtp, runs=[])
    def save():
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    save()
    resident = 4992 if args.mtp else 5760
    started = time.perf_counter()
    llm = BoundaryLLM(str(MODEL), dtype=torch.bfloat16, max_running_req=1,
        attention_backend='qsa_sparse', moe_strategy='offload', ple_backend='disk', expert_load='serial',
        mm=MultimodalConfig(disabled_encoders=frozenset(ENCODER_KINDS)),
        moe_cache_size=resident+2048, moe_q2_resident_experts=resident, moe_q2_dynamic_slots=1024,
        ple_row_cache_rows=1048576, moe_cache_auto=False, num_token_override=8192,
        cuda_graph_max_bs=1, memory_ratio=.9, max_seq_len_override=262144, max_extend_tokens=1024,
        cache_type='radix', mtp_runtime_path=str(MODEL.parent/'flash-next-mtp') if args.mtp else None,
        mtp_compact_state=args.mtp, mtp_draft_policy='paired_cjk' if args.mtp else 'full',
        linear_state_cache_slots=1, moe_q2_adapt_every=32, moe_q2_adapt_decay=.7,
        gguf_embedding_host=True, moe_q2_prefill_workspace=True, fixed_q2_rebuild_reserve_mib=0)
    report['load_s'] = time.perf_counter()-started
    def encode(text):
        return llm.tokenizer.apply_chat_template([{'role':'user','content':text}], add_generation_prompt=True,
            enable_thinking=False, tokenize=True, return_dict=False)
    try:
        target = llm.engine.max_seq_len-2
        count = target
        for _ in range(4):
            long_ids = encode(' x'*count)
            if len(long_ids) == target:
                break
            count += target-len(long_ids)
        if len(long_ids) != target:
            raise RuntimeError('Could not construct exact full-window prompt')
        shared = SamplingParams(temperature=0, top_p=1, max_tokens=16, ignore_eos=True)
        short_ids = encode('What is 17 times 31? Reply with the number only.')
        outputs = llm.generate([long_ids, short_ids], shared)
        counts = [len(value['token_ids']) for value in outputs]
        report['runs'].append(dict(name='shared-budget', input_tokens=[len(long_ids),len(short_ids)],
            actual_output_tokens=counts, caller_max_tokens=shared.max_tokens, outputs=outputs,
            passed=counts == [2,16] and shared.max_tokens == 16))
        save()
        sampled = llm.generate([short_ids], SamplingParams(temperature=.7, top_p=1, top_k=2**80, max_tokens=8))
        report['runs'].append(dict(name='large-top-k', top_k=2**80, outputs=sampled,
                                  passed=0 < len(sampled[0]['token_ids']) <= 8))
        save()
        for name, prompt, stops in [
            ('unicode', '只原样输出这两个符号，不要解释，不要加标点：🦄🦄', ['🦄🦄']),
            ('overlapping', 'Reply with exactly this single word and nothing else: prefix', ['fix','pre']),
        ]:
            ids = encode(prompt)
            reference = llm.generate([ids], SamplingParams(temperature=0, max_tokens=64))[0]
            positions = [(reference['text'].find(stop), stop) for stop in stops if stop in reference['text']]
            if not positions:
                raise RuntimeError('Reference did not generate the requested stop: '+repr(reference))
            cut, earliest = min(positions)
            stopped = llm.generate([ids], SamplingParams(temperature=0, max_tokens=64, stop_strs=stops))[0]
            actual = ''.join(llm.text[0])
            terminal = llm.deliveries[-1]
            report['runs'].append(dict(name=name, stops=stops, reference=reference, stopped=stopped,
                streamed=actual, expected=reference['text'][:cut], terminal=terminal,
                passed=actual == reference['text'][:cut] and stopped['text'] == actual and terminal['matched_stop'] == earliest
                       and terminal['finished'] and terminal['reason'] == 'stop'))
            save()
        final = llm.generate([short_ids], SamplingParams(temperature=0,max_tokens=64))[0]
        report['arithmetic_after'] = final
        if final['text'].strip() != '527' or not all(row['passed'] for row in report['runs']):
            raise RuntimeError('Boundary regression failed')
        llm.cache_manager.check_integrity()
        report['status'] = 'passed'
    except BaseException as exc:
        report.update(status='failed', error=repr(exc))
        raise
    finally:
        llm.shutdown()
        report['shutdown_complete'] = True
        save()


if __name__ == '__main__':
    main()
