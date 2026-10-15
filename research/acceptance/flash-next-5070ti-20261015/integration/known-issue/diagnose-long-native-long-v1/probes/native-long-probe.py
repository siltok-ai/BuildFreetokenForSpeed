"""绕过Speed包装层，在原生FreeToken中重放同一HTTP输入顺序和长逻辑题。"""
from pathlib import Path
import argparse
import json
import time

import torch
from freetoken.core import SamplingParams
from freetoken.llm.llm import LLM
from freetoken.mm.config import ENCODER_KINDS, MultimodalConfig


def main():
    """使用完整相同权重和专家池，保存实际输入/输出IDs及自然结束情况。"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--baseline', type=Path, required=True)
    args = parser.parse_args()
    baseline = json.loads(args.baseline.read_text(encoding='utf-8'))
    report = dict(scope='native_freetoken_long_logic_diagnostic', status='loading', runs=[])
    args.output.write_text(json.dumps(report), encoding='utf-8')
    llm = LLM(r'C:\workspace\models\siltok-speed\flash-next\checkpoints\flash-next-strata-q2',
        dtype=torch.bfloat16, max_running_req=1, attention_backend='qsa_sparse',
        moe_strategy='offload', ple_backend='disk', expert_load='serial',
        mm=MultimodalConfig(disabled_encoders=frozenset(ENCODER_KINDS)),
        moe_cache_size=5632, moe_q2_resident_experts=3584, moe_q2_dynamic_slots=1024,
        moe_cache_auto=False, kv_reserve_tokens=8192, num_token_override=8192,
        cuda_graph_max_bs=1, memory_ratio=.9, max_seq_len_override=262144,
        max_extend_tokens=4096, cache_type='radix', decode_log_interval=1000)
    cases = [dict(id='initial_http_prompt', prompt='Give one concise sentence about local inference.', max_tokens=32)]
    cases += [dict(case, max_tokens=2048 if case['id']=='constraint_logic' else 32)
              for case in baseline['quality_probes']]
    logic = cases[-1]
    cases += [dict(logic, id='constraint_logic_repeat')]
    cases += [dict(logic, id='constraint_logic_prefix23',
        prompt='测试编号 23。请独立回答以下问题。\n' + logic['prompt'], actual=None)]
    try:
        for case in cases:
            ids = llm.tokenizer.apply_chat_template([dict(role='user', content=case['prompt'])],
                add_generation_prompt=True, tokenize=True, enable_thinking=False, return_dict=False)
            started = time.perf_counter()
            output = llm.generate([ids], SamplingParams(temperature=0, top_p=1, max_tokens=case['max_tokens']))[0]
            row = dict(id=case['id'], input_ids=ids, output=output,
                wall_seconds=time.perf_counter()-started,
                matches_source_text=(output['text'].strip()==case['actual'].strip()
                    if case.get('actual') is not None else None),
                repeated_token0='!'*64 in output['text'])
            report['runs'].append(row)
            args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
            print(json.dumps(dict(id=row['id'], output_tokens=len(output['token_ids']),
                matches_source_text=row['matches_source_text'], repeated_token0=row['repeated_token0'])), flush=True)
        report['status'] = 'passed' if all(row['matches_source_text'] and not row['repeated_token0']
            for row in report['runs'] if row['id'].startswith('constraint_logic')
            and row['matches_source_text'] is not None) else 'different_or_failed'
    finally:
        llm.shutdown()
        report['shutdown_complete'] = True
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
