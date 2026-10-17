"""复核旧35B实际输出，执行生成的纯阶乘函数并记录历史文本对照。"""
from pathlib import Path
import argparse
import ast
import json
import re


def main():
    """按明确输出契约判断六类用例；中文开放问答仅检查语言与完整记录。"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    current = json.loads(args.report.read_text(encoding='utf-8'))
    baseline = json.loads(args.baseline.read_text(encoding='utf-8'))
    rows = {r['id']: r['result'] for r in current['quality_cases']}
    old = {r['id']: r['result'] for r in baseline['quality_cases']}
    code = rows['python']['text'].strip().removeprefix('```python').removesuffix('```').strip()
    tree = ast.parse(code)
    # 实际候选只需整数循环；禁止执行导入、属性访问及额外调用。
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom, ast.Attribute)):
            raise ValueError('Generated factorial contains an unexpected operation')
        if isinstance(node, ast.Call) and not (isinstance(node.func, ast.Name) and node.func.id == 'range'):
            raise ValueError('Generated factorial contains an unexpected call')
    namespace = {'__builtins__': {'range': range}}
    exec(compile(tree, '<actual-model-factorial>', 'exec'), namespace)
    expected = {0: 1, 1: 1, 5: 120, 10: 3628800}
    outputs = {n: namespace['factorial'](n) for n in expected}
    structured = rows['long_structured_no_thinking']['text']
    checks = dict(
        arithmetic=rows['arithmetic_no_thinking']['text'].strip() == '703',
        thinking=bool(rows['reasoning_with_thinking']['reasoning_content']) and
                 bool(re.search(r'FINAL:\s*76\s*$', rows['reasoning_with_thinking']['text'])),
        structured=all(name in structured for name in
                       ('Model Identity', 'Resource Admission', 'Streaming', 'Cancellation', 'Shutdown')),
        chinese_language=len(re.findall(r'[\u4e00-\u9fff]', rows['chinese']['text'])) >= 50,
        json=json.loads(rows['json']['text']) == dict(name='张三', city='上海', age=28),
        factorial=outputs == expected,
        cancel_reused=current['cancel']['worker_reused'],
        worker_exited=current['worker_pid_gone'])
    result = dict(status='passed' if all(checks.values()) else 'failed', checks=checks,
                  factorial_outputs=outputs,
                  matches_historical_text={name: row['text'] == old[name]['text']
                                          for name, row in rows.items()},
                  historical_token_ids_available=False,
                  finish_reasons={name: row['finish_reason'] for name, row in rows.items()},
                  scope='Chinese open answer is a language smoke, not factual grading; full truncated output retained.')
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False))
    if result['status'] != 'passed':
        raise RuntimeError('Old-model regression output check failed')


if __name__ == '__main__':
    main()
