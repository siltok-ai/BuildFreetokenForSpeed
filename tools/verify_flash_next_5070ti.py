"""离线复核5070 Ti源码阶段原始结果；不代表整包或干净机器验收。"""
from pathlib import Path, PurePosixPath
import gzip
import hashlib
import json
import math
import statistics

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT/'research/acceptance/flash-next-5070ti-20261015/source'


def require(condition, message):
    """验收证据不满足约束时立即报告，不把缺项变成跳过。"""
    if not condition:
        raise ValueError(message)


def read(name):
    """读取本轮固定证据目录里的JSON。"""
    return json.loads((EVIDENCE/name).read_text(encoding='utf-8'))


def verify():
    """核对封存字节，并从原始SSE时间、输出与质量基线重新计算结论。"""
    index = read('index.json')
    require(index['scope'] == '5070ti_source_acceptance', 'Unexpected evidence scope')
    require(index['full_installer'] is False and index['integration_branch'] is False,
            'Source evidence cannot certify later delivery phases')
    for name, expected in index['files'].items():
        path = PurePosixPath(name)
        require(not path.is_absolute() and '..' not in path.parts and '\\' not in name,
                f'Unsafe evidence path: {name}')
        target = (EVIDENCE/name).resolve()
        require(target.is_relative_to(EVIDENCE.resolve()), f'Evidence path escapes archive: {name}')
        content = target.read_bytes()
        require(len(content) == expected['bytes'] and hashlib.sha256(content).hexdigest() == expected['sha256'],
                f'Evidence bytes changed: {name}')

    performance = {}
    for name in ('http-off-r3584-perf-v1', 'http-on-r2304-perf-v1', 'http-off-r2304-perf-v1'):
        report = read(name+'.json')
        require(report['status'] == 'measured' and report['worker_pids_gone'], f'Incomplete run: {name}')
        require(len(report['runs']) == 6, f'Expected two contexts and three repetitions: {name}')
        groups = {}
        for row in report['runs']:
            count = row['usage']['completion_tokens']
            times = row['arrival_offsets']
            require(count == row['output_tokens'] == row['content_chunk_count'] == len(times) == 512,
                    f'Not 512 actual output tokens: {name}/{row["label"]}')
            require(row['finish_reason'] == 'length' and row['done_events'] == 1,
                    f'Incomplete SSE: {name}/{row["label"]}')
            require(all(b >= a for a, b in zip(times, times[1:])), 'Arrival times are not monotonic')
            measured = (count-1)/(times[-1]-times[0])
            require(math.isclose(measured, row['decode_tps'], rel_tol=1e-7), 'Decode rate does not match arrivals')
            context = row['label'].rsplit('-', 1)[0]
            require(row['usage']['prompt_tokens'] == {'70-0':70, 'diverse-4096-0':4096}[context],
                    'Input length changed')
            groups.setdefault(context, []).append(measured)
        for context, rates in groups.items():
            require(len(rates) == 3, 'Missing repetition')
            require(math.isclose(statistics.median(rates), report['contexts'][context]['median_tps'], rel_tol=1e-7),
                    'Median does not match raw runs')
        resource = read(name+'.resource.json')
        require(resource['returncode'] == 0 and resource['stop_reason'] is None and not resource['remaining_descendants'],
                f'Run did not finish and clean up: {name}')
        performance[name] = report

    enabled = performance['http-on-r2304-perf-v1']
    matched = performance['http-off-r2304-perf-v1']
    require(enabled['args']['protected_experts'] == matched['args']['protected_experts'] == '2304',
            'MTP comparison capacities differ')
    for on, off in zip(enabled['runs'], matched['runs']):
        require(all(on[key] == off[key] for key in ('label', 'text', 'reasoning_content', 'usage', 'finish_reason')),
                'Matched-capacity output differs')

    baseline = json.loads(gzip.decompress((EVIDENCE/'historical-quality.json.gz').read_bytes()))
    expected = {row['name']:row for row in baseline['runs']}
    for name in ('backend-off-r2816-quality-v1', 'backend-on-r1536-quality-v1',
                 'backend-off-r3584-quality-v1', 'backend-on-r2304-quality-v1'):
        report = read(name+'.json')
        require(report['status'] == 'passed' and report['worker_pids_gone'], f'Quality run incomplete: {name}')
        cases = [row for row in report['runs'] if row['name'] in expected]
        require(len(cases) == 20 and sum(row['quality_pass'] for row in cases) == 15,
                'Frozen Q2 quality scope or score changed')
        for row in cases:
            require(row['final']['tokens'] == expected[row['name']]['output']['token_ids'],
                    f'Historical Q2 IDs differ: {name}/{row["name"]}')
        thinking = [row for row in report['runs'] if row['name'].startswith('thinking-')]
        require(len(thinking) == 3 and all(row['quality_pass'] for row in thinking), 'Thinking regression')
        require(report['cancel']['worker_reused'], 'Cancellation replaced the worker')

    for mode in ('off', 'on', 'off-matched'):
        cli = read(f'cli-{mode}-delivery-v2.json')
        require(cli['status'] == 'passed' and cli['returncode'] == 0
                and cli['models']['data'][0]['id'] == f'flash-next-strata-q2-{mode}'
                and cli['normal_console_shutdown'] and cli['workers_exited']
                and not cli['remaining_descendants'] and cli['test_bypass'] is False,
                f'Formal CLI delivery did not pass: {mode}')
    return dict(verified=True, scope=index['scope'], archived_files=len(index['files']),
                actual_http_512_token_runs=18, matched_mtp_outputs=6, quality_ids_matched=80,
                formal_cli_modes=3, full_installer=False, integration_branch=False)


if __name__ == '__main__':
    print(json.dumps(verify(), ensure_ascii=False))
