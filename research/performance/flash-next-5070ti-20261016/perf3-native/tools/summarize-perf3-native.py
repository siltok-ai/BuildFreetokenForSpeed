"""对正式普通wheel请求做计数、资源及原型输出交叉核对。"""
from pathlib import Path
import json
import statistics

RUN=Path(__file__).parent
KEYS=('english','chinese','code','70-0','diverse-4096-0')


def main():
    """要求两模式全部完成，保留首轮TTFT，避免用前缀热启动冒充冷prefill。"""
    result=dict(scope='ordinary_freetoken_wheel_not_speed_http_or_installer',
        version='0.1.3+speed.strataq2perf3',
        protocol='Same full Q2 weights and frozen inputs;3 sequential requests/key,512 delivered tokens. '
                 'Adaptive experts and prefix cache persist across requests; fresh process per mode. '
                 'TPS=(N-1)/(last-first delivery); first TTFT and all request times preserved.',
        modes={})
    for mode,proto in [('off','off3-combined-perf-v1'),('on','mtp3-combined-perf-v1')]:
        name=f'candidate3-wheel-performance-{mode}-v1'
        data=json.loads((RUN/(name+'.json')).read_text(encoding='utf-8'))
        resource=json.loads((RUN/(name+'.resource.json')).read_text())
        assert data['status']=='passed' and data['shutdown_complete']
        assert data['version']==result['version'] and not data['diagnostic']
        assert resource['returncode']==0 and resource['stop_reason'] is None and not resource['remaining_descendants']
        old=json.loads((RUN/(proto+'.json')).read_text(encoding='utf-8'))
        oldrows={row['name']:row for row in old['runs']}
        rows={}
        for key in KEYS:
            selected=[row for row in data['runs'] if row['name'] in [f'{key}-{i}' for i in range(3)]]
            assert len(selected)==3
            for row in selected:
                assert row['output_tokens']==row['arrivals']==len(row['output']['token_ids'])==512
                assert row['terminal']['finished'] and row['terminal']['reason']=='length'
            rows[key]=dict(median_tps=statistics.median(row['decode_tps'] for row in selected),
                tps=[row['decode_tps'] for row in selected],
                ttft_ms=[row['ttft_ms'] for row in selected],
                request_seconds=[row['elapsed_seconds'] for row in selected],
                prototype_ids_equal=[row['output']['token_ids']==oldrows[row['name']]['output']['token_ids'] for row in selected])
        result['modes'][mode]=dict(report=name+'.json',resource=resource,
            settings={key:data['args'][key] for key in ('resident','dynamic','prefill','state_cache_slots','mtp','adaptive_owners','adapt_every')},
            workloads=rows)
    result['mtp_gain_vs_best_measured_off']={key:result['modes']['on']['workloads'][key]['median_tps']/
        result['modes']['off']['workloads'][key]['median_tps']-1 for key in KEYS}
    target=RUN/'perf3-native-summary-v1.json'
    if target.exists():raise FileExistsError(target)
    target.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':main()
