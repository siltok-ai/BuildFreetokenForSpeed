"""同一perf4普通wheel的实际交付、质量和资源汇总，不能替代Speed验收。"""
from pathlib import Path
import json
import statistics

RUN=Path(__file__).parent
KEYS=('english','chinese','code','70-0','diverse-4096-0')


def read(name):
    return json.loads((RUN/name).read_text(encoding='utf-8-sig'))


def check_resource(stem):
    """拒绝未完成或被资源保护终止的结果。"""
    data=read(stem+'.json');resource=read(stem+'.resource.json')
    assert data['status']=='passed' and data['shutdown_complete']
    assert resource['returncode']==0 and resource['stop_reason'] is None and not resource['remaining_descendants']
    return data,resource


def main():
    """完整请求IDs与同模式perf3对照，不跨推理框架强求token相同。"""
    result=dict(version='0.1.3+speed.strataq2perf4',scope='ordinary_wheel_not_speed_or_installer',modes={})
    for mode in ('on','off'):
        quality,qres=check_resource(f'candidate4-wheel-quality-{mode}-v1')
        oldquality=read(f'candidate3-wheel-quality-{mode}-v1.json')
        oldrows={r['name']:r for r in oldquality['runs']}
        qids=[r['output']['token_ids']==oldrows[r['name']]['output']['token_ids'] for r in quality['runs']]
        data,res=check_resource(f'candidate4-wheel-performance-{mode}-v1')
        assert data['version']==result['version'] and not data['diagnostic']
        old=read(f'candidate3-wheel-performance-{mode}-v1.json')
        oldrows={r['name']:r for r in old['runs']}
        ids=[r['output']['token_ids']==oldrows[r['name']]['output']['token_ids'] for r in data['runs']]
        workloads={}
        for key in KEYS:
            rows=[r for r in data['runs'] if r['name'] in [f'{key}-{i}' for i in range(3)]]
            assert len(rows)==3
            for row in rows:
                assert row['output_tokens']==row['arrivals']==len(row['output']['token_ids'])==512
                assert row['terminal']['reason']=='length' and row['terminal']['finished']
            workloads[key]=dict(median_tps=statistics.median(r['decode_tps'] for r in rows),
                tps=[r['decode_tps'] for r in rows],ttft_ms=[r['ttft_ms'] for r in rows],
                elapsed_seconds=[r['elapsed_seconds'] for r in rows])
        result['modes'][mode]=dict(settings=data['args'],embedding=data['embedding_placement'],
            quality=dict(score=quality['score'],thinking_score=quality['thinking_score'],regressions=quality['regressions'],
                ids_equal=sum(qids),requests=len(qids),min_gpu_free=qres['min_gpu_free']),
            performance=dict(workloads=workloads,ids_equal=sum(ids),requests=len(ids),resource=res))
    result['mtp_gain']={k:result['modes']['on']['performance']['workloads'][k]['median_tps']/
        result['modes']['off']['performance']['workloads'][k]['median_tps']-1 for k in KEYS}
    out=RUN/'perf4-native-summary-v1.json'
    if out.exists():raise FileExistsError(out)
    out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':main()
