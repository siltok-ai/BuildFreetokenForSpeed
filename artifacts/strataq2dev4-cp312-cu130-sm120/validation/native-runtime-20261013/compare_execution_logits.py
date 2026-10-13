"""同一完整权重、固定 token 路径的 cold/prefix/graph/chunk logits 一致性。"""
import argparse
import json
from pathlib import Path
import torch
from safetensors.torch import load_file

ROOT=Path(__file__).resolve().parent
def compare(a,b,label):
    if a['input_sha256']!=b['input_sha256'] or a['token_ids']!=b['token_ids']:
        raise RuntimeError(f'fixed input/output differs: {label}')
    left,right=load_file(a['logits_file']),load_file(b['logits_file'])
    if left.keys()!=right.keys():
        raise RuntimeError(f'logit positions differ: {label}')
    rows=[]
    for key in left:
        x,y=left[key].float(),right[key].float()
        delta=(x-y).abs()
        row=dict(position=key,max_error=delta.max().item(),
            relative_l2=(delta.norm()/x.norm().clamp_min(1e-10)).item(),top1_equal=x.argmax().item()==y.argmax().item())
        rows.append(row)
    passed=all(r['top1_equal'] and r['relative_l2']<=2e-3 for r in rows)
    return dict(label=label,passed=passed,positions=len(rows),max_error=max(r['max_error'] for r in rows),
                max_relative_l2=max(r['relative_l2'] for r in rows),rows=rows)
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--other')
    args=parser.parse_args()
    path=ROOT/'candidate-prefix128-wheel.json'
    data=json.loads(path.read_text(encoding='utf-8'))
    runs={r['label']:r for r in data['runs']}
    checks=[compare(runs['prefix-cold'],runs[k],k) for k in ('prefix-hot','prefix-restored')]
    hits={k:runs[k]['engine_events'][0]['requests'][0]['cached_len'] for k in ('prefix-cold','prefix-hot','prefix-diverged','prefix-restored')}
    if hits['prefix-cold']!=0 or any(hits[k]!=64 for k in ('prefix-hot','prefix-diverged','prefix-restored')):
        raise RuntimeError(f'actual prefix reuse was not exercised: {hits}')
    if args.other:
        other=json.loads(Path(args.other).read_text(encoding='utf-8'))
        for r in other['runs']:
            checks.append(compare(runs[r['label']],r,'execution_'+r['label']))
    report=dict(passed=all(c['passed'] for c in checks),scope='FreeToken full-model execution consistency; not complete Strata engine alignment',
        real_prefix_cached_tokens=hits,checks=checks,relative_l2_limit=2e-3)
    (ROOT/'execution-logit-consistency.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(dict(passed=report['passed'],cached_tokens=hits,checks=[{k:v for k,v in c.items() if k!='rows'} for c in checks])))
    if not report['passed']:
        raise RuntimeError('execution logit consistency failed')
if __name__=='__main__':
    main()
