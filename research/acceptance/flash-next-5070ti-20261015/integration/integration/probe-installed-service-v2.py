"""检查未注入替身的真实Windows服务，记录发布身份和许可证边界。"""
from pathlib import Path
import argparse
import json
import time

import httpx


def main():
    """未授权服务检查拒绝契约；有推理资格的探索版本实际运行同步与SSE。"""
    parser=argparse.ArgumentParser()
    parser.add_argument('--commit',required=True)
    parser.add_argument('--model',default='flash-next-strata-q2-off')
    parser.add_argument('--expect-unlicensed',action='store_true')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    report=dict(status='running',test_bypass=False,transport='tcp',base_url='http://127.0.0.1:8000')
    try:
        with httpx.Client(base_url=report['base_url'],timeout=600,trust_env=False) as client:
            deadline=time.monotonic()+90
            while True:
                try:response=client.get('/health',timeout=2)
                except httpx.TransportError:
                    if time.monotonic()>deadline:raise
                    time.sleep(.5);continue
                response.raise_for_status();break
            health=response.json();report['health']=health
            if health['commit']!=args.commit or health['dirty']:
                raise RuntimeError('Installed service build identity changed')
            payload=dict(model=args.model,temperature=0,top_p=1,enable_thinking=False,max_tokens=32,
                         messages=[dict(role='user',content='计算31乘17。只输出一个整数。')])
            if args.expect_unlicensed:
                if health['status']!='unlicensed':raise RuntimeError('License state unexpectedly changed')
                response=client.post('/v1/chat/completions',json=payload)
                report['chat_denied']=dict(status_code=response.status_code,body=response.json())
                if response.status_code!=503 or response.json().get('error',{}).get('code')!='engine_unlicensed':
                    raise RuntimeError('Unlicensed inference did not match the dev engine_unlicensed contract')
            else:
                response=client.get('/v1/models');response.raise_for_status();report['models']=response.json()
                if args.model not in {item['id'] for item in report['models']['data']}:
                    raise RuntimeError('Installed service model missing')
                response=client.post('/v1/chat/completions',json=payload)
                response.raise_for_status();report['arithmetic']=response.json()
                if report['arithmetic']['choices'][0]['message']['content'].strip()!='527':
                    raise RuntimeError('Installed service arithmetic changed')
                text=[];usage=None;done=0
                with client.stream('POST','/v1/chat/completions',json=payload|dict(stream=True,
                                   stream_options=dict(include_usage=True))) as response:
                    response.raise_for_status()
                    for line in response.iter_lines():
                        if not line.startswith('data: '):continue
                        if line=='data: [DONE]':done+=1;continue
                        body=json.loads(line[6:])
                        if body.get('usage'):usage=body['usage']
                        text.extend(choice.get('delta',{}).get('content') or '' for choice in body.get('choices',[]))
                report['sse']=dict(text=''.join(text),usage=usage,done_events=done)
                if ''.join(text).strip()!='527' or not usage or done!=1:
                    raise RuntimeError('Installed service SSE contract changed')
        report['status']='passed'
    except BaseException as error:
        report.update(status='failed',error=repr(error));raise
    finally:
        args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(dict(status=report['status'],output=str(args.output))),flush=True)


if __name__=='__main__':
    main()
