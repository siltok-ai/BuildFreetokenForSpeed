"""复核全GGUF/MTP可搬迁交付的原始记录；不将其扩大为整包验收。"""
from pathlib import Path
import gzip
import hashlib
import json

ROOT=Path(__file__).resolve().parents[1]
EVIDENCE=ROOT/'research/mtp/evidence/portable-delivery-20261014-v1'


def verify():
    """校验封存字节并交叉核对真实转换、开关推理、来源与导出结果。"""
    index=json.loads((EVIDENCE/'index.json').read_text(encoding='utf-8'))
    raw={}
    for row in index['raw']:
        path=(EVIDENCE/row['path']).resolve()
        if not path.is_relative_to(EVIDENCE.resolve()):
            raise ValueError('Evidence escaped its directory')
        packed=path.read_bytes();data=gzip.decompress(packed)
        if (hashlib.sha256(packed).hexdigest()!=row['gzip_sha256']
                or hashlib.sha256(data).hexdigest()!=row['sha256'] or len(data)!=row['bytes']):
            raise ValueError('Archived evidence changed: '+row['name'])
        if row['name'] in raw:
            raise ValueError('Duplicate evidence name')
        raw[row['name']]=data
    def record(name): return json.loads(raw[name])
    runtime=record('runtime-v2.json');resource=record('runtime-v2-resource.json')
    if runtime['status']!='passed' or resource['returncode']!=0 or {r['mode'] for r in runtime['runs']}!={'on','off'}:
        raise ValueError('Installed delivery run did not pass both modes')
    gold=record('frozen-http-comparator.json')['runs']
    for mode in runtime['runs']:
        if not mode['passed'] or not mode['worker_gone'] or mode['eos']['text'].strip()!='527' or len(mode['runs'])!=2:
            raise ValueError('Runtime lifecycle or EOS check incomplete')
        for run in mode['runs']:
            final=run['final'];expected=next(g for g in gold if g['label']=='70-0-'+str(run['repeat']))
            if (len(run['token_ids'])!=512 or run['token_ids']!=final['tokens']
                    or final['usage']['output_tokens']!=512 or final['finish_reason']!='length'
                    or final['text']!=expected['text'] or not run['matches_frozen_text']):
                raise ValueError('Delivered tokens differ from the accepted corresponding repeat')
    source=record('mtp-source-verification.json');conversion=record('mtp-conversion-report.json')
    lock=json.loads((ROOT/'model-weights/qwen3.8-flash-next-strata-q2/source-lock.json').read_text(encoding='utf-8'))
    frozen=json.loads((ROOT/'model-weights/qwen3.8-flash-next-strata-q2/mtp-runtime.json').read_text(encoding='utf-8'))
    actual={r['name']:r for r in conversion['tensors']}
    if (not source['passed'] or source['complete_tensor_count']!=31 or len(actual)!=31
            or source['total_bytes']!=lock['mtp']['source_bytes']
            or hashlib.sha256(raw['mtp-source-verification.json']).hexdigest()!=conversion['source_verification_sha256']):
        raise ValueError('Rebuilt MTP source proof incomplete')
    for row in frozen['tensors']:
        if any(actual[row['name']][k]!=row[k] for k in ('shape','kind','type_id','bytes','sha256','source_sha256')):
            raise ValueError('Rebuilt MTP identity differs from accepted payloads')
    reproduction=record('data-reproduction-summary.json')
    if (reproduction['new_runtime_sha256']!=lock['mtp']['manifest_sha256']
            or not reproduction['base_configuration_identical_to_accepted_checkpoint']
            or reproduction['new_source_verification_sha256']==reproduction['historical_source_verification_sha256']):
        raise ValueError('Different-root reproduction proof incomplete')
    if record('mtp-conversion-resource.json')['returncode']!=0:
        raise ValueError('MTP conversion did not finish')
    if record('export-record.json')['source']!='git staged tree; no untracked files':
        raise ValueError('Export used untracked files')
    for name in ('all-tests.log','export-tests.log'):
        if b'Ran 29 tests' not in raw[name] or b'\nOK\n' not in raw[name].replace(b'\r\n',b'\n'):
            raise ValueError('Repository tests were not green')
    return dict(status='verified',archived_files=len(raw),rebuilt_mtp_tensors=31,
                installed_modes=['on','off'],actual_512_token_runs=4,full_installer_clean_machine=False)


if __name__=='__main__':
    print(json.dumps(verify()))
