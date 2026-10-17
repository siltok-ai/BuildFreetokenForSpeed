"""Append supplemental protected ABBA evidence without replacing the accepted primary run."""
from pathlib import Path
import hashlib
import json
import shutil
import sys
R=Path(__file__).parent
B=Path(r'C:\workspace\wt\freetoken-mats\flash-perf')
T=R.parent/'review-20261017/consumer-tools'
sys.path.insert(0,str(B/'tools'))
from verify_performance8_consumer import frozen_index
from verify_performance_consumer import performance
def read(path):return json.loads(path.read_text(encoding='utf-8-sig'))
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    record=read(R/'compiled-followup-complete-v2.json')
    assert record['status']=='measured'
    contracts={label:read(B/f'integration/speed/consumer-contract-strataq2{tag}-compiled.json')
               for label,tag in [('baseline','review1'),('candidate','review2b')]}
    source=read(B/'integration/speed/consumer-contract-strataq2review2b.json')
    for label,names in record['reports'].items():
        for name in names:
            path=R/'compiled-off-followup-v2/results'/name
            report=read(path);origin=read(path.with_suffix('.origin.json'))
            performance(report)
            assert report['prompts_sha256']==source['prompts_sha256']
            assert origin['bundle_sha256']==contracts[label]['speed_bundle']['sha256']
            assert origin['build_commit']==contracts[label]['speed_commit']
    E=B/'research/review/20261017-evening/candidate-b-compiled-off-abba-v1'
    E.mkdir(parents=True,exist_ok=False)
    shutil.copytree(R/'compiled-off-followup-v2/results',E/'runs')
    for name in ('run-compiled-off-followup.py','run-compiled-off-followup-v2.py',
                 'compiled-off-followup-driver-v1.log','compiled-off-followup-driver-v2.log',
                 'compiled-followup-complete-v2.json','freeze-compiled-followup.py'):
        shutil.copy2(R/name,E/name)
    shutil.copytree(T,E/'consumer-tools')
    files={path.relative_to(E).as_posix():dict(bytes=path.stat().st_size,sha256=sha(path))
           for path in sorted(E.rglob('*')) if path.is_file()}
    index=dict(format=1,scope='supplemental_old_new_protected_off_abba_not_new_installation',
        original_primary_retained=True,actual_install_validated=False,clean_os_validated=False,
        source_contract_sha256=sha(B/'integration/speed/consumer-contract-strataq2review2b.json'),
        variants=record['variants'],files=files)
    (E/'index.json').write_text(json.dumps(index,indent=2)+'\n',encoding='utf-8')
    reference=dict(index=(E/'index.json').relative_to(B).as_posix(),sha256=sha(E/'index.json'))
    frozen_index(reference)
    lines=['','Supplemental protected OFF ABBA comparison after a low primary short sample:',
        'Old/new actual embedded payloads, same1024prefill/5760resident/weights/prompts/options,',
        'four sequential rounds ABBA,24real512 outputs, no concurrent CPU build/tests.',
        'The original primary compiled short median91.694tps is retained in its frozen contract.',
        'Supplementary pooled medians (six per input/variant), old -> new:']
    for context,row in record['summary'].items():
        lines.append(f"  {context}: {row['median_tps']['baseline']:.3f} -> {row['median_tps']['candidate']:.3f} ({row['change_percent']:+.3f}%).")
    lines.extend(['All four runs use their own accepted embedded interpreter/protected loader/bundle SHA;',
        'all measured512 counts, sampled resource guards and natural owned process exits pass.',
        'The first supervisor launch used an unsuitable host interpreter and exited before producing',
        'an embedded run; its driver/config are preserved. Freshv2 uses the candidate task interpreter.',
        'This supplemental measurement adds no actual-install or clean-OS qualification.',
        reference['index'],'IndexSHA256: '+reference['sha256'],''])
    with (B/'docs/flash-next-review2-20261017.txt').open('a',encoding='utf-8',newline='\n') as handle:
        handle.write('\n'.join(lines))
    (R/'supplemental-checks-v1.json').write_text(json.dumps(dict(status='passed',evidence=reference,
        summary=record['summary'],runtime_code_changed=False),indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(status='passed',evidence=reference,summary=record['summary'])))

if __name__=='__main__':main()
