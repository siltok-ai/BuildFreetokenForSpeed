"""Keep long-output identity and startup variation visible alongside throughput medians."""
from pathlib import Path
import hashlib
import json
R=Path(__file__).parent
summary=json.loads((R/'http-abba-summary-v3.json').read_text())
result=dict(scope='final48 HTTP outputs; text identity diagnostics, separate from frozen formal quality IDs',modes={})
for mode,variants in summary['reports'].items():
    groups={};resources=[]
    for variant,names in variants.items():
        for name in names:
            path=R/name
            report=json.loads(path.read_text())
            resource=json.loads(path.with_suffix('.resource.json').read_text())
            resources.append(dict(report=name,elapsed_s=resource['elapsed_s']))
            for row in report['runs']:
                context='70-0' if row['label'].startswith('70-0') else 'diverse-4096-0'
                value=dict(text=row['text'],reasoning_content=row.get('reasoning_content',''))
                digest=hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
                groups.setdefault(context,{}).setdefault(variant,[]).append(dict(report=name,label=row['label'],
                    output_sha256=digest,output_tokens=row['output_tokens'],decode_tps=row['decode_tps'],
                    repeated_bang32_text='!'*32 in row['text']))
    identities={context:{variant:sorted({row['output_sha256'] for row in rows})
                          for variant,rows in variants.items()} for context,variants in groups.items()}
    result['modes'][mode]=dict(outputs=groups,identities=identities,total_run_elapsed=resources)
target=R/'abba-output-audit-v1.json'
if target.exists():raise FileExistsError(target)
target.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
print(json.dumps({mode:dict(identities=data['identities'],elapsed=data['total_run_elapsed'])
                  for mode,data in result['modes'].items()}))
