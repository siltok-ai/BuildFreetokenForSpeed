"""Compare matched64-token Nsight captures; do not treat diagnostic throughput as acceptance."""
from pathlib import Path
from collections import defaultdict
from statistics import median
import json
import sqlite3

ROOT=Path(__file__).resolve().parent


def analyze(stem):
    report=json.loads((ROOT/(stem+'.json')).read_text(encoding='utf-8'))
    resource=json.loads((ROOT/(stem+'.resource.json')).read_text(encoding='utf-8'))
    assert report['status']=='passed' and report['capture']['steps']==64
    assert resource['returncode']==0 and not resource['remaining_descendants']
    with sqlite3.connect('file:'+(ROOT/(stem+'.sqlite')).as_posix()+'?mode=ro',uri=True) as c:
        c.text_factory=lambda raw:raw.decode('utf-8',errors='replace')
        names=dict(c.execute('SELECT id,value FROM StringIds'))
        kernels=c.execute('SELECT start,end,shortName,gridX,gridY,blockX,blockY FROM CUPTI_ACTIVITY_KIND_KERNEL ORDER BY start').fetchall()
        groups=defaultdict(lambda:[0,0])
        geometries=defaultdict(lambda:[0,0])
        union=0;previous=0;gaps=[]
        for start,end,name,gx,gy,bx,by in kernels:
            union+=max(0,end-max(start,previous))
            if previous and start>previous:gaps.append((start-previous)/1e6)
            previous=max(previous,end)
            groups[names[name]][0]+=1;groups[names[name]][1]+=end-start
            geometries[(names[name],gx,gy,bx,by)][0]+=1
            geometries[(names[name],gx,gy,bx,by)][1]+=end-start
        span=max(row[1] for row in kernels)-kernels[0][0]
        apis=[dict(name=name,calls=count,ms_per_token=duration/64e6) for name,count,duration in c.execute(
            'SELECT s.value,count(*),sum(a.end-a.start) FROM CUPTI_ACTIVITY_KIND_RUNTIME a '
            'JOIN StringIds s ON a.nameId=s.id GROUP BY a.nameId ORDER BY sum(a.end-a.start) DESC LIMIT 15')]
        nvtx=c.execute('SELECT text,start,end FROM NVTX_EVENTS WHERE text IN (?,?)',
            ('steady_native_token','steady_mtp_window')).fetchall()
        tables={row[0] for row in c.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        copies=[]
        if 'CUPTI_ACTIVITY_KIND_MEMCPY' in tables:
            copies=[dict(kind=k,count=n,bytes=b,ms_per_token=t/64e6) for k,n,b,t in c.execute(
                'SELECT copyKind,count(*),sum(bytes),sum(end-start) FROM CUPTI_ACTIVITY_KIND_MEMCPY GROUP BY copyKind')]
    rows=[dict(name=name,calls=n,ms_per_token=t/64e6) for name,(n,t) in sorted(groups.items(),key=lambda item:-item[1][1])]
    geometry=[dict(name=key[0],grid=list(key[1:3]),block=list(key[3:]),calls=n,ms_per_token=t/64e6)
        for key,(n,t) in sorted(geometries.items(),key=lambda item:-item[1][1])[:25]]
    # Every traced graph must include all48 expert layers; this also detects boundary/partial captures.
    assert groups['_owner_gather'][0]==64*48,groups['_owner_gather']
    assert len(nvtx)==64,len(nvtx)
    return dict(args=report['args'],nvtx_ranges=len(nvtx),kernel_count=len(kernels),
        kernel_sum_ms_per_token=sum(end-start for start,end,*_ in kernels)/64e6,
        gpu_union_ms_per_token=union/64e6,gpu_span_ms_per_token=span/64e6,
        gpu_idle_ms_per_token=(span-union)/64e6,max_gpu_gap_ms=max(gaps),
        median_gpu_gap_ms=median(gaps),top_kernels=rows[:25],top_geometries=geometry,apis=apis,copies=copies,
        captured_request={k:v for k,v in next(row for row in report['runs'] if row['name']==report['capture']['request']).items() if k!='output'},
        resource=resource,scope='64 native decode steps192..255 within70-0-1 aftersame3EN/3CN/3code history;profiler overhead,not throughput acceptance')


def main():
    result={stem:analyze(stem) for stem in ('nsys-off-late-r5120-v1','nsys-off-late-r5376-v1')}
    target=ROOT/'nsys-off-late-comparison-v2.json'
    if target.exists():raise FileExistsError(target)
    target.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    for stem,r in result.items():
        print(json.dumps(dict(stem=stem,kernel_sum=r['kernel_sum_ms_per_token'],union=r['gpu_union_ms_per_token'],
            span=r['gpu_span_ms_per_token'],idle=r['gpu_idle_ms_per_token'],top=r['top_kernels'][:8],apis=r['apis'][:5],
            diagnostic_whole_request_tps=r['captured_request']['decode_tps'])))


if __name__=='__main__':main()
