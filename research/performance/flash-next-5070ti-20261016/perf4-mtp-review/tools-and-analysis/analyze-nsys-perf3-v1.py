"""Compare complete warmed capture work with the same Nsight collection method."""
from pathlib import Path
import collections
import json
import sqlite3

ROOT = Path(__file__).parent
PATHS = {
    'freetoken': ROOT / 'nsys-perf3-mtp-v1.sqlite',
    'strata': Path(r'C:\workspace\runs\strata\flash-perf-20261016\trace-mtp-v3.sqlite'),
}


def analyze(path):
    """Report GPU work separately from host waits; captures are not TPS evidence."""
    with sqlite3.connect(f'file:{path.as_posix()}?mode=ro', uri=True) as connection:
        connection.text_factory = lambda raw: raw.decode('utf-8', errors='replace')
        names = dict(connection.execute('SELECT id,value FROM StringIds'))
        kernels = connection.execute('SELECT start,end,shortName FROM CUPTI_ACTIVITY_KIND_KERNEL ORDER BY start').fetchall()
        total = 0
        merged_end = 0
        for start, end, _ in kernels:
            total += max(0, end-max(start, merged_end))
            merged_end = max(merged_end, end)
        groups = collections.defaultdict(lambda: [0, 0])
        for start, end, name in kernels:
            groups[names[name]][0] += 1
            groups[names[name]][1] += end-start
        rows = [dict(name=name,calls=values[0],ms_per_window=values[1]/64e6)
                for name, values in sorted(groups.items(),key=lambda row:-row[1][1])]
        head = [dict(zip(('grid_x','grid_y','block_x','block_y','calls','ms_per_call'),row)) for row in connection.execute(
            'SELECT gridX,gridY,blockX,blockY,count(*),avg(end-start)/1e6 '
            'FROM CUPTI_ACTIVITY_KIND_KERNEL WHERE shortName IN '
            '(SELECT id FROM StringIds WHERE value = ?) GROUP BY gridX,gridY,blockX,blockY',
            ('native_q5_k_mmvq_kernel',))]
        apis = [dict(name=row[0],calls=row[1],ms_per_window=row[2]) for row in connection.execute(
            'SELECT s.value,count(*),sum(a.end-a.start)/64e6 FROM CUPTI_ACTIVITY_KIND_RUNTIME a '
            'JOIN StringIds s ON a.nameId=s.id GROUP BY a.nameId ORDER BY sum(a.end-a.start) DESC LIMIT 12')]
    return dict(windows=64,kernel_count=len(kernels),kernel_sum_ms=sum(e-s for s,e,_ in kernels)/64e6,
                gpu_union_ms=total/64e6,gpu_span_ms=(max(e for _,e,_ in kernels)-kernels[0][0])/64e6,
                top_kernels=rows[:25],q5_draft_head=head,apis=apis)


result = dict(scope='diagnostic64_windows; different output trajectories; no throughput claim',
              captures={name:analyze(path) for name,path in PATHS.items()},
              finding='Q5 draft head launches248320 rows in FreeToken,106299 in Strata; source confirms draft vocabulary subset')
output = ROOT / 'nsys-perf3-comparison-v1.json'
if output.exists():
    raise FileExistsError(output)
output.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
print(json.dumps(result,indent=2))
