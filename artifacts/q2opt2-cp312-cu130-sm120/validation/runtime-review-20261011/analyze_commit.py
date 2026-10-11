"""Summarize real capacity records without equating process private memory with system commit."""
import json
from pathlib import Path

root = Path(__file__).parent
records = []
for name in ('packaged-flash-backend-capacity-002', 'packaged-flash-http-49p5-002',
             'packaged-flash-http-default'):
    path = root/name/'capacity.json'
    if not path.exists():
        continue
    record = json.loads(path.read_text(encoding='utf-8'))
    report_path = path.with_name('report.json')
    report = json.loads(report_path.read_text(encoding='utf-8')) if report_path.exists() else {}
    worker_pid = report.get('worker_pid')
    if worker_pid is None:
        worker_pid = report.get('worker', {}).get('pid')
    initial = next((item for item in record['samples']
                    if any(p['pid'] == worker_pid for p in item['processes'])), None)
    if 'admission_probe' in report:
        initial_commit = record['baseline']['commit_limit'] - report['admission_probe']['current']['commit_available_bytes']
        reference = 'HTTP admission snapshot before worker load'
    elif initial:
        initial_commit = initial['commit_used']
        reference = 'first sample containing the new worker; may include initial allocation'
    else:
        initial_commit = record['baseline']['commit_used']
        reference = 'launcher baseline; includes launcher/import overhead'
    records.append({'case': name, 'returncode': record['returncode'],
                    'peak_system_commit_gib': record['peak_commit_used']/1024**3,
                    'reference_system_commit_gib': initial_commit/1024**3,
                    'increment_gib': (record['peak_commit_used']-initial_commit)/1024**3,
                    'reference': reference,
                    'min_commit_available_gib': record['min_commit_available']/1024**3,
                    'min_physical_available_gib': record['min_physical_available']/1024**3,
                    'safety_abort': record['safety_abort'],
                    'wall_seconds': record['wall_seconds']})
(root/'commit-measurements.json').write_text(json.dumps(records, indent=2)+'\n', encoding='utf-8')
print(json.dumps(records, indent=2))
