"""Frozen native Strata engine, documented event profiling, no source changes."""
import hashlib
import json
from pathlib import Path
ROOT = Path(__file__).resolve().parent
source = ROOT / 'benchmark_strata_q2_local_v2.py'
target = ROOT / 'benchmark_strata_gpu_stages_v1.py'
if target.exists():
    raise FileExistsError(target)
text = source.read_text(encoding='utf-8')
text = text.replace("env = os.environ.copy()", "env = os.environ.copy()\n    env['STRATA_DECODE_TIMING'] = '1'\n    env['STRATA_VERIFY_PROFILE'] = '1'")
text = text.replace('No CUDA instrumentation or layer dumps.', 'Native CUDA-event stage profiling enabled. Diagnostic only; instrumented throughput is not a sustained speed benchmark.')
start = text.index("            report['thinking_smoke'] = []")
end = text.index("            report['all_measurements_sufficient']", start)
text = text[:start] + "            report['thinking_smoke'] = []\n            report['thinking_skipped_reason'] = 'This is stage attribution only; existing frozen thinking quality evidence remains separate'\n" + text[end:]
text = text.replace("report['status'] = 'measured'", "report['status'] = 'diagnostic_gpu_stages_complete'\n            report['profiling_env'] = dict(STRATA_DECODE_TIMING='1', STRATA_VERIFY_PROFILE='1')\n            report['speed_acceptance'] = False")
target.write_text(text, encoding='utf-8')
(ROOT / 'strata-stage-diagnostic-harness-identity-v1.json').write_text(json.dumps(dict(
    source=str(source),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
    derived=str(target),derived_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
    native_engine_source_changed=False,mtp_enabled=False,
    reference='Frozen docs/DETAILS.md:297-304, native packs use serve-path CUDA events, not --gpu-stages',
    changes=['Enable native CPU and GPU stage timing environment flags','Diagnostic status and instrumented speed scope','Omit already frozen thinking suite from stage attribution job']),indent=2),encoding='utf-8')
