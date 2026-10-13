"""Supervised research entry point; explicitly tags engine overrides in all results."""
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent


def main():
    mode = sys.argv[1]
    if mode not in ('state', 'performance'):
        raise ValueError('Expected state or performance')
    sys.argv.pop(1)
    output = Path(sys.argv[sys.argv.index('--output') + 1])
    if output.exists():
        raise FileExistsError(output)
    import freetoken
    if freetoken.__version__ != '0.1.3+speed.strataq2dev5':
        raise RuntimeError('Unexpected installed native base')
    import prototype_q2_protected_cache_v1 as prototype
    prototype.install(protected=9216)
    if mode == 'performance':
        state = json.loads((ROOT / 'prototype-protected-cache-state-gate-11264-v1.json').read_text(encoding='utf-8'))
        if not state['passed'] or state['prototype']['source_sha256'] != hashlib.sha256(Path(prototype.__file__).read_bytes()).hexdigest():
            raise RuntimeError('Exact protected-cache prototype has not passed full-model state checks')
        import first_freetoken_performance_dev5_v1 as harness
    else:
        import candidate_native_probe as harness
    try:
        harness.main()
    finally:
        if output.exists():
            result = json.loads(output.read_text(encoding='utf-8'))
            result['native_engine_only'] = False
            result['process_engine_override'] = True
            result['full_runtime_acceptance'] = False
            result['speed_acceptance'] = False
            result['prototype'] = prototype.evidence()
            result['cold_cache_semantics'] = 'Prefix and dynamic LRU reset; fixed protected owners survive all requests. Unlike the dev5 full LRU cold reset, permanent owners remain resident.'
            if 'status' in result:
                result['status'] = 'protected_cache_prototype_' + result['status']
            result['wrapper_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
            output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
