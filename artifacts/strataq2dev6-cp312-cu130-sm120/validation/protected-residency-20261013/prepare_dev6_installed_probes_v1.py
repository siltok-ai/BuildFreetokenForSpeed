"""Derive native installed-wheel probes; configure residency without overriding engine code."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def emit(name, text):
    path = ROOT / name
    if path.exists():
        raise FileExistsError(path)
    path.write_text(text, encoding='utf-8')
    return dict(path=name, sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def main():
    source = ROOT / 'candidate_native_probe.py'
    state = source.read_text(encoding='utf-8')
    state = state.replace('moe_cache_size=args.cache,moe_cache_auto=False,',
        'moe_q2_resident_experts=9216,moe_q2_dynamic_slots=1024,moe_cache_size=args.cache,moe_cache_auto=False,')
    state = state.replace('    result["load_seconds"]=time.perf_counter()-started', '''    result["load_seconds"]=time.perf_counter()-started
    if result['runtime_version'] != '0.1.3+speed.strataq2dev6':
        raise RuntimeError('Wrong installed FreeToken revision')
    cache = llm.engine.moe_offload_cache
    from freetoken.moe.q2_residency import ProtectedQ2Cache
    from freetoken.engine.engine import CacheRebuildRejected
    if type(cache) is not ProtectedQ2Cache:
        raise RuntimeError('Explicit resident configuration did not select the native owner cache')
    pointers = {role: value.data_ptr() for role, value in cache.bank_caches.items()}
    graph_before = llm.engine.graph_runner
    rejected = False
    try:
        llm.engine.rebuild_runtime_cache(moe_cache_size=cache.cache_size - 1)
    except CacheRebuildRejected:
        rejected = True
    unchanged = graph_before is llm.engine.graph_runner and pointers == {role: value.data_ptr() for role, value in cache.bank_caches.items()}
    if not rejected or not unchanged or llm.engine.rebuild_teardown_started:
        raise RuntimeError('Unsafe expert resize did not reject before teardown')
    llm.engine.rebuild_runtime_cache(num_pages=llm.engine.num_pages)
    retained = pointers == {role: value.data_ptr() for role, value in cache.bank_caches.items()}
    if not retained:
        raise RuntimeError('KV-only rebuild changed expert owner addresses')
    result['rebuild_checks'] = dict(expert_resize_rejected_before_teardown=rejected and unchanged,
        kv_only_rebuild_preserves_owner_addresses=retained, fixed_capacity=cache.cache_size)
    result['ownership'] = dict(protected=len(cache.protected_payload.protected_slots),
        dynamic=cache.protected_payload.dynamic, host_bytes=cache.protected_payload.host_bytes,
        gpu_bank_bytes=cache.protected_payload.gpu_bank_bytes)
    result['process_engine_override'] = False
    result['native_engine_only'] = True''')
    records = [emit('dev6_installed_state_v1.py', state)]
    source = ROOT / 'first_freetoken_performance_dev5_v1.py'
    performance = source.read_text(encoding='utf-8').replace('speed.strataq2dev5', 'speed.strataq2dev6')
    performance = performance.replace("ROOT/'dev5-routed-runtime-gate-v1.json'", "ROOT/'dev6-installed-runtime-gate-v1.json'")
    performance = performance.replace('moe_cache_size=a.cache,moe_cache_auto=',
        'moe_q2_resident_experts=9216,moe_q2_dynamic_slots=1024,moe_cache_size=a.cache,moe_cache_auto=')
    performance = performance.replace('Prefix and GPU expert cold reset;', 'Prefix and dynamic LRU cold reset; fixed protected owners persist;')
    records.append(emit('dev6_installed_performance_v1.py', performance))
    component = '''import capture_freetoken_reference
import freetoken
from pathlib import Path
import pytest
assert freetoken.__version__ == '0.1.3+speed.strataq2dev6'
root=Path(__file__).resolve().parent / 'dev6-frozen-replay-v1/build-source/tests'
raise SystemExit(pytest.main(['-v',str(root/'moe/test_q2_residency.py'),str(root/'moe/test_q2_grouped.py'),str(root/'models/qwen4_exp/test_gguf_execution.py')]))
'''
    records.append(emit('test_dev6_installed_component_v1.py', component))
    emit('dev6-installed-probe-derivation-v1.json', json.dumps(dict(
        source_state_sha256=hashlib.sha256((ROOT/'candidate_native_probe.py').read_bytes()).hexdigest(),
        source_performance_sha256=hashlib.sha256(source.read_bytes()).hexdigest(), probes=records,
        process_engine_override=False, quality_acceptance=False), indent=2))
    print(json.dumps(records))


if __name__ == '__main__':
    main()
