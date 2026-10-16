"""核对修订8的Python修复及来自7的完整原生字节链，不授予运行验收。"""
import argparse
import json
from pathlib import Path
import tempfile
import zipfile
from prepare_source import ROOT,prepare,load_profile
from validate_python_revision import validate as validate_python
from validate_performance7_revision import validate as validate_native_base
import sys
sys.path.insert(0,str(ROOT/'build'))
from derive_python_materials import validate_python_only_changes


def validate(profile_path,materials):
    """原生7真实重建和修订8逐字节复用分别验证，不能继承7的运行状态。"""
    validate_native_base(ROOT/'build/profiles/strataq2perf7.json',ROOT/'artifacts/strataq2perf7-cp312-cu130-sm120')
    validate_python(profile_path,materials)
    materials=Path(materials)
    record=json.loads((materials/'build-record.json').read_text())
    profile,_=load_profile(profile_path)
    with tempfile.TemporaryDirectory(prefix='draft-policy-replay-') as directory:
        source,_=prepare(profile_path,Path(directory)/'source')
        before,_=prepare(ROOT/'build/profiles/strataq2perf7.json',Path(directory)/'base')
        changed=validate_python_only_changes(before,source)
        if sorted(changed)!=record['binary_reuse']['changed_source_files']:
            raise ValueError('Python-only source delta differs from reuse receipt')
        with zipfile.ZipFile(materials/'wheels'/profile['wheel_filename']) as archive:
            for p in (source/'python/freetoken').rglob('*'):
                if p.is_file() and p.suffix in ('.py','.cpp','.cu','.cuh','.h','.hpp'):
                    name=p.relative_to(source/'python').as_posix()
                    if archive.read(name)!=p.read_bytes():raise ValueError('Frozen source differs: '+name)
    return dict(verified=True,scope='offline_identity_only',version=profile['custom_version'],
        reused_native=3,reused_aot=27,native_base='strataq2perf7',rebuilt_from6_aot='freetoken__gguf_mmvq_r3')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--profile',type=Path,default=ROOT/'build/profiles/strataq2perf8.json')
    p.add_argument('--materials',type=Path,default=ROOT/'artifacts/strataq2perf8-cp312-cu130-sm120')
    args=p.parse_args();print(json.dumps(validate(args.profile,args.materials)))
