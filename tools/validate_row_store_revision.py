"""核对行缓存修订的真实重编扩展、另外两扩展/AOT复用及完整源码。"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import zipfile
from prepare_source import ROOT, prepare, load_profile, sha256
sys.path.insert(0,str(ROOT/'build'))
from derive_python_materials import validate_row_store_changes, ROW_STORE_CPP


def validate(profile_path, materials):
    """逐字节区分重编和复用材料；不把离线身份检查视为运行验收。"""
    profile,series=load_profile(profile_path)
    materials=Path(materials).resolve(strict=True)
    record=json.loads((materials/'build-record.json').read_text(encoding='utf-8'))
    for key in ('custom_version','input_sha256','source_commit','wheel_filename','torch','cuda','sm'):
        if record[key]!=profile[key]:raise ValueError('Row-store material profile differs: '+key)
    subprocess.run([sys.executable,'-B',str(ROOT/'tools/verify_materials.py'),str(materials)],check=True)
    reuse=record['binary_reuse']
    if Path(reuse['base_materials']).name!=reuse['base_materials']:raise ValueError('Base materials path escapes artifacts')
    base=ROOT/'artifacts'/reuse['base_materials']
    if sha256(base/'build-record.json')!=reuse['base_build_record_sha256']:raise ValueError('Base identity changed')
    baseline=json.loads((base/'build-record.json').read_text(encoding='utf-8'))
    native_record=record['row_store_rebuild']
    if native_record['module']!='_row_store' or native_record['source']!=ROW_STORE_CPP:
        raise ValueError('Wrong native module rebuilt')
    with tempfile.TemporaryDirectory(prefix='row-cache-replay-') as directory:
        directory=Path(directory)
        source,_=prepare(profile_path,directory/'candidate')
        base_profile_path=(ROOT/native_record['base_profile']).resolve(strict=True)
        if not base_profile_path.is_relative_to(ROOT/'build/profiles'):
            raise ValueError('Base profile escapes the profile catalog')
        base_profile,_=load_profile(base_profile_path)
        if any(base_profile[key]!=baseline[key] for key in ('custom_version','input_sha256','source_commit')):
            raise ValueError('Base profile differs from reused material identity')
        before,_=prepare(base_profile_path,directory/'base')
        changed=validate_row_store_changes(before,source)
        if sorted(changed)!=reuse['changed_source_files']:raise ValueError('Source delta receipt differs')
        receipt=native_record['receipt']
        if receipt['source_sha256']!=sha256(source/ROW_STORE_CPP):raise ValueError('Rebuilt C++ source differs')
        if receipt['torch']!=profile['torch'] or receipt['torch_cuda']!=profile['cuda']:
            raise ValueError('Native compiler Torch differs')
        with zipfile.ZipFile(materials/'wheels'/profile['wheel_filename']) as wheel:
            expected={p.relative_to(source/'python').as_posix():p.read_bytes()
                for p in (source/'python/freetoken').rglob('*') if p.is_file()
                and (p.suffix in ('.py','.cpp','.cu','.cuh','.h','.hpp')
                    or p.relative_to(source/'python/freetoken').as_posix().startswith('moe/configs/'))}
            for name,data in expected.items():
                if wheel.read(name)!=data:raise ValueError('Wheel source differs: '+name)
            py_names={n for n in wheel.namelist() if n.startswith('freetoken/') and n.endswith('.py')}
            if py_names!={n for n in expected if n.endswith('.py')}:raise ValueError('Wheel Python inventory differs')
            actual={name:hashlib.sha256(wheel.read(name)).hexdigest() for name in record['native_modules']}
            if actual!=record['native_sha256']:raise ValueError('Native inventory hash differs')
            row=next(n for n in actual if Path(n).name.startswith('_row_store.'))
            if actual[row]!=receipt['artifact']['sha256'] or len(wheel.read(row))!=receipt['artifact']['bytes']:
                raise ValueError('Rebuilt row-store bytes differ from compiler receipt')
            reused={n:d for n,d in actual.items() if n!=row}
            if len(reused)!=2 or reused!=reuse['native_sha256']:raise ValueError('Expected exactly two reused modules')
            with zipfile.ZipFile(base/'wheels'/baseline['wheel_filename']) as old:
                if reused!={n:hashlib.sha256(old.read(n)).hexdigest() for n in reused}:
                    raise ValueError('Other native module bytes changed')
    if record['patches']!=series['patches']:raise ValueError('Ordered patch receipt differs')
    before={n:r for n,r in baseline['artifacts'].items() if n.endswith('.dll')}
    after={n:r for n,r in record['artifacts'].items() if n.endswith('.dll')}
    if len(after)!=27 or after!=before:raise ValueError('AOT reuse differs from accepted base')
    receipt_path=(materials/native_record['receipt_path']).resolve(strict=True)
    if not receipt_path.is_relative_to(materials) or sha256(receipt_path)!=native_record['receipt_sha256']:
        raise ValueError('Compiler receipt file differs')
    if json.loads(receipt_path.read_text(encoding='utf-8'))!=receipt:raise ValueError('Embedded compiler receipt differs')
    if sha256(materials/'row-store-build.log')!=native_record['log_sha256']:raise ValueError('Compiler log differs')
    return dict(verified=True,scope='offline_identity_only',version=profile['custom_version'],
        rebuilt_native='_row_store',reused_native=2,reused_aot=27,python_files=len(py_names))


def main():
    """要求显式候选路径，保留历史当前版本与历史校验规则。"""
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile',type=Path,required=True)
    parser.add_argument('--materials',type=Path,required=True)
    args=parser.parse_args()
    print(json.dumps(validate(args.profile,args.materials)))


if __name__=='__main__':main()
