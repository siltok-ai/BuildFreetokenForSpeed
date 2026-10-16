"""冻结第三轮源码；旧profile/patch/wheel均不改写，复用perf2的完整27项原生输入。"""
from pathlib import Path
import difflib
import hashlib
import json

RUN=Path(__file__).parent
ROOT=Path(r'C:\workspace\wt\freetoken-mats\flash-perf')
BASE=Path(r'C:\workspace\build\freetoken-mats\flash-perf-materials-02\candidate-source\build-source')
SOURCE=Path(r'C:\workspace\build\freetoken-mats\flash-perf-candidate3\build-source')


def sha(path):
    """绑定实际字节。"""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def content(root,name):
    """新增文件使用/dev/null补丁，不假设它已存在。"""
    path=root/name
    return path.read_text() if path.exists() else ''


def without_adaptation(name,text):
    """先生成仅含事务/预算的中间态，让后续热点交换补丁有清晰独立差异。"""
    if name.endswith('engine/config.py'):
        text=text.replace('    # Online owner exchange is opt-in; zero retains the frozen initial residency.\n'
                          '    moe_q2_adapt_every: int = 0\n','')
        text=text.replace("        if self.moe_q2_adapt_every < 0 or (self.moe_q2_adapt_every and not self.moe_q2_resident_experts):\n"
                          "            raise ValueError('Online Q2 adaptation requires protected owners and a nonnegative interval')\n",'')
    if name.endswith('engine/engine.py'):
        text=text.replace('        self.q2_adaptive_owners = None\n','')
        for prefix in ('config','self.config'):
            text=text.replace(f'        if {prefix}.moe_q2_adapt_every:\n'
                              '            self.q2_adaptive_owners = self.moe_offload_cache.start_adaptation()\n','')
        text=text.replace("**({'ownership': ownership, 'decode_overlap': config.moe_q2_decode_overlap,\n"
                          "                'adapt_every': config.moe_q2_adapt_every}",
                          "**({'ownership': ownership, 'decode_overlap': config.moe_q2_decode_overlap}")
        text=text.replace('        if batch.is_decode and self.q2_adaptive_owners is not None:\n'
                          '            self.q2_adaptive_owners.step()\n','')
    if name.endswith('models/qwen4_exp/mtp_runtime.py'):
        text=text.replace('        if self.engine.q2_adaptive_owners is not None:\n'
                          '            self.engine.q2_adaptive_owners.step()\n','')
    if 'moe_q2_adapt_every' in text:raise ValueError('Unsplit adaptive binding: '+name)
    return text


def main():
    """只封存组件和整模型质量/重建已通过的源码；不会继承perf2安装验收状态。"""
    component=json.loads((RUN/'candidate3-source-components-v2.resource.json').read_text())
    quality=json.loads((RUN/'candidate3-source-quality-on-v3.json').read_text())
    assert component['returncode']==0 and quality['status']=='passed'
    assert quality['score']>=15 and quality['thinking_score']==3 and len(quality['rebuild_steps'])==2
    version='0.1.3+speed.strataq2perf3'
    (SOURCE/'python/freetoken/version.py').write_text(f'__version__ = "{version}"\n',encoding='utf-8',newline='\n')
    shared=['python/freetoken/engine/config.py','python/freetoken/engine/engine.py',
            'python/freetoken/models/qwen4_exp/mtp_runtime.py']
    intermediate={name:without_adaptation(name,content(SOURCE,name)) for name in shared}
    groups=[
        ('0028','FTPERF-008','Compact MTP recurrence and measured linear-state cache budgets','performance',[
            *shared,'python/freetoken/attention/linear.py','python/freetoken/kvcache/linear_state_pool.py',
            'python/freetoken/kernel/triton/gguf_gdn.py','python/freetoken/kernel/triton/gguf_gdn_compact.py',
            'tests/models/qwen4_exp/test_mtp_compact.py']),
        ('0029','FTPERF-009','Exchange GPU and compact host Q2 owners online at idle boundaries','performance',[
            *shared,'python/freetoken/moe/q2_residency.py','python/freetoken/moe/q2_adaptive.py',
            'tests/moe/test_q2_adaptive.py']),
        ('0030','FTPERF-010','Widen Q2 vector tiles with FP32 activations and Q8 arithmetic','performance',[
            'python/freetoken/moe/fused_q2_q8.py','tests/moe/test_q2_q8_tiles.py']),
        ('0031','FTPERF-011','Distinct immutable third performance wheel identity','version',[
            'python/freetoken/version.py']),
    ]
    series=json.loads((ROOT/'patches/strataq2perf2-series.json').read_text())
    profile=json.loads((ROOT/'build/profiles/strataq2perf2.json').read_text())
    for number,ident,title,kind,names in groups:
        patch=[]
        for name in names:
            old=intermediate[name] if number=='0029' and name in intermediate else content(BASE,name)
            new=intermediate[name] if number=='0028' and name in intermediate else content(SOURCE,name)
            if old==new:raise ValueError('Empty advertised source delta: '+name)
            patch.extend(difflib.unified_diff(old.splitlines(keepends=True),new.splitlines(keepends=True),
                fromfile='a/'+name if old else '/dev/null',tofile='b/'+name))
        path=ROOT/f'patches/series/strataq2perf3-{number}.patch'
        if path.exists():raise FileExistsError(path)
        path.write_bytes(''.join(patch).encode())
        relative=path.relative_to(ROOT).as_posix()
        series['patches'].append(dict(id=ident,path=relative,sha256=sha(path),kind=kind,title=title,
            status='active',upstream_status='local performance revision; reassess on upstream upgrade',files=names))
        profile['input_sha256'][relative]=sha(path)
    selected='patches/strataq2perf3-series.json'
    path=ROOT/selected
    if path.exists():raise FileExistsError(path)
    path.write_text(json.dumps(series,indent=2)+'\n',encoding='utf-8',newline='\n')
    profile['input_sha256'].pop(profile['patch_series'])
    profile.update(custom_version=version,wheel_filename=f'freetoken-{version}-cp312-cp312-win_amd64.whl',patch_series=selected)
    profile['input_sha256'][selected]=sha(path)
    path=ROOT/'build/profiles/strataq2perf3.json'
    if path.exists():raise FileExistsError(path)
    path.write_text(json.dumps(profile,indent=2)+'\n',encoding='utf-8',newline='\n')
    print(json.dumps(dict(version=version,patches=len(series['patches']),unchanged_spec=profile['spec_file'])))


if __name__=='__main__':main()
