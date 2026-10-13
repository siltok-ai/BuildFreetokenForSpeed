"""Add a distinct replay validator while leaving dev6's strict scope intact."""
from pathlib import Path

ROOT = Path('C:/workspace/worktrees/build-freetoken-flash-next-strata-q2')
source = (ROOT/'tools/validate_protected_revision.py').read_text(encoding='utf-8')
start = source.index('    expected = {')
end = source.index('    with tempfile.', start)
source = source[:start] + "    expected = {'python/freetoken/version.py', 'python/freetoken/moe/q2_residency.py', 'tests/moe/test_q2_residency.py'}\n" + source[end:]
source = source.replace("ROOT / 'build/profiles/strataq2dev5.json'", "ROOT / 'build/profiles/strataq2dev6.json'")
source = source.replace("default=ROOT / 'build/profiles/strataq2dev6.json'", "default=ROOT / 'build/profiles/strataq2dev7.json'")
source = source.replace('核对保护驻留补丁、排名数据和原生复用', '核对短 prefill 路由边界修订、排名数据和原生复用')
target = ROOT/'tools/validate_sparse_residency_revision.py'
if target.exists():
    raise FileExistsError(target)
target.write_text(source, encoding='utf-8')
path = ROOT/'tools/validate_repository.py'
text = path.read_text(encoding='utf-8')
anchor = '    print(json.dumps({"official_files":'
insert = '''    research_dev7 = None
    dev7_profile = ROOT/'build/profiles/strataq2dev7.json'
    if dev7_profile.exists():
        from validate_sparse_residency_revision import validate as validate_sparse_revision
        dev7_materials = ROOT/'artifacts/strataq2dev7-cp312-cu130-sm120'
        research_dev7 = validate_sparse_revision(dev7_profile,dev7_materials if dev7_materials.exists() else None)
'''
if anchor not in text:
    raise RuntimeError('Missing validator insertion point')
text = text.replace(anchor, insert+anchor)
text = text.replace('"strata_research_dev6":research_dev6, "verified": True',
    '"strata_research_dev6":research_dev6, "strata_research_dev7":research_dev7, "verified": True')
path.write_text(text, encoding='utf-8')
print(target)
