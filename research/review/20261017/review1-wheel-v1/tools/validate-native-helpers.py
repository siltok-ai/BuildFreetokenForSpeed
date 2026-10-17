"""实际编译两种扩展，验证重构后的共同编译/回执路径；产物不进入发布材料。"""
from pathlib import Path
import json
import sys

BUILD = Path(r'C:\workspace\wt\freetoken-mats\flash-perf')
SOURCE = Path(r'C:\workspace\build\freetoken-mats\review1-materials-01\candidate-source\build-source')
OUTPUT = Path(__file__).parent/'native-helper-validation'
sys.path.insert(0, str(BUILD/'build'))
from derive_python_materials import rebuild_row_store, rebuild_pinned_tensor

OUTPUT.mkdir()
results = []
for function in (rebuild_row_store, rebuild_pinned_tensor):
    binary, receipt = function(SOURCE, OUTPUT, '2.11.0+cu130')
    results.append(dict(artifact=str(binary), **receipt))
    (OUTPUT/'results.json').write_text(json.dumps(results, indent=2)+'\n')
    print('Compiled and receipt verified:', receipt['module'], flush=True)
