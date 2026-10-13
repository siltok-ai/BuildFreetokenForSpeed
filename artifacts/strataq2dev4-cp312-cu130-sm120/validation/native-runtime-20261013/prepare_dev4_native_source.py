"""Create licensed, source-pinned native MMVQ inputs in a fresh FT candidate."""
import hashlib
import json
import subprocess
from pathlib import Path

ROOT=Path(__file__).resolve().parent
SOURCE=ROOT/'candidate-source-dev4-working/build-source'
REFERENCE=Path(r'C:\workspace\references\strata-flash-next-research')
COMMIT='6674a0065fb96bacde33e3eb10f91a1df86f95f2'

def blob(name):
    return subprocess.check_output(['git','show',COMMIT+':'+name],cwd=REFERENCE)

def main():
    vendor=SOURCE/'python/freetoken/kernel/csrc/third_party/strata_mmvq'
    vendor.mkdir(parents=True,exist_ok=False)
    names=['src/kernels/cuda/native_mmvq.cu','src/kernels/cuda/s26_tsum.cuh',
        'src/kernels/cuda/q8_1_il.cuh','include/strata/kernels/native_mmvq.hpp',
        'include/strata/kernels/dp4a.hpp','include/strata/kernels/q8_1_finite.hpp',
        'include/strata/kernels/iq_kernels.hpp','include/strata/kernels/pdl.hpp']
    entries=[]
    for name in names:
        original=blob(name)
        text=original.decode('utf-8').replace('strata::kernels','freetoken::gguf_native')
        text=text.replace('STRATA_','FREETOKEN_GGUF_')
        if name.endswith('/native_mmvq.cu'):
            old='void validate_stream(void* stream) {\n    if (!stream) throw std::invalid_argument("native MMVQ requires an explicit non-null CUDA stream");\n}'
            if old not in text:raise RuntimeError('Pinned CUDA stream adapter changed')
            text=text.replace(old,'void validate_stream(void* stream) {\n    // FreeToken passes its current CUDA stream explicitly; CUDA also permits handle 0.\n    (void) stream;\n}')
        if name.endswith('/native_mmvq.hpp'):
            text=text.replace('explicit non-null CUDA stream','explicit current CUDA stream')
        destination=vendor/name;destination.parent.mkdir(parents=True,exist_ok=True)
        destination.write_text(text,encoding='utf-8',newline='\n')
        entries.append(dict(upstream_path=name,upstream_blob_sha256=hashlib.sha256(original).hexdigest(),
            ported_sha256=hashlib.sha256(destination.read_bytes()).hexdigest()))
    for original,name in [('LICENSE','LICENSE-strata.txt'),('third_party/ggml/LICENSE','LICENSE-ggml.txt')]:
        original_bytes=blob(original);(vendor/name).write_bytes(original_bytes)
        entries.append(dict(upstream_path=original,upstream_blob_sha256=hashlib.sha256(original_bytes).hexdigest(),
            ported_path=name,ported_sha256=hashlib.sha256(original_bytes).hexdigest()))
    provenance=dict(upstream='https://github.com/Niko1221/Strata',commit=COMMIT,
        ggml_commit='3cf03257f219afbe7334045ff7c6a06ac68c627d',files=entries,
        transformations=['C++ namespace strata::kernels -> freetoken::gguf_native',
            'STRATA_ macro/environment names -> FREETOKEN_GGUF_',
            'Allow explicit CUDA default-stream handle 0; CUDA launch errors remain checked'],
        external_runtime_dependency=False,license='MIT',
        scope='GGUF GPU Q8_1 quantizer and MMVQ only. External IQ dispatcher is unsupported and rejected; PDL disabled for FreeToken graph ordering.')
    (vendor/'UPSTREAM.json').write_text(json.dumps(provenance,indent=2)+'\n',encoding='utf-8')
    pyproject=SOURCE/'pyproject.toml'
    text=pyproject.read_text(encoding='utf-8')
    text=text.replace('license-files = ["LICENSE"]',
        'license-files = ["LICENSE", "python/freetoken/kernel/csrc/third_party/strata_mmvq/LICENSE-*.txt"]')
    pyproject.write_text(text,encoding='utf-8')
    version=SOURCE/'python/freetoken/version.py'
    version.write_text(version.read_text(encoding='utf-8').replace('0.1.3+speed.strataq2dev3','0.1.3+speed.strataq2dev4'),encoding='utf-8')
    print(json.dumps(dict(vendor=str(vendor),files=len(entries),version='0.1.3+speed.strataq2dev4')))

if __name__=='__main__':main()
