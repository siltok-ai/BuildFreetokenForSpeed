"""Record the installed Windows dependency closure without borrowing another venv."""
import importlib.metadata as md
from pathlib import Path
import tomllib
import argparse
import json
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True, help='new candidate lock; never overwrite the accepted lock')
    parser.add_argument('--profile', type=Path, default=Path(__file__).resolve().parent / 'profiles/win1.json')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('candidate lock output must be new')
    root = Path(__file__).resolve().parents[1]
    profile = json.loads(args.profile.read_text(encoding='utf-8'))
    import zipfile
    with zipfile.ZipFile(root / profile['source_archive']) as archive:
        project = tomllib.loads(archive.read('pyproject.toml').decode())
    pending = list(project['project']['dependencies']) + list(project['build-system']['requires'])
    pending += ['triton-windows==3.6.0.post26', 'ninja', 'packaging']
    seen = {}
    while pending:
        req = Requirement(pending.pop())
        if req.marker and not any(req.marker.evaluate({'extra': extra}) for extra in ('', *req.extras)):
            continue
        name = canonicalize_name(req.name)
        dist = md.distribution(name)
        if not req.specifier.contains(dist.version, prereleases=True):
            raise RuntimeError(f'{req} does not accept installed {dist.version}')
        contexts = {'', *req.extras}
        previous = seen.setdefault(name, {'version': dist.version, 'extras': set()})
        contexts -= previous['extras']
        if not contexts:
            continue
        previous['extras'].update(contexts)
        for dependency in dist.requires or []:
            child = Requirement(dependency)
            if not child.marker or any(child.marker.evaluate({'extra': extra}) for extra in contexts):
                child.marker = None
                pending.append(str(child))
    lines = ['# CPython 3.12 Windows x64; Torch wheels from the official cu130 index.']
    lines += [f"{name}=={seen[name]['version']}" for name in sorted(seen)]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(f'{len(seen)} pinned distributions')

if __name__ == '__main__':
    main()
