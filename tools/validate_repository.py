"""Validate the upstream snapshot and the current FreeToken source/materials."""
import json
import tempfile
from pathlib import Path

from prepare_source import ROOT, extract_archive, repository_path, sha256
from verify_current_release import verify


def tree_hashes(root):
    return {path.relative_to(root).as_posix(): sha256(path)
            for path in root.rglob('*') if path.is_file()}


def main():
    lock = json.loads((ROOT/'upstream/lock.json').read_text(encoding='utf-8'))
    with tempfile.TemporaryDirectory(prefix='freetoken-upstream-') as temporary:
        extracted = Path(temporary)/'source'
        extract_archive(repository_path(lock['archive']), extracted)
        snapshot = ROOT/'upstream/FreeToken'
        if tree_hashes(extracted) != tree_hashes(snapshot):
            raise ValueError('Upstream browsing snapshot differs from the source archive')
    result = verify()
    print(json.dumps(dict(status='verified', upstream_snapshot=True, current=result,
                          runtime_executed=False)))


if __name__ == '__main__':
    main()
