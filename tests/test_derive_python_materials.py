"""实际重放源码和封装新 wheel，验证二进制复用的边界及 RECORD 完整性。"""
import base64
import csv
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'build'))
from derive_python_materials import derive, validate_python_only_changes


class PythonMaterials(unittest.TestCase):
    def test_native_input_changes_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            before, after = root / 'before', root / 'after'
            for tree in (before, after):
                file = tree / 'python/freetoken/kernel/csrc/example.cpp'
                file.parent.mkdir(parents=True)
                file.write_bytes(b'original native source\n')
            (after / 'python/freetoken/kernel/csrc/example.cpp').write_bytes(b'changed native source\n')
            with self.assertRaisesRegex(ValueError, 'binary reuse forbidden'):
                validate_python_only_changes(before, after)

    def test_existing_artifacts_are_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            marker = root / 'accepted-wheel.whl'
            marker.write_bytes(b'accepted bytes')
            with self.assertRaises(FileExistsError):
                derive(None, None, None, root)
            self.assertEqual(marker.read_bytes(), b'accepted bytes')

    def test_real_profile_replays_and_repackages_without_changing_native_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'derived'
            derive(ROOT / 'build/profiles/q2opt1.json', ROOT / 'build/profiles/q2exp1.json',
                   ROOT / 'artifacts/q2exp1-cp312-cu130-sm120', output)
            record = json.loads((output / 'build-record.json').read_text(encoding='utf-8'))
            with zipfile.ZipFile(output / 'wheels' / record['wheel_filename']) as archive:
                dist = f"freetoken-{record['custom_version']}.dist-info"
                rows = list(csv.reader(io.StringIO(archive.read(dist + '/RECORD').decode())))
                self.assertEqual({r[0] for r in rows}, set(archive.namelist()))
                for name, encoded, size in rows:
                    if name.endswith('/RECORD'):
                        continue
                    data = archive.read(name)
                    digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b'=').decode()
                    self.assertEqual(encoded, 'sha256=' + digest)
                    self.assertEqual(int(size), len(data))
                    if name.startswith('freetoken/') and name.endswith('.py'):
                        self.assertEqual(data, (output / 'candidate-source/build-source/python' / name).read_bytes())
                native = {name: hashlib.sha256(archive.read(name)).hexdigest() for name in record['native_modules']}
                self.assertEqual(native, record['binary_reuse']['native_sha256'])


if __name__ == '__main__':
    unittest.main()
