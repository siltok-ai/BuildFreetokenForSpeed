"""Rebuild the current wheel and exercise native reuse boundaries with real files."""
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
sys.path.insert(0, str(ROOT/'build'))
from derive_python_materials import derive, validate_python_only_changes


class PythonMaterials(unittest.TestCase):
    def test_current_wheel_replays_and_repackages_without_changing_native_bytes(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)/'derived'
            derive(ROOT/'build/profiles/strataq2portable1.json',
                   ROOT/'build/profiles/strataq2perf9.json',
                   ROOT/'artifacts/strataq2perf9-cp312-cu130-sm120', output)
            record = json.loads((output/'build-record.json').read_text(encoding='utf-8'))
            base = json.loads((ROOT/'artifacts/strataq2perf9-cp312-cu130-sm120/build-record.json')
                              .read_text(encoding='utf-8'))
            self.assertEqual(record['status'], 'derived_requires_runtime_validation')
            self.assertNotIn('validation_summary', record)
            self.assertEqual({n:r for n,r in record['artifacts'].items() if n.endswith('.dll')},
                             {n:r for n,r in base['artifacts'].items() if n.endswith('.dll')})
            with zipfile.ZipFile(output/'wheels'/record['wheel_filename']) as wheel:
                dist = 'freetoken-'+record['custom_version']+'.dist-info'
                rows = list(csv.reader(io.StringIO(wheel.read(dist+'/RECORD').decode())))
                self.assertEqual({r[0] for r in rows}, set(wheel.namelist()))
                for name, encoded, size in rows:
                    if name.endswith('/RECORD'):
                        continue
                    data = wheel.read(name)
                    digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b'=').decode()
                    self.assertEqual(encoded, 'sha256='+digest)
                    self.assertEqual(int(size), len(data))
                    if name.startswith('freetoken/') and name.endswith('.py'):
                        self.assertEqual(data, (output/'candidate-source/build-source/python'/name).read_bytes())
                asset = 'freetoken/moe/configs/q2_residency/strata_flash_next.json'
                self.assertEqual(len(json.loads(wheel.read(asset))['ranked']), 48*512)
                self.assertEqual({name:hashlib.sha256(wheel.read(name)).hexdigest()
                                  for name in record['native_modules']},
                                 record['binary_reuse']['native_sha256'])
            changed = validate_python_only_changes(output/'base-source/build-source',
                                                    output/'candidate-source/build-source')
            self.assertEqual(sorted(changed), record['binary_reuse']['changed_source_files'])

    def test_native_input_changes_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            before, after = Path(temporary)/'before', Path(temporary)/'after'
            for tree, text in ((before, 'original'), (after, 'changed')):
                path = tree/'python/freetoken/kernel/csrc/example.cpp'
                path.parent.mkdir(parents=True)
                path.write_text(text)
            with self.assertRaisesRegex(ValueError, 'binary reuse forbidden'):
                validate_python_only_changes(before, after)

    def test_unlisted_package_data_cannot_bypass_native_reuse_guard(self):
        with tempfile.TemporaryDirectory() as temporary:
            before, after = Path(temporary)/'before', Path(temporary)/'after'
            before.mkdir()
            asset = after/'python/freetoken/moe/configs/unknown.json'
            asset.parent.mkdir(parents=True)
            asset.write_text('{}')
            with self.assertRaisesRegex(ValueError, 'binary reuse forbidden'):
                validate_python_only_changes(before, after)

    def test_existing_artifacts_are_preserved(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            marker = root/'accepted-wheel.whl'
            marker.write_bytes(b'accepted')
            with self.assertRaises(FileExistsError):
                derive(None, None, None, root)
            self.assertEqual(marker.read_bytes(), b'accepted')


if __name__ == '__main__':
    unittest.main()
