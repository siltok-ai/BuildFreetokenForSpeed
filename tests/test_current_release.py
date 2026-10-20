"""Exercise the public compatibility entry point without a GPU."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'tools'))
from current_release import load_current
from verify_current_release import verify


class CurrentAcceptanceTests(unittest.TestCase):
    def test_speed_cli_rejects_incompatible_runtime_before_source_replay(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            target = root/'silitok_speed/runtimes/freetoken/release.json'
            target.parent.mkdir(parents=True)
            target.write_text(json.dumps(dict(version='unsupported')), encoding='utf-8')
            result = subprocess.run([sys.executable, '-B', str(ROOT/'tools/verify_portable_consumer.py'),
                '--speed-root', str(root)], capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('Speed runtime contract differs: version', result.stderr)

    def test_unknown_consumer_kind_is_rejected(self):
        current = load_current()
        contract = json.loads((ROOT/current['consumer_contract']).read_text(encoding='utf-8'))
        contract['kind'] = 'unknown'
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/'consumer.json'
            path.write_text(json.dumps(contract), encoding='utf-8')
            with patch('verify_current_release.load_current',
                       return_value=dict(current, consumer_contract=str(path))):
                with self.assertRaisesRegex(ValueError, 'Unsupported current consumer kind'):
                    verify()


if __name__ == '__main__':
    unittest.main()
