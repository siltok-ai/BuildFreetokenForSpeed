"""当前指针不能把未知验收类型当成已验证交付。"""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'tools'))
from current_release import load_current
from verify_current_release import verify


class CurrentAcceptanceTests(unittest.TestCase):
    def test_unknown_consumer_kind_cannot_claim_verified_current_release(self):
        """使用真实profile和实际wheel/AOT，仅替换消费合同类型，仍须明确拒绝。"""
        current = load_current()
        contract = json.loads((ROOT/current['consumer_contract']).read_text(encoding='utf-8'))
        contract['kind'] = 'unrecognized_acceptance'
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/'consumer.json'
            path.write_text(json.dumps(contract), encoding='utf-8')
            selection = dict(current, consumer_contract=str(path))
            with patch('verify_current_release.load_current', return_value=selection):
                with self.assertRaisesRegex(ValueError, 'Unsupported current consumer kind'):
                    verify()


if __name__ == '__main__':
    unittest.main()
