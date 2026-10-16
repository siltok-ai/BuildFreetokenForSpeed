"""用真实冻结证据验证交付合同的篡改拒绝边界，不启动模型。"""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from current_release import ROOT
from verify_flash_next_5070ti_consumer import verify


class InstalledConsumerTests(unittest.TestCase):
    def test_rejects_changed_contract_bindings(self):
        """真正执行冻结报告核对，拒绝身份、封存摘要和已知风险被改写。"""
        # 历史实装证据必须继续受防篡改回归保护，不随默认版本切换而跳过。
        contract = json.loads((ROOT / 'integration/speed/consumer-contract-5070ti-integration.json').read_text(encoding='utf-8'))
        changes = (
            ('index digest', lambda item: item['evidence'].update(sha256='0' * 64)),
            ('source identity', lambda item: item['speed_adapter'].update(source_commit='0' * 40)),
            ('hidden issue', lambda item: item.update(known_issues=[])),
            ('installer identity', lambda item: item['speed_adapter']['package']['installer'].update(sha256='0' * 64)),
            ('path escape', lambda item: item['evidence'].update(index='../unowned-evidence.json')),
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'contract.json'
            for label, change in changes:
                with self.subTest(label=label):
                    modified = copy.deepcopy(contract)
                    change(modified)
                    path.write_text(json.dumps(modified), encoding='utf-8')
                    with self.assertRaises((ValueError, AssertionError)):
                        verify(path)


if __name__ == '__main__':
    unittest.main()
