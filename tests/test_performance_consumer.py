"""性能合同必须依据完整流式时序，不能接受自报速度或截断输出。"""
import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from verify_performance_consumer import installation, performance


class PerformanceConsumerTests(unittest.TestCase):
    def setUp(self):
        """构造每 10 毫秒交付一个 token 的已知流，覆盖两种输入长度。"""
        self.report = dict(status='measured', worker_pids_gone=True, runs=[], contexts={})
        for label, tokens in [('70-0', 70), ('diverse-4096-0', 4096)]:
            self.report['contexts'][label] = dict(median_tps=100.0)
            for repeat in range(3):
                self.report['runs'].append(dict(label=f'{label}-{repeat}',
                    usage=dict(prompt_tokens=tokens, completion_tokens=512), output_tokens=512,
                    content_chunk_count=512, arrival_offsets=[i / 100 for i in range(512)],
                    done_events=1, finish_reason='length', decode_tps=100.0))

    def test_actual_stream_timing(self):
        self.assertEqual(performance(self.report), {'70-0': 100.0, 'diverse-4096-0': 100.0})

    def test_rejects_fabricated_statistics_and_broken_stream(self):
        for fault in ('reported_rate', 'missing_token', 'backwards', 'missing_done', 'live_worker'):
            with self.subTest(fault=fault):
                report = copy.deepcopy(self.report)
                row = report['runs'][0]
                if fault == 'reported_rate':
                    row['decode_tps'] = 120
                elif fault == 'missing_token':
                    row['arrival_offsets'].pop()
                elif fault == 'backwards':
                    row['arrival_offsets'][10] = -1
                elif fault == 'missing_done':
                    row['done_events'] = 0
                else:
                    report['worker_pids_gone'] = False
                with self.assertRaises(ValueError):
                    performance(report)

    def test_installer_ownership_rejects_staging_and_existing_data(self):
        before = dict(install_path=r'C:\workspace\build\siltok-speed\i06',
                      install_present=False, data_present=False, service=None,
                      event_source_present=False, uninstall_keys=[])
        owned = dict(before, phase='installed', install_present=True, data_present=True,
                     service=dict(path=r'"C:\workspace\build\siltok-speed\i06\SiltokSpeed.exe" --service'),
                     bundle_sha256='1234')
        result = dict(status='installed_service_stopped', launcher_admin=True,
                      installer_exitcode=0, install_path=before['install_path'])
        self.assertEqual(installation(before, owned, result, '1234').name, 'i06')
        for fault in ('existing_data', 'wrong_service', 'staging', 'failed_install', 'bundle'):
            with self.subTest(fault=fault):
                baseline, ownership, completed = copy.deepcopy((before, owned, result))
                if fault == 'existing_data': baseline['data_present'] = True
                elif fault == 'wrong_service': ownership['service']['path'] = r'C:\other\SiltokSpeed.exe'
                elif fault == 'staging': ownership['install_path'] = r'C:\workspace\build\siltok-speed\p03\app'
                elif fault == 'failed_install': completed['installer_exitcode'] = 2
                else: ownership['bundle_sha256'] = 'changed'
                with self.assertRaises(ValueError):
                    installation(baseline, ownership, completed, '1234')


if __name__ == '__main__':
    unittest.main()
