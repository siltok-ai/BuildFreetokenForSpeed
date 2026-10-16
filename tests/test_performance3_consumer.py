"""新源码消费检查点必须核对真实原生输出，拒绝抄写通过标记。"""
import copy
import json
from pathlib import Path
import sys
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
from verify_performance3_consumer import quality


class SourceQualityTests(unittest.TestCase):
    def setUp(self):
        """使用已冻结普通wheel的23题完整输出，构造对应的IPC报告形态。"""
        self.native=json.loads((ROOT/'research/performance/flash-next-5070ti-20261016/perf3-native/runs/candidate3-wheel-quality-on-v1.json').read_text(encoding='utf-8'))
        self.report=dict(status='passed',worker_pids_gone=True,cancel=dict(worker_reused=True),
                         score=15,thinking_score=3,regressions=[],runs=[])
        for row in self.native['runs']:
            if 'quality_pass' in row:
                self.report['runs'].append(dict(name=row['name'],quality_pass=row['quality_pass'],
                    final=dict(tokens=list(row['output']['token_ids']))))

    def test_real_native_cases_can_be_cross_checked(self):
        quality(self.report,self.native)

    def test_claimed_pass_cannot_hide_output_case_score_or_lifecycle_changes(self):
        for fault in ('missing','ids','quality','score','lifecycle'):
            with self.subTest(fault=fault):
                report=copy.deepcopy(self.report)
                if fault=='missing':report['runs'].pop()
                elif fault=='ids':report['runs'][0]['final']['tokens'][0]+=1
                elif fault=='quality':report['runs'][0]['quality_pass']=not report['runs'][0]['quality_pass']
                elif fault=='score':report['score']=20
                else:report['cancel']['worker_reused']=False
                with self.assertRaises(ValueError):quality(report,self.native)


if __name__=='__main__':unittest.main()
