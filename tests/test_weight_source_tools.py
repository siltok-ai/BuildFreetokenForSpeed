"""实际文件损坏和dry-run副作用边界，不需要网络或GPU。"""
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

SCRIPT=Path(__file__).resolve().parents[1]/'model-weights/qwen3.8-flash-next-q2/download_sources.py'
spec=importlib.util.spec_from_file_location('weight_source_download',SCRIPT)
download=importlib.util.module_from_spec(spec)
spec.loader.exec_module(download)


class WeightSourceBoundaries(unittest.TestCase):
    def test_same_size_corruption_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'payload.bin'
            valid=b'quantized-original'
            path.write_bytes(valid)
            entry=dict(bytes=len(valid),sha256=hashlib.sha256(valid).hexdigest())
            download.verify(path,entry)
            path.write_bytes(b'x'+valid[1:])
            with self.assertRaisesRegex(ValueError,'SHA256 mismatch'):
                download.verify(path,entry)

    def test_truncation_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'payload.bin'
            path.write_bytes(b'abc')
            with self.assertRaisesRegex(ValueError,'size mismatch'):
                download.verify(path,dict(bytes=4,sha256=hashlib.sha256(b'abcd').hexdigest()))

    def test_dry_run_never_creates_destination(self):
        with tempfile.TemporaryDirectory() as directory:
            destination=Path(directory)/'not-created'
            result=subprocess.run([sys.executable,'-B',str(SCRIPT),'--output',str(destination),'--dry-run'],
                                  check=True,capture_output=True,text=True,encoding='utf-8')
            plan=json.loads(result.stdout)
            self.assertFalse(destination.exists())
            self.assertFalse(any(name.startswith('layer-') for name in plan['hf_files']))
            self.assertEqual(plan['expert_file'].rsplit('/',1)[-1],
                             'Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf')


if __name__=='__main__':
    unittest.main()
