"""使用实际小文件验证来源锁和 finalize 的内容校验边界。"""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
from q2_source_identity import load_expert_source, model_geometry, verify_expert_file


class ExpertIdentity(unittest.TestCase):
    def test_same_size_corruption_cannot_refresh_a_download_receipt(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            expert = root / 'expert.gguf'
            original = b'verified source' * 512
            expert.write_bytes(original)
            identity = {'repo': 'example/model', 'revision': 'a' * 40, 'filename': expert.name,
                        'size': len(original), 'sha256': hashlib.sha256(original).hexdigest()}
            lock = root / 'lock.json'
            lock.write_text(json.dumps({'direct_sources': {'experts': {
                **identity, 'bytes': identity['size']}}}), encoding='utf-8')
            self.assertEqual(load_expert_source(lock), identity)
            self.assertEqual(verify_expert_file(expert, identity).st_size, len(original))
            config = {'model_type': 'qwen4_exp', 'num_hidden_layers': 1, 'num_experts': 2,
                      'hidden_size': 64, 'moe_intermediate_size': 64}
            config['freetoken_expert_source'] = {
                **{key: value for key, value in identity.items() if key != 'filename'},
                **model_geometry(config), 'path': str(expert), 'verified_mtime_ns': None}
            (root / 'expert-source.json').write_text(json.dumps(identity), encoding='utf-8')
            checkpoint = root / 'checkpoint'
            checkpoint.mkdir()
            config_path = checkpoint / 'config.json'
            config_path.write_text(json.dumps(config), encoding='utf-8')
            before = config_path.read_bytes()
            expert.write_bytes(b'X' + original[1:])
            result = subprocess.run([sys.executable, '-B', str(ROOT / 'tools/prepare_q2_checkpoint.py'),
                                     '--trunk', str(root), '--experts', str(root), '--output', str(checkpoint),
                                     '--source-lock', str(lock), '--finalize'],
                                    capture_output=True, text=True, encoding='utf-8')
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('SHA256 mismatch', result.stderr)
            self.assertEqual(config_path.read_bytes(), before)
            self.assertFalse((checkpoint / 'mixed-source.json').exists())

    def test_source_filename_cannot_escape_its_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'lock.json'
            path.write_text(json.dumps({'direct_sources': {'experts': {
                'filename': '../outside.gguf'}}}), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'within the download directory'):
                load_expert_source(path)

    def test_nested_text_config_uses_the_outer_architecture(self):
        config = {'model_type': 'qwen4_exp', 'text_config': {
            'model_type': 'qwen4_exp_text', 'num_hidden_layers': 2, 'num_experts': 3,
            'hidden_size': 64, 'moe_intermediate_size': 128}}
        self.assertEqual(model_geometry(config)['expert_bytes'], 41472)


if __name__ == '__main__':
    unittest.main()
