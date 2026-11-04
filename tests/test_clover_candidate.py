"""Clover 候选重放及 tokenizer 的 CPU 回归。"""
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'tools'))
from clover_bundle import export_tokenizer, verify_bundle, prepare
from verify_clover_candidate import verify
from test_flash_next_gguf import metadata_fixture


class CloverCandidateTests(unittest.TestCase):
    def test_same_length_transfer_corruption_is_rejected(self):
        import hashlib
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'data').write_bytes(b'original')
            record = {'data': dict(size=8, sha256=hashlib.sha256(b'original').hexdigest())}
            (root/'files.json').write_text(json.dumps(record), encoding='utf-8')
            self.assertEqual(verify_bundle(root)['status'], 'transfer_verified')
            (root/'data').write_bytes(b'changed!')
            with self.assertRaisesRegex(ValueError, 'SHA256'):
                verify_bundle(root)

    def test_wrong_source_is_rejected_before_creating_package(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'input.gguf').write_bytes(b'not the official model')
            with self.assertRaisesRegex(ValueError, 'source size/SHA256'):
                prepare(root/'output', root/'input.gguf')
            self.assertFalse((root/'output').exists())

    def test_complete_candidate_replays_independently_of_accepted_release(self):
        result = verify()
        self.assertEqual(result['patches'], 66)
        self.assertFalse(result['materials_checked'])
        self.assertFalse(result['gpu_executed'])

    def test_exported_tokenizer_preserves_ids_and_defaults_after_move(self):
        from transformers import AutoTokenizer
        meta = metadata_fixture()
        meta['general.architecture'] = 'clover_moe'
        meta['clover_moe.context_length'] = 131072
        meta['tokenizer.ggml.tokens'].append('<|mask|>')
        meta['tokenizer.ggml.token_type'].append(3)
        meta['tokenizer.ggml.mask_token_id'] = 260
        meta['tokenizer.ggml.tokens'].append('ab')
        meta['tokenizer.ggml.token_type'].append(1)
        meta['tokenizer.ggml.merges'] = ['a b']
        with tempfile.TemporaryDirectory() as directory:
            before, after = Path(directory)/'before', Path(directory)/'moved directory'
            before.mkdir()
            export_tokenizer(before, meta)
            shutil.move(before, after)
            tokenizer = AutoTokenizer.from_pretrained(after, local_files_only=True)
            self.assertEqual(tokenizer.encode('<think><|mask|>', add_special_tokens=False), [258,260])
            text = '中文 café🙂 123\nA'
            self.assertEqual(tokenizer.decode(tokenizer.encode(text)), text)
            config = json.loads((after/'generation_config.json').read_text(encoding='utf-8'))
            self.assertFalse(config['do_sample'])
            self.assertEqual(config['eos_token_id'], [257,256])
            self.assertEqual(tokenizer.model_max_length, 131072)


if __name__ == '__main__':
    unittest.main()
