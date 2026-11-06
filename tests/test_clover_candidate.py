"""Clover 候选重放及 tokenizer 的 CPU 回归。"""
import json
import hashlib
from pathlib import Path
import shutil
import struct
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'tools'))
from clover_bundle import export_tokenizer, verify_bundle, prepare
from verify_clover_candidate import verify
from test_flash_next_gguf import metadata_fixture


def write_clover_metadata_fixture(path):
    """写入真实GGUF元数据和BPE字典，不伪造可供GPU推理的张量。"""
    meta = metadata_fixture()
    meta['general.architecture'] = 'clover_moe'
    meta['tokenizer.ggml.tokens'].append('<|mask|>')
    meta['tokenizer.ggml.token_type'].append(3)
    meta['tokenizer.ggml.mask_token_id'] = 260
    meta['tokenizer.ggml.tokens'].append('ab')
    meta['tokenizer.ggml.token_type'].append(1)
    meta['tokenizer.ggml.merges'] = ['a b']
    for key in ('embedding_length', 'expert_feed_forward_length',
                'expert_shared_feed_forward_length', 'attention.key_length', 'ssm.inner_size'):
        meta['clover_moe.'+key] = 2
    for key in ('block_count', 'attention.head_count', 'attention.head_count_kv',
                'expert_count', 'expert_used_count', 'ssm.group_count',
                'ssm.time_step_rank', 'ssm.state_size', 'ssm.conv_kernel'):
        meta['clover_moe.'+key] = 1
    meta['clover_moe.context_length'] = 4096
    meta['clover_moe.attention.recurrent_layers'] = [True]

    def string(value):
        data = value.encode('utf-8')
        return struct.pack('<Q', len(data))+data

    def value(item):
        if isinstance(item, str):
            return 8, string(item)
        if isinstance(item, bool):
            return 7, struct.pack('<?', item)
        if isinstance(item, int):
            return 4, struct.pack('<I', item)
        if isinstance(item, float):
            return 6, struct.pack('<f', item)
        encoded = [value(child) for child in item]
        subtype = encoded[0][0] if encoded else 8
        return 9, struct.pack('<IQ', subtype, len(encoded))+b''.join(data for _,data in encoded)

    rows = []
    for key, item in meta.items():
        kind, data = value(item)
        rows.append(string(key)+struct.pack('<I', kind)+data)
    path.write_bytes(struct.pack('<4sIQQ', b'GGUF', 3, 0, len(rows))+b''.join(rows))


class CloverCandidateTests(unittest.TestCase):
    def test_generated_bundle_retains_model_notice_and_transfer_inventory(self):
        """实际组包后核对模型署名及清单，防止误带不含权重的仓库NOTICE。"""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            gguf = root/'fixture.gguf'
            write_clover_metadata_fixture(gguf)
            lock = root/'source-lock.json'
            lock.write_text(json.dumps(dict(filename=gguf.name, size=gguf.stat().st_size,
                sha256=hashlib.sha256(gguf.read_bytes()).hexdigest())), encoding='utf-8')
            output = root/'resource bundle'
            prepare(output, gguf, lock)
            self.assertEqual((output/'NOTICE').read_bytes(),
                             (ROOT/'model-weights/clover-q8/NOTICE').read_bytes())
            self.assertNotIn('Model weights are not included',
                             (output/'NOTICE').read_text(encoding='utf-8'))
            for name in ('LICENSE', 'ATTRIBUTIONS.md'):
                self.assertEqual((output/name).read_bytes(),
                                 (ROOT/'model-weights/clover-q8'/name).read_bytes())
            self.assertEqual((output/gguf.name).read_bytes(), gguf.read_bytes())
            self.assertEqual(verify_bundle(output)['status'], 'transfer_verified')

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
