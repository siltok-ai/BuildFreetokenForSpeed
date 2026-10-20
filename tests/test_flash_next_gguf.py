"""验证 GGUF tokenizer 导出和可迁移运行输入，不依赖历史 HF 元数据。"""
from pathlib import Path
import json
import shutil
import struct
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'tools'))
from flash_next_gguf import export_metadata, read_metadata


def metadata_fixture():
    """独立构造 byte-level BPE 字典，确保 Unicode 和 special token 处理真实执行。"""
    visible = list(range(33, 127))+list(range(161, 173))+list(range(174, 256))
    extra = 0
    tokens = []
    for byte in range(256):
        if byte in visible:
            tokens.append(chr(byte))
        else:
            tokens.append(chr(256+extra))
            extra += 1
    specials = ['<|endoftext|>', '<|im_end|>', '<think>', '</think>']
    return {'general.architecture': 'qwen4exp', 'tokenizer.ggml.model': 'gpt2',
        'tokenizer.ggml.pre': 'qwen35', 'tokenizer.ggml.tokens': tokens+specials,
        'tokenizer.ggml.token_type': [1]*256+[3,3,4,4], 'tokenizer.ggml.merges': [],
        'tokenizer.ggml.eos_token_id': 257, 'tokenizer.ggml.padding_token_id': 256,
        'tokenizer.ggml.bos_token_id': 256, 'qwen4exp.ple.eos_token_id': 256,
        'tokenizer.chat_template': '{{ messages[0].content }}{% if enable_thinking %}<think>{% else %}<think></think>{% endif %}',
        'general.sampling.temp': 1., 'general.sampling.top_k': 20, 'general.sampling.top_p': .95}


class EmbeddedTokenizerTests(unittest.TestCase):
    def test_exported_tokenizer_preserves_unicode_digits_control_and_user_tokens_after_move(self):
        from tokenizers import Tokenizer
        with tempfile.TemporaryDirectory() as directory:
            before = Path(directory)/'before'
            after = Path(directory)/'moved directory'
            metadata = metadata_fixture()
            metadata['general.sampling.top_p'] = struct.unpack('<f', struct.pack('<f', .95))[0]
            export_metadata(before, metadata,
                {'text_config': {'max_position_embeddings': 262144}})
            shutil.move(before, after)
            tokenizer = Tokenizer.from_file(str(after/'tokenizer.json'))
            value = '中文 café🙂 123\n\tA'
            self.assertEqual(tokenizer.encode(value).ids, list(value.encode('utf-8')))
            self.assertEqual(tokenizer.decode(tokenizer.encode(value).ids), value)
            self.assertEqual(tokenizer.encode('<think>x</think><|im_end|>').ids, [258, 120, 259, 257])
            self.assertEqual(tokenizer.decode([258, 120, 259, 257], skip_special_tokens=True), '<think>x</think>')
            generation = json.loads((after/'generation_config.json').read_text())
            self.assertEqual(generation['eos_token_id'], [257,256])
            self.assertEqual(generation['top_p'], .95)

    def test_reader_decodes_real_gguf_arrays_strings_and_architecture_without_tensor_data(self):
        def string(value):
            data = value.encode('utf-8')
            return struct.pack('<Q', len(data))+data
        rows = [string('general.architecture')+struct.pack('<I',8)+string('qwen4exp'),
                string('tokenizer.ggml.tokens')+struct.pack('<IIQ',9,8,2)+string('中文')+string('abc'),
                string('tokenizer.ggml.token_type')+struct.pack('<IIQII',9,4,2,1,3)]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'metadata.gguf'
            path.write_bytes(struct.pack('<IIQQ',0x46554747,3,0,len(rows))+b''.join(rows))
            metadata = read_metadata(path)
            self.assertEqual(metadata['tokenizer.ggml.tokens'], ['中文', 'abc'])
            self.assertEqual(metadata['tokenizer.ggml.token_type'], [1,3])
            path.write_bytes(b'invalid file')
            with self.assertRaises((ValueError, struct.error)):
                read_metadata(path)


if __name__ == '__main__':
    unittest.main()
