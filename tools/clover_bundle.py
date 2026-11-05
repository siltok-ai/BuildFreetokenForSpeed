"""从固定官方 Q8 GGUF 制作可搬迁 Clover 模型目录，不运行 CUDA。"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

from flash_next_gguf import read_metadata

ROOT = Path(__file__).resolve().parents[1]
LOCK = ROOT/'model-weights/clover-q8/source-lock.json'


def write(path, value):
    """写入可跨机器复用的 UTF-8 元数据。"""
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')


def model_config(meta, filename, size):
    """从 GGUF 导出几何；引擎仍直接读取原始 GGUF，不解量化权重。"""
    def g(key):
        return meta['clover_moe.'+key]
    text = dict(hidden_size=g('embedding_length'), vocab_size=len(meta['tokenizer.ggml.tokens']),
        num_hidden_layers=g('block_count'), num_attention_heads=g('attention.head_count'),
        num_key_value_heads=g('attention.head_count_kv'), head_dim=g('attention.key_length'),
        num_experts=g('expert_count'), num_experts_per_tok=g('expert_used_count'),
        moe_intermediate_size=g('expert_feed_forward_length'),
        shared_expert_intermediate_size=g('expert_shared_feed_forward_length'),
        linear_num_key_heads=g('ssm.group_count'), linear_num_value_heads=g('ssm.time_step_rank'),
        linear_key_head_dim=g('ssm.state_size'),
        linear_value_head_dim=g('ssm.inner_size')//g('ssm.time_step_rank'),
        linear_conv_kernel_dim=g('ssm.conv_kernel'), max_position_embeddings=g('context_length'),
        layer_types=['linear_attention' if r else 'full_attention'
                     for r in g('attention.recurrent_layers')])
    return dict(architectures=['CloverMoeGGUFModelLM'], model_type='clover_moe',
        text_config=text, freetoken_gguf_source=dict(format='clover_gguf',
            quantization='q8_0', shards=[dict(path=filename, size=size)]))


def export_tokenizer(output, meta):
    """复现引擎 Clover tokenizer 的转换和控制 token 注册，保留 GGUF 模板。"""
    from transformers import PreTrainedTokenizerFast
    from transformers.integrations.ggml import convert_gguf_tokenizer
    from tokenizers import AddedToken, normalizers
    raw = {k[len('tokenizer.ggml.'):]: v for k,v in meta.items() if k.startswith('tokenizer.ggml.')}
    fast, _ = convert_gguf_tokenizer('qwen2', raw)
    fast.normalizer = normalizers.NFC() if raw.get('normalizer.nfc', False) else None
    for token, kind in zip(raw['tokens'], raw['token_type'], strict=True):
        if kind in (3, 4):
            added = AddedToken(token, normalized=False, special=kind == 3)
            (fast.add_special_tokens if kind == 3 else fast.add_tokens)([added])
    def token(key):
        return raw['tokens'][raw[key]]
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=fast, unk_token=None,
        bos_token=token('bos_token_id'), eos_token=token('eos_token_id'),
        pad_token=token('padding_token_id'), mask_token=token('mask_token_id'))
    tokenizer.chat_template = meta['tokenizer.chat_template']
    tokenizer.model_max_length = meta['clover_moe.context_length']
    tokenizer.init_kwargs['fix_mistral_regex'] = False
    tokenizer.save_pretrained(output)
    (output/'chat_template.jinja').write_text(tokenizer.chat_template, encoding='utf-8')
    write(output/'generation_config.json', dict(do_sample=False, temperature=0.0,
        top_p=1.0, top_k=0, eos_token_id=list(dict.fromkeys(
            [raw['eos_token_id'], raw['padding_token_id']])), pad_token_id=raw['padding_token_id']))


def prepare(output, gguf, lock_path=LOCK):
    """流式验证固定来源后复制权重，生成不含制作机器绝对路径的资源包。"""
    output, gguf = Path(output).resolve(), Path(gguf).resolve()
    lock = json.loads(Path(lock_path).read_text(encoding='utf-8'))
    if output.exists():
        raise FileExistsError('Use a new resource directory: '+str(output))
    with gguf.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    if gguf.stat().st_size != lock['size'] or digest != lock['sha256']:
        raise ValueError('Official Clover Q8 source size/SHA256 mismatch')
    meta = read_metadata(gguf, 'clover_moe')
    output.mkdir(parents=True)
    shutil.copyfile(gguf, output/lock['filename'])
    write(output/'config.json', model_config(meta, lock['filename'], lock['size']))
    export_tokenizer(output, meta)
    models = [dict(id='clover-35b-a3b.q8', name='Clover 35B A3B Q8', type='language',
        family='clover', features=['chat'], engine=dict(pipeline='chat', backend='FreeTokenBackend',
        offload=False, resource_profile='auto'),
        artifacts=dict(weights=dict(path='.', layout='hf_model')))]
    write(output/'manifest.json', dict(schema='silitok.models/1', models=models))
    write(output/'sources.json', lock)
    shutil.copy2(ROOT/'NOTICE', output/'NOTICE')
    shutil.copy2(ROOT/'LICENSE', output/'LICENSE')
    shutil.copy2(ROOT/'model-weights/clover-q8/README.md', output/'README.md')
    inventory = {}
    for path in sorted(output.rglob('*')):
        if path.is_file():
            with path.open('rb') as stream:
                sha = hashlib.file_digest(stream, 'sha256').hexdigest()
            inventory[path.relative_to(output).as_posix()] = dict(size=path.stat().st_size, sha256=sha)
    write(output/'files.json', inventory)
    return dict(status='prepared_requires_gpu_validation', model='clover-35b-a3b.q8', files=len(inventory))


def verify_bundle(root):
    """在接收机器流式核验传输清单，不加载模型或触碰GPU。"""
    root = Path(root).resolve()
    inventory = json.loads((root/'files.json').read_text(encoding='utf-8'))
    for name, expected in inventory.items():
        path = (root/name).resolve()
        if not path.is_relative_to(root) or path.stat().st_size != expected['size']:
            raise ValueError('Resource file length/path mismatch: '+name)
        with path.open('rb') as stream:
            if hashlib.file_digest(stream, 'sha256').hexdigest() != expected['sha256']:
                raise ValueError('Resource file SHA256 mismatch: '+name)
    return dict(status='transfer_verified', files=len(inventory), runtime_qualified=False)


def main():
    """从显式权重输入生成新资源包，不修改输入或覆盖现有包。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gguf', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--verify', type=Path)
    args = parser.parse_args()
    if args.verify:
        if args.gguf or args.output:
            parser.error('--verify cannot be combined with preparation options')
        print(json.dumps(verify_bundle(args.verify)))
    else:
        if not args.gguf or not args.output:
            parser.error('preparation requires --gguf and --output')
        print(json.dumps(prepare(args.output, args.gguf)))


if __name__ == '__main__':
    main()
