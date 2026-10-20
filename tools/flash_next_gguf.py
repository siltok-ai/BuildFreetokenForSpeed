"""从 GGUF 自身导出 Flash Next 的 HF 兼容配置和 tokenizer，不下载第二份元数据。"""
from pathlib import Path
import json
import struct


# GGUF qwen35 的预分词合同；与 Qwen2Tokenizer 的逐数字、Unicode 标记处理一致。
PRETOKENIZER = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
SCALARS = {0: 'B', 1: 'b', 2: 'H', 3: 'h', 4: 'I', 5: 'i', 6: 'f',
           7: '?', 10: 'Q', 11: 'q', 12: 'd'}


def read_metadata(path):
    """只读 GGUF v3 元数据，不读取大张量，也不依赖 Strata 或 GPU。"""
    with Path(path).open('rb') as stream:
        def unpack(fmt):
            return struct.unpack('<'+fmt, stream.read(struct.calcsize('<'+fmt)))

        def string():
            return stream.read(unpack('Q')[0]).decode('utf-8')

        def value(kind):
            if kind == 8:
                return string()
            if kind == 9:
                element, count = unpack('IQ')
                return [value(element) for _ in range(count)]
            return unpack(SCALARS[kind])[0]

        magic, version, _, count = unpack('IIQQ')
        if magic != 0x46554747 or version != 3:
            raise ValueError('Flash Next requires GGUF v3: '+str(path))
        metadata = {}
        for _ in range(count):
            key = string()
            metadata[key] = value(unpack('I')[0])
    if metadata.get('general.architecture') != 'qwen4exp':
        raise ValueError('Flash Next requires qwen4exp GGUF metadata')
    return metadata


def model_config(metadata, shards, geometry):
    """将 GGUF 几何映射到现有 FreeToken 配置；未存储的值使用该架构固定常量。"""
    fields = {
        'block_count': 'num_hidden_layers', 'context_length': 'max_position_embeddings',
        'embedding_length': 'hidden_size', 'attention.head_count': 'num_attention_heads',
        'attention.head_count_kv': 'num_key_value_heads', 'attention.key_length': 'head_dim',
        'expert_count': 'num_experts', 'expert_used_count': 'num_experts_per_tok',
        'expert_feed_forward_length': 'moe_intermediate_size',
        'expert_shared_feed_forward_length': 'shared_expert_intermediate_size',
        'ssm.conv_kernel': 'linear_conv_kernel_dim', 'ssm.state_size': 'linear_key_head_dim',
        'ssm.group_count': 'linear_num_key_heads', 'ssm.time_step_rank': 'linear_num_value_heads',
        'full_attention_interval': 'full_attention_interval',
        'hyper_connection.count': 'hc_count', 'hyper_connection.low_rank': 'hc_lowrank',
        'attention.indexer.head_count': 'indexer_n_heads',
        'attention.indexer.key_length': 'indexer_head_dim', 'attention.indexer.top_k': 'indexer_budget',
        'ple.ngram_size': 'ngram_size', 'ple.heads_per_ngram': 'heads_per_ngram',
        'ple.conv_kernel': 'ple_conv_kernel_size',
    }
    text = {field: metadata['qwen4exp.'+key] for key, field in fields.items()}
    text.update(model_type='qwen4_exp_text', vocab_size=len(metadata['tokenizer.ggml.tokens']),
        dtype='bfloat16', attention_bias=False, hidden_act='silu', output_gate_type='sigmoid',
        linear_value_head_dim=metadata['qwen4exp.ssm.inner_size']//text['linear_num_value_heads'],
        indexer_kv_heads=1, indexer_compress_ratio=max(metadata['qwen4exp.attention.compress_ratios']),
        # GGUF PLE 层编号是零基；FreeToken 的 HF 配置是一区间编号。
        ple_layer_ids=[i+1 for i in metadata['qwen4exp.ple.layers']],
        ple_embed_dim=metadata['qwen4exp.embedding_length_per_layer_input']*
            (text['ngram_size']-1)*text['heads_per_ngram'],
        ngram_vocab_size_base=20000000, split_ngram_parts=128,
        make_ngram_vocab_size_divisible_by=128, ple_embedding_dtype='float8_e4m3fn',
        rms_norm_eps=round(metadata['qwen4exp.attention.layer_norm_rms_epsilon'], 12),
        bos_token_id=metadata['tokenizer.ggml.bos_token_id'],
        eos_token_id=metadata['qwen4exp.ple.eos_token_id'], tie_word_embeddings=False,
        mamba_ssm_dtype='float32', mtp_num_hidden_layers=1, mtp_use_dedicated_embeddings=False)
    interval = text['full_attention_interval']
    text['layer_types'] = ['full_attention' if (i+1)%interval == 0 else 'linear_attention'
                           for i in range(text['num_hidden_layers'])]
    theta = metadata['qwen4exp.rope.freq_base']
    text['partial_rotary_factor'] = metadata['qwen4exp.rope.dimension_count']/text['head_dim']
    text['rope_parameters'] = dict(rope_theta=theta, rope_type='default',
        partial_rotary_factor=text['partial_rotary_factor'], mrope_interleaved=True,
        mrope_section=metadata['qwen4exp.rope.dimension_sections'][:3])
    text['mtp'] = dict(hybrid=True, layer_types=['full_attention'], num_hidden_layers=1,
                       mtp_use_hidden_state_from_layer=None, rope_theta=theta)
    return dict(architectures=['Qwen4ExpForConditionalGeneration'], model_type='qwen4_exp',
        language_model_only=True, text_config=text, tie_word_embeddings=False,
        freetoken_gguf_source=dict(format='gguf_strata', shards=shards),
        freetoken_expert_source=dict(format='q2_0', scope='routed_experts_only',
            path=shards[0]['path'], size=shards[0]['size'], **geometry))


def export_metadata(output, metadata, config):
    """生成标准 HF 本地 tokenizer；所有 token ID、merge 顺序和模板均来自 GGUF。"""
    from tokenizers import AddedToken, Regex, Tokenizer, decoders, models, pre_tokenizers

    if (metadata['tokenizer.ggml.model'], metadata['tokenizer.ggml.pre']) != ('gpt2', 'qwen35'):
        raise ValueError('Unsupported Flash Next tokenizer metadata')
    tokens = metadata['tokenizer.ggml.tokens']
    tokenizer = Tokenizer(models.BPE(vocab={t:i for i,t in enumerate(tokens)},
        merges=[tuple(pair.split(' ', 1)) for pair in metadata['tokenizer.ggml.merges']],
        byte_fallback=False))
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Split(Regex(PRETOKENIZER), behavior='isolated'),
        pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)])
    tokenizer.decoder = decoders.ByteLevel()
    # CONTROL 参与 skip_special_tokens；USER_DEFINED（例如 think）需保留为普通输出。
    for kind in (3, 4):
        added = [AddedToken(t, normalized=False, special=kind == 3)
                 for t,typ in zip(tokens, metadata['tokenizer.ggml.token_type']) if typ == kind]
        if kind == 3:
            tokenizer.add_special_tokens(added)
        else:
            tokenizer.add_tokens(added)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(output/'tokenizer.json'))
    template = metadata['tokenizer.chat_template']
    (output/'chat_template.jinja').write_text(template, encoding='utf-8', newline='\n')
    def write(name, data):
        (output/name).write_text(json.dumps(data, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    write('config.json', config)
    write('tokenizer_config.json', dict(tokenizer_class='Qwen2Tokenizer',
        model_max_length=config['text_config']['max_position_embeddings'],
        add_prefix_space=False, add_bos_token=metadata.get('tokenizer.ggml.add_bos_token', False),
        clean_up_tokenization_spaces=False, split_special_tokens=False,
        # 显式保留 qwen35 正则，避免 Transformers 对大词表触发 Mistral 修补猜测。
        fix_mistral_regex=False,
        eos_token=tokens[metadata['tokenizer.ggml.eos_token_id']],
        pad_token=tokens[metadata['tokenizer.ggml.padding_token_id']], bos_token=None,
        chat_template=template))
    eos = list(dict.fromkeys([metadata['tokenizer.ggml.eos_token_id'], metadata['qwen4exp.ple.eos_token_id']]))
    write('generation_config.json', dict(bos_token_id=metadata['tokenizer.ggml.bos_token_id'],
        eos_token_id=eos, pad_token_id=metadata['tokenizer.ggml.padding_token_id'], do_sample=True,
        temperature=metadata['general.sampling.temp'], top_k=metadata['general.sampling.top_k'],
        # GGUF 使用 float32；还原采样默认值的十进制表示（例如 0.949999988 -> 0.95）。
        top_p=round(metadata['general.sampling.top_p'], 6)))
