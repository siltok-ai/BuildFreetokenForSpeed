"""Process-only draft head subset; target weights and full-vocabulary verify stay intact."""
from __future__ import annotations

import hashlib
import inspect
import textwrap
from pathlib import Path

import torch

VOCAB = Path(r'C:\workspace\models\strata\flash-next-mtp-reference\draft_vocab.bin')
VOCAB_SHA = 'b1e1d3a7a9e4bf862dcd5923ce661fb59bbd07907e594df5cf86a62ac235cb91'


def enable():
    """Bind byte-identical Q5 rows to the drafter only, before graph capture."""
    import freetoken.models.qwen4_exp.mtp_drafter as module

    original_init = module.MtpDrafter.__init__
    raw = VOCAB.read_bytes()
    if hashlib.sha256(raw).hexdigest() != VOCAB_SHA:
        raise ValueError('Reference draft vocabulary identity changed')
    ids = torch.frombuffer(bytearray(raw), dtype=torch.int32).to(torch.int64)
    # Ascending token ids preserve the full head's argmax tie order within the subset.
    ids = ids.sort().values
    if len(ids) != 106299 or ids.unique().numel() != len(ids):
        raise ValueError('Reference draft vocabulary is incomplete or duplicated')

    def init(self, runtime, main_model, **kwargs):
        """Allocate the extra draft head explicitly and include it in ownership accounting."""
        original_init(self, runtime, main_model, **kwargs)
        vocab_size = main_model._config.vocab_size
        if vocab_size != 248320 or int(ids[0]) < 0 or int(ids[-1]) >= vocab_size:
            raise ValueError('Draft vocabulary does not match this target model')
        head = self.head.packed_0
        if head.dtype != torch.uint8 or head.numel() != vocab_size * 1760:
            raise ValueError('Prototype requires the original 2560-wide Q5_K head')
        self.draft_vocab = ids.to(self.device)
        self.draft_head_packed = head.view(vocab_size, 1760).index_select(0, self.draft_vocab).contiguous().view(-1)
        self.owned_bytes += self.draft_vocab.numel() * self.draft_vocab.element_size() + self.draft_head_packed.numel()
        from freetoken.kernel.gguf_quant import compressed_linear
        generator = torch.Generator(device=self.device).manual_seed(731)
        checks = []
        for _ in range(3):
            x = torch.randn((1, 2560), device=self.device, dtype=torch.float32, generator=generator)
            full = self.head.quant_method.apply(self.head, x)
            subset = compressed_linear(self.draft_head_packed, x, len(self.draft_vocab), 13)
            equal = torch.equal(full.index_select(1, self.draft_vocab), subset)
            checks.append(equal)
            if not equal:
                raise RuntimeError('Copied draft rows changed the original Q5_K logits')
        self.draft_subset_check = dict(rows=len(ids),packed_bytes=self.draft_head_packed.numel(),
                                      index_bytes=self.draft_vocab.numel()*self.draft_vocab.element_size(),
                                      exact_full_head_subset_checks=checks,target_full_rows=vocab_size,
                                      vocab_sha256=VOCAB_SHA)

    original = textwrap.dedent(inspect.getsource(module.MtpDrafter.step))
    source = original.replace(
        'logits = self.head.quant_method.apply(self.head, mixed)',
        'logits = compressed_linear(self.draft_head_packed, mixed, len(self.draft_vocab), 13)',
    ).replace(
        'return (logits.argmax(-1).to(torch.int32), r)',
        'return (self.draft_vocab[logits.argmax(-1)].to(torch.int32), r)',
    )
    if source == original or source.count('self.draft_vocab[logits.argmax') != 1:
        raise RuntimeError('Draft step no longer matches the prototype source boundary')
    namespace = dict(module.__dict__)
    exec(compile(source, str(Path(__file__)), 'exec'), namespace)
    module.MtpDrafter.__init__ = init
    module.MtpDrafter.step = namespace['step']

