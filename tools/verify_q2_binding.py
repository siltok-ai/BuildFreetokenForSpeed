"""Exercise all 48 real model MoE constructors with the mixed-source configuration."""
import argparse
import json
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    from freetoken.engine.config import EngineConfig
    from freetoken.distributed import DistributedInfo, set_tp_info
    from freetoken.layers.quantization import QuantKind
    from freetoken.models.qwen4_exp.moe import Qwen4ExpMoE
    from freetoken.utils.torch_utils import torch_dtype

    set_tp_info(0, 1)
    config = EngineConfig(model_path=args.model, tp_info=DistributedInfo(0, 1), dtype=torch.bfloat16,
                          moe_strategy="offload", ple_backend="disk")
    model_config = config.model_config
    object.__setattr__(model_config, "moe_strategy", "offload")
    with torch.device("meta"), torch_dtype(torch.bfloat16):
        layers = [Qwen4ExpMoE(model_config, i, prefix=f"model.layers.{i}.mlp") for i in range(48)]
    for layer in layers:
        assert layer.experts.quant_method.kind is QuantKind.Q2_0
        assert layer.shared_expert.gate_up_proj.quant_method.kind is QuantKind.NONE
        assert layer.shared_expert.down_proj.quant_method.kind is QuantKind.NONE
        assert layer.gate.weight.dtype is torch.bfloat16
        assert layer.shared_expert_gate.weight.dtype is torch.bfloat16
    result = {"status": "passed_construction_only", "layers": len(layers), "expert_kind": "q2_0",
              "router_dtype": "bfloat16", "shared_expert_kind": "none", "expert_geometry": repr(layers[0].experts.quant_method.cfg),
              "layout": repr(layers[0].experts.quant_method.layout())}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
