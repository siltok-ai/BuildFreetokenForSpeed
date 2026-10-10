"""Cross-check expert row/projection identity between independent quantized releases."""
import argparse
import json
from pathlib import Path

import numpy as np
from safetensors import safe_open
import torch
import torch.nn.functional as F


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trunk-source", type=Path, required=True)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    lookup = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6, 0, -.5, -1, -1.5, -2, -3, -4, -6])
    records = {}
    with safe_open(args.trunk_source / "layer-00000-experts-0000-0127.safetensors", framework="pt", device="cpu") as source:
        for role in ("gate", "up", "down"):
            rows, columns = (2560, 640) if role == "down" else (640, 2560)
            raw = (args.samples / f"q2-layer0-{role}-sample.bin").read_bytes()
            blocks = np.frombuffer(raw, np.uint8).reshape(3, rows, columns // 64, 18)
            scales = blocks[..., :2].copy().view("<f2").astype(np.float32)[..., 0]
            codes = np.stack([(blocks[..., 2:] >> shift) & 3 for shift in (0, 2, 4, 6)], -1).reshape(3, rows, columns // 64, 64)
            q2 = torch.from_numpy(((codes.astype(np.float32) - 1) * scales[..., None]).reshape(3, -1))
            originals = []
            for expert in range(3):
                name = f"model.language_model.layers.0.mlp.experts.{expert}.{role}_proj"
                packed = source.get_tensor(name + ".weight").long()
                codes4 = torch.stack((packed & 15, packed >> 4), -1).reshape(rows, columns)
                scale4 = source.get_tensor(name + ".weight_scale").float().repeat_interleave(16, -1)
                global4 = source.get_tensor(name + ".weight_scale_2").float()
                originals.append((lookup[codes4] * scale4 * global4).reshape(-1))
            original = torch.stack(originals)
            correlation = F.normalize(q2, dim=1) @ F.normalize(original, dim=1).T
            if not torch.equal(correlation.argmax(1), torch.arange(3)) or correlation.diag().min() < .65:
                raise AssertionError(f"Q2_0 expert/projection identity mismatch: {role}: {correlation}")
            records[role] = correlation.tolist()
    result = {"status": "passed_sample_identity_only", "scope": "layer 0, experts 0/1/2, gate/up/down",
              "interpretation": "matching expert/projection has the highest cosine similarity across independently quantized sources",
              "correlations": records}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
