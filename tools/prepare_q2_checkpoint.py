"""Build an explicit BF16-trunk / Q2_0-expert checkpoint from verified raw sources."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import struct

from download_q2_experts import FILENAME, REPO, REVISION, SHA256, SIZE


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trunk", type=Path, required=True)
    parser.add_argument("--trunk-receipt", type=Path,
                        help="HF source checksum receipt; defaults to the historical receipt in --trunk")
    parser.add_argument("--experts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--finalize", action="store_true")
    args = parser.parse_args()
    if args.finalize:
        verified = read(args.experts / "expert-source.json")
        expert_file = args.experts / FILENAME
        if verified.get("sha256") != SHA256 or expert_file.stat().st_size != SIZE:
            raise ValueError("complete Q2_0 artifact has not been verified")
        from freetoken.models.qwen4_exp.q2_source import read_q2_directory
        directory = read_q2_directory(expert_file)
        config = read(args.output / "config.json")
        config["freetoken_expert_source"].update(verified_mtime_ns=expert_file.stat().st_mtime_ns)
        write(args.output / "config.json", config)
        record = read(args.output / "mixed-source.json")
        record.update(status="complete_verified_sources_requires_inference_validation", expert_receipt=verified,
                      expert_tensor_count=len(directory), expert_bytes=33_973_862_400)
        write(args.output / "mixed-source.json", record)
        print(json.dumps({"status": record["status"], "experts": len(directory), "expert_GiB": 31.640625}), flush=True)
        return

    if args.output.exists():
        raise FileExistsError("use a new checkpoint directory")
    # 独立原始 HF 目录使用显式校验清单；旧目录仅作为原始文件来源兼容。
    provenance = read(args.trunk_receipt or args.trunk / "INVALID-EXPERT-ALIASES.json")
    if provenance.get("model_id") != "RadixArk/Qwen3.8-Flash-Next-NVFP4":
        raise ValueError("unexpected raw trunk provenance")
    index_path = args.trunk / "model.safetensors.index.json"
    with index_path.open("rb") as stream:
        if hashlib.file_digest(stream, "sha256").hexdigest() != provenance["index_sha256"]:
            raise ValueError("original HF index identity changed")
    index = read(index_path)
    # 原实验仅改变运行时专家映射；复用重新验 hash 的原始非专家文件，不复用任何别名专家。
    weight_map = {key: value for key, value in index["weight_map"].items() if ".experts." not in key}
    files = sorted(set(weight_map.values()))
    receipts = {entry["name"]: entry for entry in provenance["completed"]}
    total = 0
    verified = {}
    for name in files:
        path = args.trunk / name
        expected = receipts[name]
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if digest != expected["sha256"] or path.stat().st_size != expected["bytes"]:
            raise ValueError(f"HF raw trunk checksum mismatch: {name}")
        with path.open("rb") as stream:
            header_size, = struct.unpack("<Q", stream.read(8))
            header = json.loads(stream.read(header_size))
        for key in (key for key, shard in weight_map.items() if shard == name):
            if key not in header:
                raise ValueError(f"indexed raw trunk tensor missing: {key}")
            ple_lookup = ".ple.ple_embedding.ngram_embedding." in key
            if header[key]["dtype"] not in (("BF16", "F8_E4M3") if ple_lookup else ("BF16", "I64")):
                raise ValueError(f"unexpected trunk precision: {key}: {header[key]['dtype']}")
            total += header[key]["data_offsets"][1] - header[key]["data_offsets"][0]
        verified[name] = {"sha256": digest, "bytes": path.stat().st_size, "mtime_ns": path.stat().st_mtime_ns}
        print(f"verified raw non-expert shard: {name}", flush=True)
    args.output.mkdir(parents=True)
    for name in files:
        os.link(args.trunk / name, args.output / name)
    side_files = ("config.json", "tokenizer.json", "tokenizer_config.json", "generation_config.json",
                  "chat_template.jinja", "merges.txt", "vocab.json", "preprocessor_config.json",
                  "processor_config.json", "special_tokens_map.json", "added_tokens.json")
    for name in side_files:
        if (args.trunk / name).exists():
            shutil.copyfile(args.trunk / name, args.output / name)
    config = read(args.output / "config.json")
    config["freetoken_expert_source"] = {"format": "q2_0", "scope": "routed_experts_only",
        "path": str((args.experts / FILENAME).resolve()), "sha256": SHA256, "size": SIZE,
        "repo": REPO, "revision": REVISION, "verified_mtime_ns": None,
        "layers": 48, "experts_per_layer": 512, "hidden": 2560, "intermediate": 640,
        "expert_bytes": 33_973_862_400}
    write(args.output / "config.json", config)
    write(args.output / "model.safetensors.index.json", {"metadata": {"total_size": total}, "weight_map": weight_map})
    write(args.output / "mixed-source.json", {"format": 1, "status": "trunk_verified_experts_pending",
        "base_model": "Qwen/Qwen3.8-Flash-Next", "trunk_source": provenance["model_id"],
        "trunk_source_directory": str(args.trunk.resolve()), "original_index_sha256": provenance["index_sha256"],
        "non_expert_files": verified, "trunk_precision": "BF16", "ple_precision": "FP8 E4M3 with original BF16 scale",
        "expert_precision": "GSQ Q2_0 (64 elements / 18 bytes)", "expert_aliases": False,
        "scope": "all 48x512 routed experts; HF non-experts and tokenizer; SSD HF PLE"})
    print(json.dumps({"status": "trunk_verified_experts_pending", "non_expert_shards": len(files),
                      "non_expert_tensors": len(weight_map)}), flush=True)


if __name__ == "__main__":
    main()
