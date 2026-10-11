"""Q2 下载来源由锁文件定义，实际字节在准备阶段校验。"""
import hashlib
import json
from pathlib import Path

DEFAULT_LOCK = Path(__file__).resolve().parents[1] / "model-weights/qwen3.8-flash-next-q2/source-lock.json"


def load_expert_source(path=DEFAULT_LOCK):
    """读取明确锁定的专家来源，防止身份常量在多个下载工具中漂移。"""
    entry = json.loads(Path(path).read_text(encoding="utf-8"))["direct_sources"]["experts"]
    name = Path(entry["filename"])
    if name.is_absolute() or ".." in name.parts:
        raise ValueError("expert filename must stay within the download directory")
    return {"repo": entry["repo"], "revision": entry["revision"], "filename": entry["filename"],
            "size": entry["bytes"], "sha256": entry["sha256"]}


def verify_expert_file(path, identity):
    """重新核验文件内容后才返回本次身份；旧下载回执不能替代实际字节。"""
    path = Path(path)
    before = path.stat()
    if before.st_size != identity["size"]:
        raise ValueError("Q2_0 expert source size mismatch")
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    after = path.stat()
    if (before.st_size, before.st_mtime_ns, before.st_ino) != (after.st_size, after.st_mtime_ns, after.st_ino):
        raise ValueError("Q2_0 expert source changed during checksum verification")
    if digest != identity["sha256"]:
        raise ValueError("Q2_0 expert source SHA256 mismatch")
    return after


def model_geometry(config):
    """从 HF 文本配置计算完整专家容量，不复制某个模型的数值常量。"""
    text = config.get("text_config", config)
    if config.get("model_type") != "qwen4_exp":
        raise ValueError("Q2_0 external experts require a qwen4_exp trunk")
    values = {"layers": text["num_hidden_layers"], "experts_per_layer": text["num_experts"],
              "hidden": text["hidden_size"], "intermediate": text["moe_intermediate_size"]}
    if (any(value <= 0 for value in values.values()) or values["hidden"] % 64
            or values["intermediate"] % 64):
        raise ValueError("invalid GSQ Q2_0 model geometry")
    values["expert_bytes"] = 3 * values["layers"] * values["experts_per_layer"] * values["hidden"] * values["intermediate"] // 64 * 18
    return values
