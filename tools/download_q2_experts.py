"""Download the fixed, complete routed-expert source; shard 2 is not consumed."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import shutil
from q2_source_identity import DEFAULT_LOCK, load_expert_source, verify_expert_file


def download_ranges(output: Path, workers: int, source: str, identity: dict) -> Path:
    """可续传并发 Range 下载；每片精确核验范围，最终仍校验整文件 SHA256。"""
    import requests

    repo, revision, filename, size = (identity[key] for key in ("repo", "revision", "filename", "size"))
    destination = output / filename
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file():
        return destination
    parts = output / "range-parts" / identity["sha256"]
    parts.mkdir(parents=True, exist_ok=True)
    chunk = 16 << 20

    def transfer(start):
        end = min(start + chunk, size) - 1
        part = parts / f"{start:012d}-{end:012d}.part"
        if part.exists() and part.stat().st_size == end - start + 1:
            return part.stat().st_size
        temp = part.with_suffix(".incomplete")
        base = (f"https://www.modelscope.cn/models/{repo}/resolve/master/{filename}" if source == "modelscope"
                else f"https://huggingface.co/{repo}/resolve/{revision}/{filename}")
        url = f"{base}?range_start={start}"
        for attempt in range(6):
            try:
                with requests.get(url, headers={"Range": f"bytes={start}-{end}"},
                                  timeout=(20, 90), stream=True) as response:
                    response.raise_for_status()
                    if response.status_code != 206 or response.headers.get("Content-Range") != f"bytes {start}-{end}/{size}":
                        raise ValueError("server returned an unexpected range")
                    with temp.open("wb") as target:
                        for block in response.iter_content(1 << 20):
                            target.write(block)
                if temp.stat().st_size != end - start + 1:
                    raise ValueError("incomplete range download")
                temp.replace(part)
                return part.stat().st_size
            except (requests.RequestException, ValueError):
                if attempt == 5:
                    raise
                time.sleep(min(2 ** attempt, 10))

    completed = 0
    started = time.monotonic()
    with ThreadPoolExecutor(workers) as pool:
        jobs = [pool.submit(transfer, start) for start in range(0, size, chunk)]
        last = 0
        for job in as_completed(jobs):
            completed += job.result()
            now = time.monotonic()
            if now - last >= 30 or completed == size:
                print(json.dumps({"stage": "range_download", "bytes": completed, "total": size,
                                  "percent": round(100 * completed / size, 2),
                                  "MiB_s": round(completed / (now - started) / 2**20, 2)}), flush=True)
                last = now
    temporary = destination.with_suffix(".gguf.assembling")
    with temporary.open("wb") as target:
        for start in range(0, size, chunk):
            end = min(start + chunk, size) - 1
            with (parts / f"{start:012d}-{end:012d}.part").open("rb") as source:
                shutil.copyfileobj(source, target, 8 << 20)
    temporary.replace(destination)
    return destination

def main() -> None:
    # 第一片包含全部专家；主干和 PLE 使用独立、明确记录的 HF 来源。
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--transport", choices=("hub", "range"), default="range")
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--source", choices=("huggingface", "modelscope"), default="modelscope")
    parser.add_argument("--source-lock", type=Path, default=DEFAULT_LOCK)
    args = parser.parse_args()
    identity = load_expert_source(args.source_lock)
    from huggingface_hub import hf_hub_download

    started = time.time()
    print(json.dumps({"stage": "download", **identity}), flush=True)
    if not 1 <= args.workers <= 32:
        raise ValueError("download workers must be between 1 and 32")
    path = download_ranges(args.output, args.workers, args.source, identity) if args.transport == "range" else Path(
        hf_hub_download(repo_id=identity["repo"], revision=identity["revision"], filename=identity["filename"],
                       local_dir=str(args.output), token=False))
    print("Download complete; verifying SHA256", flush=True)
    stat = verify_expert_file(path, identity)
    record = {**identity, "verified_mtime_ns": stat.st_mtime_ns, "elapsed_s": time.time() - started,
              "download_source": args.source if args.transport == "range" else "huggingface",
              "scope": "complete routed experts; GGUF trunk and PLE not used"}
    (args.output / "expert-source.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(json.dumps({"stage": "verified", **record}), flush=True)


if __name__ == "__main__":
    main()
