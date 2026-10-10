"""Download the fixed, complete routed-expert source; shard 2 is not consumed."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import shutil


def download_ranges(output: Path, workers: int, source: str) -> Path:
    """可续传并发 Range 下载；每片精确核验范围，最终仍校验整文件 SHA256。"""
    import requests

    destination = output / FILENAME
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file():
        return destination
    parts = output / "range-parts"
    parts.mkdir(exist_ok=True)
    chunk = 16 << 20

    def transfer(start):
        end = min(start + chunk, SIZE) - 1
        part = parts / f"{start:012d}-{end:012d}.part"
        if part.exists() and part.stat().st_size == end - start + 1:
            return part.stat().st_size
        temp = part.with_suffix(".incomplete")
        base = (f"https://www.modelscope.cn/models/{REPO}/resolve/master/{FILENAME}" if source == "modelscope"
                else f"https://huggingface.co/{REPO}/resolve/{REVISION}/{FILENAME}")
        url = f"{base}?range_start={start}"
        for attempt in range(6):
            try:
                with requests.get(url, headers={"Range": f"bytes={start}-{end}"},
                                  timeout=(20, 90), stream=True) as response:
                    response.raise_for_status()
                    if response.status_code != 206 or response.headers.get("Content-Range") != f"bytes {start}-{end}/{SIZE}":
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
        jobs = [pool.submit(transfer, start) for start in range(0, SIZE, chunk)]
        last = 0
        for job in as_completed(jobs):
            completed += job.result()
            now = time.monotonic()
            if now - last >= 30 or completed == SIZE:
                print(json.dumps({"stage": "range_download", "bytes": completed, "total": SIZE,
                                  "percent": round(100 * completed / SIZE, 2),
                                  "MiB_s": round(completed / (now - started) / 2**20, 2)}), flush=True)
                last = now
    temporary = destination.with_suffix(".gguf.assembling")
    with temporary.open("wb") as target:
        for start in range(0, SIZE, chunk):
            end = min(start + chunk, SIZE) - 1
            with (parts / f"{start:012d}-{end:012d}.part").open("rb") as source:
                shutil.copyfileobj(source, target, 8 << 20)
    temporary.replace(destination)
    return destination

REPO = "ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF"
REVISION = "ed59f92082b1e93c0e96d60a8b11aab089b52f09"
FILENAME = "Q2_0/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf"
SIZE = 37_623_740_192
SHA256 = "69820c02ec7d0b45ef2ebb19d6620299db749fe2aded7f39f93c6b88b199b720"


def main() -> None:
    # 第一片包含全部专家；主干和 PLE 使用独立、明确记录的 HF 来源。
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--transport", choices=("hub", "range"), default="range")
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--source", choices=("huggingface", "modelscope"), default="modelscope")
    args = parser.parse_args()
    from huggingface_hub import hf_hub_download

    started = time.time()
    print(json.dumps({"stage": "download", "repo": REPO, "revision": REVISION,
                      "filename": FILENAME, "bytes": SIZE}), flush=True)
    if not 1 <= args.workers <= 32:
        raise ValueError("download workers must be between 1 and 32")
    path = download_ranges(args.output, args.workers, args.source) if args.transport == "range" else Path(
        hf_hub_download(repo_id=REPO, revision=REVISION, filename=FILENAME,
                       local_dir=str(args.output), token=False))
    if path.stat().st_size != SIZE:
        raise ValueError("expert source has an unexpected size")
    print("Download complete; verifying SHA256", flush=True)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    if digest.hexdigest() != SHA256:
        raise ValueError("expert source SHA256 mismatch")
    record = {"repo": REPO, "revision": REVISION, "filename": FILENAME,
              "size": SIZE, "sha256": SHA256, "elapsed_s": time.time() - started,
              "download_source": args.source if args.transport == "range" else "huggingface",
              "scope": "complete 48x512 routed experts; GGUF trunk and PLE not used"}
    (args.output / "expert-source.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(json.dumps({"stage": "verified", **record}), flush=True)


if __name__ == "__main__":
    main()
