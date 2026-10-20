"""Resume fixed-revision GGUF downloads using verified HTTP byte ranges."""
from pathlib import Path
import json
import shutil
import time
from concurrent.futures import ThreadPoolExecutor, as_completed


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
