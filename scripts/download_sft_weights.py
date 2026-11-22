"""Download the pinned Qwen checkpoint and verify upstream hashes; never train."""
from __future__ import annotations

import hashlib
import argparse
import json
import os
import re
import time
import urllib.request
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from zoneinfo import ZoneInfo

def sha256(path):
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def download_http(repository, revision, output, asset):
    """Stream with Range resumption using the OS HTTP stack, with bounded memory."""
    name, size = asset["rfilename"], asset["size"]
    path = output / name
    if path.exists() and path.stat().st_size == size:
        return  # The common verification below still checks every upstream hash.
    partial = path.with_suffix(path.suffix + ".download.part")
    attempts = 0
    while not partial.exists() or partial.stat().st_size < size:
        offset = partial.stat().st_size if partial.exists() else 0
        if offset > size:
            raise ValueError("Partial checkpoint exceeds expected size")
        url = f"https://huggingface.co/{repository}/resolve/{revision}/{name}?download=true&resume_offset={offset}"
        request = urllib.request.Request(url, headers={"Range": f"bytes={offset}-{size-1}", "Cache-Control": "no-cache"})
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                if response.status == 206:
                    match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", response.headers.get("Content-Range", ""))
                    if not match or int(match[1]) != offset or int(match[3]) != size:
                        raise ValueError("Server returned a mismatched checkpoint range")
                elif response.status != 200 or offset != 0:
                    raise ValueError("Server did not honor checkpoint resumption")
                print(json.dumps({"phase": "http_stream", "file": name, "resume_bytes": offset, "total_bytes": size}), flush=True)
                next_progress = offset + 128 * 1024 * 1024
                with partial.open("ab") as handle:
                    while chunk := response.read(4 * 1024 * 1024):
                        handle.write(chunk)
                        offset += len(chunk)
                        if offset > size:
                            raise ValueError("Checkpoint transfer exceeds expected size")
                        if offset >= next_progress:
                            handle.flush()
                            print(json.dumps({"phase": "http_progress", "downloaded_bytes": offset, "total_bytes": size}), flush=True)
                            next_progress = offset + 128 * 1024 * 1024
            attempts = 0
        except (OSError, TimeoutError) as exc:
            attempts += 1
            print(json.dumps({"phase": "http_retry", "attempt": attempts, "error_type": type(exc).__name__}), flush=True)
            if attempts >= 8:
                raise
            time.sleep(2)
    if partial.stat().st_size != size:
        raise ValueError("Partial checkpoint size mismatch")
    if asset.get("lfs") and sha256(partial) != asset["lfs"]["sha256"]:
        raise ValueError("Checkpoint LFS hash mismatch before publication")
    partial.replace(path)


def download_http_parallel(repository, revision, output, asset, workers):
    name, size = asset["rfilename"], asset["size"]
    path = output / name
    if path.exists() and path.stat().st_size == size:
        return
    partial = path.with_suffix(path.suffix + ".download.part")
    offset = partial.stat().st_size if partial.exists() else 0
    if offset > size:
        raise ValueError("Checkpoint prefix exceeds expected size")
    cache = output / ".cache" / "sft-weight-ranges"
    cache.mkdir(parents=True, exist_ok=True)
    ranges = [(start, min(size-1, start+32*1024*1024-1)) for start in range(offset, size, 32*1024*1024)]

    def fetch(bounds):
        start, end = bounds
        segment = cache / f"{start}-{end}.part"
        expected = end-start+1
        if segment.exists() and segment.stat().st_size > expected:
            raise ValueError("Checkpoint segment exceeds expected size")
        failures = 0
        while not segment.exists() or segment.stat().st_size < expected:
            cursor = start + (segment.stat().st_size if segment.exists() else 0)
            url = f"https://huggingface.co/{repository}/resolve/{revision}/{name}?download=true&range_start={cursor}&range_end={end}"
            request = urllib.request.Request(url, headers={"Range": f"bytes={cursor}-{end}", "Cache-Control": "no-cache"})
            try:
                started = time.monotonic()
                with urllib.request.urlopen(request, timeout=30) as response:
                    match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", response.headers.get("Content-Range", ""))
                    if response.status != 206 or not match or tuple(map(int, match.groups())) != (cursor, end, size):
                        raise ValueError("Checkpoint segment response has incorrect range")
                    with segment.open("ab") as handle:
                        remaining = end-cursor+1
                        while remaining:
                            # read1 exposes each received TLS block instead of waiting
                            # for a full multi-MB buffer on a stalled connection.
                            chunk = response.read1(min(1024*1024, remaining))
                            if not chunk:
                                raise OSError("Checkpoint range ended before all bytes arrived")
                            handle.write(chunk)
                            remaining -= len(chunk)
                            if remaining and time.monotonic()-started > 90:
                                raise TimeoutError("Restart a slow range while preserving received bytes")
                failures = 0
            except OSError as exc:
                failures += 1
                print(json.dumps({"phase": "range_retry", "start": start, "attempt": failures, "error_type": type(exc).__name__}), flush=True)
                if failures >= 8:
                    raise
                time.sleep(2)
        if segment.stat().st_size != expected:
            raise ValueError("Checkpoint segment length mismatch")
        return segment

    print(json.dumps({"phase": "parallel_http", "resume_bytes": offset, "total_bytes": size, "workers": workers}), flush=True)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending = [pool.submit(fetch, bounds) for bounds in ranges]
        with partial.open("ab") as destination:
            for future in pending:
                segment = future.result()
                with segment.open("rb") as source:
                    for chunk in iter(lambda: source.read(4*1024*1024), b""):
                        destination.write(chunk)
                        offset += len(chunk)
                destination.flush()
                # Only our completed segment is removed, after its bytes are appended in order.
                segment.unlink()
                print(json.dumps({"phase": "http_progress", "downloaded_bytes": offset, "total_bytes": size}), flush=True)
    if partial.stat().st_size != size or (asset.get("lfs") and sha256(partial) != asset["lfs"]["sha256"]):
        raise ValueError("Assembled checkpoint size or LFS hash mismatch")
    partial.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--http", action="store_true", help="Use resumable HTTP instead of the Xet transfer backend")
    parser.add_argument("--workers", type=int, default=1, choices=range(1, 9), help="Parallel HTTP range transfers")
    args = parser.parse_args()
    if args.http:
        os.environ["HF_HUB_DISABLE_XET"] = "1"
    from huggingface_hub import snapshot_download
    manifest = json.loads(Path("reports/sft-tokenizer-download.json").read_text(encoding="utf-8"))
    repository, revision = manifest["repository"], manifest["revision"]
    if repository != "Qwen/Qwen3.5-2B" or revision != "15852e8c16360a2fea060d615a32b45270f8a8fc":
        raise ValueError("Expected the same approved model revision as the audited tokenizer")
    output = Path("models/Qwen3.5-2B")
    for asset in manifest["files"]:
        if sha256(output / asset["file"]) != asset["sha256"]:
            raise ValueError("Audited tokenizer asset changed: " + asset["file"])
    with urllib.request.urlopen(f"https://huggingface.co/api/models/{repository}/revision/{revision}?blobs=true", timeout=60) as response:
        info = json.load(response)
    if info["sha"] != revision:
        raise ValueError("Upstream revision mismatch")
    assets = [asset for asset in info["siblings"] if asset["rfilename"].endswith((".safetensors", ".safetensors.index.json"))
              or asset["rfilename"] == "generation_config.json"]
    print(json.dumps({"phase": "download", "revision": revision, "files": len(assets),
                      "bytes": sum(asset.get("size", 0) for asset in assets)}), flush=True)
    if args.http:
        for asset in assets:
            if args.workers > 1:
                download_http_parallel(repository, revision, output, asset, args.workers)
            else:
                download_http(repository, revision, output, asset)
    else:
        snapshot_download(repo_id=repository, revision=revision, local_dir=str(output), token=False,
                          allow_patterns=["*.safetensors", "*.safetensors.index.json", "generation_config.json"], max_workers=2)
    verified = []
    for asset in assets:
        name = asset["rfilename"]
        path = output / name
        path.resolve().relative_to(output.resolve())
        actual = sha256(path)
        if asset.get("lfs"):
            if actual != asset["lfs"]["sha256"]:
                raise ValueError("Checkpoint LFS hash mismatch: " + name)
        else:
            blob_hash = hashlib.sha1(b"blob " + str(path.stat().st_size).encode() + b"\0" + path.read_bytes()).hexdigest()
            if blob_hash != asset["blobId"]:
                raise ValueError("Git blob hash mismatch: " + name)
        if asset.get("size") is not None and path.stat().st_size != asset["size"]:
            raise ValueError("Checkpoint size mismatch: " + name)
        verified.append({"file": name, "sha256": actual, "bytes": path.stat().st_size, "upstream_hash_verified": True})
        print(json.dumps({"phase": "verified", **verified[-1]}), flush=True)
    index_path = output / "model.safetensors.index.json"
    if index_path.exists():
        index = json.loads(index_path.read_text(encoding="utf-8"))
        for name in set(index["weight_map"].values()):
            if name not in {item["file"] for item in verified}:
                raise ValueError("Checkpoint index references an unverified shard: " + name)
    for asset in manifest["files"]:
        if sha256(output / asset["file"]) != asset["sha256"]:
            raise ValueError("Tokenizer unexpectedly changed during checkpoint download")
    report = {"status": "passed", "finished_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
              "repository": repository, "revision": revision, "files": verified,
              "tokenizer_assets_unchanged": True, "model_weights_downloaded": True,
              "transfer_backend": f"http-{args.workers}-workers" if args.http else "default",
              "model_weights_loaded": False, "training_started": False}
    Path("reports/sft-model-download.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "passed", "report": "reports/sft-model-download.json"}), flush=True)


if __name__ == "__main__":
    main()
