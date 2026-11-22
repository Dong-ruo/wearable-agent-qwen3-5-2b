"""Download only the public Qwen tokenizer assets, pinned and hash-checked."""
from __future__ import annotations

import hashlib
import json
import urllib.request
from pathlib import Path


def main():
    repository = "Qwen/Qwen3.5-2B"
    with urllib.request.urlopen(f"https://huggingface.co/api/models/{repository}?blobs=true", timeout=60) as response:
        info = json.load(response)
    revision = info["sha"]
    siblings = {item["rfilename"]: item for item in info["siblings"]}
    output = Path("models/Qwen3.5-2B")
    output.mkdir(parents=True, exist_ok=True)
    assets = []
    for name in ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt", "chat_template.jinja", "config.json"):
        path = output / name
        blob = siblings[name]
        with urllib.request.urlopen(f"https://huggingface.co/{repository}/resolve/{revision}/{name}", timeout=120) as response:
            content = response.read()
        sha256 = hashlib.sha256(content).hexdigest()
        if blob.get("lfs"):
            if sha256 != blob["lfs"]["sha256"]:
                raise ValueError("Upstream LFS hash mismatch: " + name)
        elif hashlib.sha1(b"blob " + str(len(content)).encode() + b"\0" + content).hexdigest() != blob["blobId"]:
            raise ValueError("Upstream Git blob mismatch: " + name)
        if blob.get("size") is not None and len(content) != blob["size"]:
            raise ValueError("Upstream size mismatch: " + name)
        temporary = path.with_suffix(path.suffix + ".part")
        temporary.write_bytes(content)
        temporary.replace(path)
        assets.append({"file": name, "sha256": sha256, "bytes": len(content), "upstream_hash_verified": True})
        print(json.dumps({"downloaded": name, "bytes": len(content)}, ensure_ascii=False), flush=True)
    report = {"repository": repository, "revision": revision, "files": assets, "model_weights_downloaded": False}
    Path("reports").mkdir(exist_ok=True)
    Path("reports/sft-tokenizer-download.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"revision": revision, "files": len(assets), "model_weights_downloaded": False}), flush=True)


if __name__ == "__main__":
    main()
