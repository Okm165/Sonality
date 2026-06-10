#!/usr/bin/env python3
"""Download the local llama.cpp GGUF model set into .models/.

The script intentionally uses only the Python standard library so model setup
does not depend on project extras or Hugging Face CLI state.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODELS_DIR = REPO_ROOT / ".models"
CHUNK_SIZE = 8 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class ModelSpec:
    key: str
    repo: str
    filename: str
    role: str
    sha256: str = ""

    @property
    def url(self) -> str:
        return f"https://huggingface.co/{self.repo}/resolve/main/{quote(self.filename)}"


MODELS: tuple[ModelSpec, ...] = (
    ModelSpec(
        key="llm",
        repo="unsloth/Qwen3.6-35B-A3B-GGUF",
        filename="Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf",
        role="Main LLM (22.4GB). MoE 35B with 3B active. Fits 24GB VRAM with 16K ctx + q4_0 KV cache",
    ),
    ModelSpec(
        key="embed",
        repo="Qwen/Qwen3-Embedding-4B-GGUF",
        filename="Qwen3-Embedding-4B-Q4_K_M.gguf",
        role="Dense embedding (2560d native, MRL to 32d). Runs on CPU (~2.3GB RAM)",
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "models",
        nargs="*",
        choices=[m.key for m in MODELS],
        help="Model keys to download. Defaults to all models when --all is set.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Download the full default model set.",
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=DEFAULT_MODELS_DIR,
        help="Destination directory. Defaults to .models at the repo root.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Redownload files that already exist.",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify SHA-256 after download. This reads every selected GGUF file.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List known model keys and exit.",
    )
    return parser.parse_args()


def selected_models(args: argparse.Namespace) -> list[ModelSpec]:
    if args.list:
        return []
    if args.all:
        return list(MODELS)
    if args.models:
        wanted = set(args.models)
        return [model for model in MODELS if model.key in wanted]
    raise SystemExit("Choose at least one model key or pass --all. Use --list to inspect options.")


def list_models() -> None:
    for model in MODELS:
        print(f"{model.key:10} {model.filename}")
        print(f"{'':10} repo: {model.repo}")
        print(f"{'':10} role: {model.role}")


def human_size(value: int) -> str:
    size = float(value)
    for unit in ("B", "KiB", "MiB", "GiB"):
        if size < 1024 or unit == "GiB":
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} GiB"


def download(model: ModelSpec, models_dir: Path, force: bool) -> Path:
    models_dir.mkdir(parents=True, exist_ok=True)
    destination = models_dir / model.filename
    partial = destination.with_suffix(destination.suffix + ".part")

    if force:
        destination.unlink(missing_ok=True)
        partial.unlink(missing_ok=True)
    elif destination.exists():
        print(f"ok: {destination} already exists ({human_size(destination.stat().st_size)})")
        return destination

    existing = partial.stat().st_size if partial.exists() else 0
    headers = {"User-Agent": "sonality-model-downloader/1.0"}
    if existing:
        headers["Range"] = f"bytes={existing}-"

    print(f"downloading {model.key}: {model.filename}")
    print(f"from {model.url}")

    request = Request(model.url, headers=headers)
    try:
        with urlopen(request, timeout=60) as response:
            if existing and response.status != 206:
                print("server did not accept resume; restarting download")
                partial.unlink(missing_ok=True)
                existing = 0
            total = _download_size(response, existing)
            write_mode = "ab" if existing else "wb"
            bytes_done = existing
            last_print = time.monotonic()
            with partial.open(write_mode) as handle:
                while True:
                    chunk = response.read(CHUNK_SIZE)
                    if not chunk:
                        break
                    handle.write(chunk)
                    bytes_done += len(chunk)
                    now = time.monotonic()
                    if now - last_print >= 2:
                        print(progress_line(bytes_done, total), flush=True)
                        last_print = now
    except (HTTPError, URLError, TimeoutError) as exc:
        raise SystemExit(f"download failed for {model.filename}: {exc}") from exc

    os.replace(partial, destination)
    print(f"done: {destination} ({human_size(destination.stat().st_size)})")
    return destination


def _download_size(response: Any, existing: int) -> int | None:
    headers = response.headers
    content_range = headers.get("Content-Range")
    if content_range and "/" in content_range:
        total = content_range.rsplit("/", 1)[1]
        return int(total) if total.isdigit() else None
    length = headers.get("Content-Length")
    if length and length.isdigit():
        return existing + int(length)
    return None


def progress_line(done: int, total: int | None) -> str:
    if not total:
        return f"  {human_size(done)}"
    percent = (done / total) * 100
    return f"  {human_size(done)} / {human_size(total)} ({percent:.1f}%)"


def verify_sha256(path: Path, expected: str) -> None:
    if not expected:
        print(f"skip verify: no SHA-256 recorded for {path.name}")
        return
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK_SIZE), b""):
            digest.update(chunk)
    actual = digest.hexdigest()
    if actual != expected:
        raise SystemExit(f"SHA-256 mismatch for {path.name}: expected {expected}, got {actual}")
    print(f"verified: {path.name}")


def main() -> int:
    args = parse_args()
    if args.list:
        list_models()
        return 0

    for model in selected_models(args):
        target = download(model, args.models_dir, args.force)
        if args.verify:
            verify_sha256(target, model.sha256)
    return 0


if __name__ == "__main__":
    sys.exit(main())
