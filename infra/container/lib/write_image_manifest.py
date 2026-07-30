from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import subprocess
from pathlib import Path

BASE_IMAGE = (
    "nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04@"
    "sha256:ad6d59a3bbf3e82c1c849c9ac09cfc2a3e0bbb8655042fd899be6681b3fe2a85"
)


def package_version(name: str) -> str:
    return importlib.metadata.version(name)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--python-version", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--uv-lock-sha256", required=True)
    args = parser.parse_args()

    uv_version = subprocess.run(
        ["uv", "--version"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    payload = {
        "schema_version": 1,
        "base_image": BASE_IMAGE,
        "source_commit": args.source_commit,
        "uv_lock_sha256": args.uv_lock_sha256,
        "python": {
            "requested": args.python_version,
            "runtime": platform.python_version(),
        },
        "packages": {
            "lerobot": package_version("lerobot"),
            "tensorboard": package_version("tensorboard"),
            "torch": package_version("torch"),
            "torchvision": package_version("torchvision"),
            "uv": uv_version.removeprefix("uv "),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
