"""Verification of the pinned official EDULITE_A3 source contract."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path
from typing import Any

from embodied_ai.ops.canonical import (
    load_json,
    resolve_relative,
    sha256_file,
    verify_sealed_document,
)
from embodied_ai.ops.errors import IntegrityError, ValidationError


def verify_upstream_lock(
    lock_path: str | Path, checkout: str | Path | None = None
) -> dict[str, Any]:
    lock = load_json(lock_path)
    verify_sealed_document(lock, "lock_sha256")
    commit = lock.get("commit")
    if not isinstance(commit, str) or not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValidationError(
            "upstream lock commit must be 40 lowercase hex characters"
        )
    files = lock.get("files")
    if not isinstance(files, list) or not files:
        raise ValidationError("upstream lock must contain selected files")
    paths = [item.get("path") for item in files]
    if len(paths) != len(set(paths)):
        raise ValidationError("upstream lock contains duplicate paths")

    verified = False
    checkout_path: Path | None = None
    if checkout is not None:
        checkout_path = Path(checkout).resolve()
        if not checkout_path.is_dir():
            raise ValidationError(
                f"upstream checkout is not a directory: {checkout_path}"
            )
        git_dir = checkout_path / ".git"
        if git_dir.exists():
            try:
                actual_commit = subprocess.check_output(
                    ["git", "-C", str(checkout_path), "rev-parse", "HEAD"], text=True
                ).strip()
            except (OSError, subprocess.CalledProcessError) as exc:
                raise ValidationError("cannot inspect upstream Git checkout") from exc
            if actual_commit != commit:
                raise IntegrityError(
                    f"upstream commit mismatch: expected {commit}, got {actual_commit}"
                )
        for item in files:
            target = resolve_relative(checkout_path, item["path"])
            if not target.is_file():
                raise IntegrityError(f"missing locked upstream file: {item['path']}")
            actual = sha256_file(target)
            if actual != item.get("sha256"):
                raise IntegrityError(
                    f"upstream file hash mismatch for {item['path']}: expected {item.get('sha256')}, got {actual}"
                )
        verified = True
    return {
        "status": "ok",
        "repository": lock.get("repository"),
        "commit": commit,
        "selected_files": len(files),
        "source_files_verified": verified,
        "checkout": str(checkout_path) if checkout_path else None,
        "lock_sha256": lock["lock_sha256"],
    }
