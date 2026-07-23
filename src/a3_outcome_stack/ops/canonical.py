"""Canonical JSON, SHA-256, safe path resolution, and atomic writes."""

from __future__ import annotations

import hashlib
import json
import math
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .errors import IntegrityError, StateConflict, ValidationError

HASH_PREFIX = "sha256:"


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _validate_json(value: Any, path: str = "$") -> None:
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValidationError(f"non-finite JSON number at {path}")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _validate_json(item, f"{path}[{index}]")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValidationError(f"non-string JSON key at {path}")
            _validate_json(item, f"{path}.{key}")
        return
    raise ValidationError(f"unsupported JSON value at {path}: {type(value).__name__}")


def canonical_json_bytes(value: Any) -> bytes:
    """Serialize with the project's canonical_json_v1 contract."""

    _validate_json(value)
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def sha256_bytes(data: bytes) -> str:
    return HASH_PREFIX + hashlib.sha256(data).hexdigest()


def sha256_file(path: str | Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return HASH_PREFIX + digest.hexdigest()


def validate_sha256(value: str, field: str = "sha256") -> None:
    if not isinstance(value, str) or len(value) != len(HASH_PREFIX) + 64 or not value.startswith(HASH_PREFIX):
        raise ValidationError(f"{field} must be sha256:<64 lowercase hex characters>")
    try:
        int(value[len(HASH_PREFIX) :], 16)
    except ValueError as exc:
        raise ValidationError(f"{field} is not hexadecimal") from exc
    if value != value.lower():
        raise ValidationError(f"{field} must use lowercase hexadecimal")


def load_json(path: str | Path) -> Any:
    try:
        with Path(path).open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValidationError(f"cannot read JSON {path}: {exc}") from exc


def atomic_write_bytes(path: str | Path, data: bytes, *, immutable: bool = False) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        existing = destination.read_bytes()
        if immutable:
            if existing == data:
                return
            raise StateConflict(f"immutable file already exists with different content: {destination}")

    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
        if hasattr(os, "O_DIRECTORY"):
            directory_fd = os.open(destination.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_write_json(path: str | Path, value: Any, *, immutable: bool = False) -> None:
    atomic_write_bytes(path, canonical_json_bytes(value) + b"\n", immutable=immutable)


def seal_document(value: dict[str, Any], hash_field: str) -> dict[str, Any]:
    if hash_field in value:
        raise ValidationError(f"input must not contain generated field {hash_field}")
    sealed = dict(value)
    sealed[hash_field] = sha256_bytes(canonical_json_bytes(value))
    return sealed


def verify_sealed_document(value: dict[str, Any], hash_field: str) -> None:
    recorded = value.get(hash_field)
    validate_sha256(recorded, hash_field)
    body = dict(value)
    del body[hash_field]
    actual = sha256_bytes(canonical_json_bytes(body))
    if actual != recorded:
        raise IntegrityError(f"{hash_field} mismatch: expected {recorded}, got {actual}")


def resolve_relative(root: str | Path, relative: str) -> Path:
    if not isinstance(relative, str) or not relative or "\\" in relative:
        raise ValidationError("relative paths must be non-empty POSIX paths")
    relative_path = Path(relative)
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise ValidationError(f"path escapes root: {relative}")
    root_path = Path(root).resolve()
    candidate = (root_path / relative_path).resolve()
    try:
        candidate.relative_to(root_path)
    except ValueError as exc:
        raise ValidationError(f"path escapes root: {relative}") from exc
    return candidate


def directory_entries(root: str | Path) -> list[dict[str, Any]]:
    root_path = Path(root).resolve()
    if not root_path.is_dir():
        raise ValidationError(f"not a directory: {root_path}")

    entries: list[dict[str, Any]] = []
    for current, directories, files in os.walk(root_path, followlinks=False):
        current_path = Path(current)
        for name in list(directories):
            candidate = current_path / name
            if candidate.is_symlink():
                raise ValidationError(f"symbolic links are not allowed: {candidate}")
        for name in files:
            candidate = current_path / name
            if candidate.is_symlink():
                raise ValidationError(f"symbolic links are not allowed: {candidate}")
            relative = candidate.relative_to(root_path).as_posix()
            entries.append(
                {
                    "path": relative,
                    "size_bytes": candidate.stat().st_size,
                    "sha256": sha256_file(candidate),
                }
            )
    entries.sort(key=lambda item: item["path"])
    return entries


def directory_content_sha256(entries: list[dict[str, Any]]) -> str:
    return sha256_bytes(canonical_json_bytes(entries))


def verify_directory_entries(root: str | Path, expected: list[dict[str, Any]]) -> None:
    actual = directory_entries(root)
    if actual != expected:
        raise IntegrityError(f"directory content differs from manifest: {Path(root)}")
