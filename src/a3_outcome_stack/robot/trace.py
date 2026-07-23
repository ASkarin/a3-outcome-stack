"""Atomic, hash-bound trace-v1 writing and verification."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from a3_outcome_stack.ops.canonical import (
    atomic_write_bytes,
    atomic_write_json,
    canonical_json_bytes,
    load_json,
    resolve_relative,
    seal_document,
    sha256_file,
    verify_sealed_document,
)
from a3_outcome_stack.ops.errors import IntegrityError, StateConflict, ValidationError

from .types import ActionEnvelope, ActionReceipt, Observation, validate_feature_map

TRACE_SCHEMA = "a3-trace-v1"


def _fsync_directory(path: Path) -> None:
    if hasattr(os, "O_DIRECTORY"):
        descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def _validate_record(record: Mapping[str, Any], expected_index: int) -> None:
    if record.get("record_index") != expected_index:
        raise IntegrityError(f"trace record index mismatch at {expected_index}")
    observation = Observation.from_dict(record.get("observation", {}))
    action = ActionEnvelope.from_dict(record.get("action", {}))
    receipt = ActionReceipt.from_dict(record.get("receipt", {}))
    receipt.validate(action)
    next_clock = record.get("next_clock_ns")
    if (
        not isinstance(next_clock, int)
        or next_clock < observation.timestamps.receive_monotonic_ns
    ):
        raise IntegrityError(f"invalid next_clock_ns at record {expected_index}")
    transitions = record.get("safety_transitions")
    if not isinstance(transitions, list):
        raise IntegrityError(
            f"safety_transitions must be a list at record {expected_index}"
        )


@dataclass(frozen=True)
class VerifiedTrace:
    path: Path
    meta: dict[str, Any]
    records: tuple[dict[str, Any], ...]
    complete: dict[str, Any]


class TraceWriter:
    def __init__(self, destination: str | Path, meta: Mapping[str, Any]):
        self.destination = Path(destination)
        self.partial = self.destination.with_name(self.destination.name + ".partial")
        if self.destination.exists() or self.partial.exists():
            raise StateConflict(f"trace destination already exists: {self.destination}")
        self.partial.mkdir(parents=True)
        (self.partial / "blobs").mkdir()
        body = dict(meta)
        body["schema_version"] = TRACE_SCHEMA
        if (
            not isinstance(body.get("clock_domain_id"), str)
            or not body["clock_domain_id"]
        ):
            raise ValidationError("trace metadata requires clock_domain_id")
        validate_feature_map(body.get("observation_features", {}))
        validate_feature_map(body.get("action_features", {}))
        atomic_write_json(self.partial / "meta.json", body, immutable=True)
        self._records = (self.partial / "records.jsonl").open("xb")
        self._record_count = 0
        self._closed = False

    def append(self, record: Mapping[str, Any]) -> None:
        if self._closed:
            raise StateConflict("trace writer is closed")
        value = dict(record)
        _validate_record(value, self._record_count)
        self._records.write(canonical_json_bytes(value) + b"\n")
        self._records.flush()
        os.fsync(self._records.fileno())
        self._record_count += 1

    def add_blob(self, relative_path: str, data: bytes) -> dict[str, Any]:
        if self._closed:
            raise StateConflict("trace writer is closed")
        target = resolve_relative(self.partial / "blobs", relative_path)
        atomic_write_bytes(target, data, immutable=True)
        return {
            "path": f"blobs/{Path(relative_path).as_posix()}",
            "size_bytes": len(data),
            "sha256": sha256_file(target),
        }

    def finalize(self) -> dict[str, Any]:
        if self._closed:
            raise StateConflict("trace writer is already closed")
        if self._record_count == 0:
            raise ValidationError("cannot finalize an empty trace")
        self._records.flush()
        os.fsync(self._records.fileno())
        self._records.close()
        self._closed = True

        blobs = []
        for path in sorted((self.partial / "blobs").rglob("*")):
            if path.is_symlink():
                raise ValidationError(
                    f"trace blobs cannot contain symbolic links: {path}"
                )
            if path.is_file():
                blobs.append(
                    {
                        "path": path.relative_to(self.partial).as_posix(),
                        "size_bytes": path.stat().st_size,
                        "sha256": sha256_file(path),
                    }
                )
        complete = seal_document(
            {
                "schema_version": TRACE_SCHEMA,
                "record_count": self._record_count,
                "meta_sha256": sha256_file(self.partial / "meta.json"),
                "records_sha256": sha256_file(self.partial / "records.jsonl"),
                "blobs": blobs,
            },
            "trace_sha256",
        )
        atomic_write_json(self.partial / "complete.json", complete, immutable=True)
        _fsync_directory(self.partial)
        self.destination.parent.mkdir(parents=True, exist_ok=True)
        os.replace(self.partial, self.destination)
        _fsync_directory(self.destination.parent)
        return complete

    def close_incomplete(self) -> None:
        if not self._closed:
            self._records.close()
            self._closed = True


def verify_trace(path: str | Path) -> VerifiedTrace:
    root = Path(path)
    if root.name.endswith(".partial") or not root.is_dir():
        raise ValidationError(f"trace must be a finalized directory: {root}")
    expected_top = {"meta.json", "records.jsonl", "blobs", "complete.json"}
    actual_top = {child.name for child in root.iterdir()}
    if actual_top != expected_top:
        raise IntegrityError(f"trace top-level entries differ: {actual_top}")
    if any(child.is_symlink() for child in root.rglob("*")):
        raise IntegrityError("trace cannot contain symbolic links")

    meta = load_json(root / "meta.json")
    complete = load_json(root / "complete.json")
    if (
        meta.get("schema_version") != TRACE_SCHEMA
        or complete.get("schema_version") != TRACE_SCHEMA
    ):
        raise ValidationError("unsupported trace schema")
    validate_feature_map(meta.get("observation_features", {}))
    validate_feature_map(meta.get("action_features", {}))
    verify_sealed_document(complete, "trace_sha256")
    if sha256_file(root / "meta.json") != complete.get("meta_sha256"):
        raise IntegrityError("trace meta.json hash mismatch")
    if sha256_file(root / "records.jsonl") != complete.get("records_sha256"):
        raise IntegrityError("trace records.jsonl hash mismatch")

    expected_blobs = {item.get("path"): item for item in complete.get("blobs", [])}
    actual_blob_paths = {
        path.relative_to(root).as_posix()
        for path in (root / "blobs").rglob("*")
        if path.is_file()
    }
    if set(expected_blobs) != actual_blob_paths:
        raise IntegrityError("trace blob set mismatch")
    for relative, item in expected_blobs.items():
        target = resolve_relative(root, relative)
        if target.stat().st_size != item.get("size_bytes") or sha256_file(
            target
        ) != item.get("sha256"):
            raise IntegrityError(f"trace blob mismatch: {relative}")

    records: list[dict[str, Any]] = []
    raw_lines = (root / "records.jsonl").read_bytes().splitlines()
    for index, raw_line in enumerate(raw_lines):
        try:
            record = json.loads(raw_line.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise IntegrityError(f"invalid trace JSON at record {index}") from exc
        if canonical_json_bytes(record) != raw_line:
            raise IntegrityError(f"non-canonical trace JSON at record {index}")
        _validate_record(record, index)
        for camera_name, reference in (
            record["observation"].get("camera_refs", {}).items()
        ):
            relative = reference.get("path")
            if relative not in expected_blobs:
                raise IntegrityError(
                    f"camera reference has no sealed blob: {camera_name}"
                )
            expected = expected_blobs[relative]
            if reference.get("sha256") != expected.get("sha256") or reference.get(
                "size_bytes"
            ) != expected.get("size_bytes"):
                raise IntegrityError(
                    f"camera reference metadata mismatch: {camera_name}"
                )
        records.append(record)
    if len(records) != complete.get("record_count") or not records:
        raise IntegrityError("trace record count mismatch")
    return VerifiedTrace(root.resolve(), meta, tuple(records), complete)
