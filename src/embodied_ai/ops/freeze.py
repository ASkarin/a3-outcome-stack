"""Sealed final-evaluation identity."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from .canonical import atomic_write_json, load_json, seal_document, validate_sha256, verify_sealed_document
from .errors import ValidationError

REQUIRED_FREEZE_FIELDS = {
    "freeze_id",
    "created_at_utc",
    "preregistration",
    "split_protocol",
    "code",
    "environment",
    "configs",
    "seeds",
    "data",
    "checkpoints",
    "normalization",
    "calibration_assets",
    "camera_assets",
    "action_chunk",
    "thresholds",
    "evaluation_conditions",
    "episode_budget",
    "run_order",
    "randomization_seed",
    "metrics",
    "ci_method",
    "failure_taxonomy",
    "exclusion_rules",
    "final_test_manifest_sha256",
}


def _validate_hash_fields(value: Any, path: str = "$") -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            next_path = f"{path}.{key}"
            if key.endswith("sha256"):
                validate_sha256(item, next_path)
            else:
                _validate_hash_fields(item, next_path)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _validate_hash_fields(item, f"{path}[{index}]")


def validate_freeze_body(body: dict[str, Any]) -> None:
    missing = sorted(REQUIRED_FREEZE_FIELDS - set(body))
    if missing:
        raise ValidationError(f"missing final freeze fields: {missing}")
    if re.fullmatch(r"FRZ-[0-9]{8}-[A-Z0-9-]+", body.get("freeze_id", "")) is None:
        raise ValidationError("freeze_id must match FRZ-YYYYMMDD-<LABEL>")
    _validate_hash_fields(body)


def create_freeze(source: dict[str, Any], output: str | Path) -> dict[str, Any]:
    if "freeze_sha256" in source:
        raise ValidationError("input must not contain generated freeze_sha256")
    validate_freeze_body(source)
    sealed = seal_document(dict(source), "freeze_sha256")
    atomic_write_json(output, sealed, immutable=True)
    return sealed


def verify_freeze(manifest: dict[str, Any]) -> None:
    verify_sealed_document(manifest, "freeze_sha256")
    body = dict(manifest)
    del body["freeze_sha256"]
    validate_freeze_body(body)


def verify_freeze_file(path: str | Path) -> dict[str, Any]:
    manifest = load_json(path)
    verify_freeze(manifest)
    return manifest

