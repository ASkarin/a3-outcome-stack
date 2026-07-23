"""Immutable experiment specifications and generated registry."""

from __future__ import annotations

import csv
import io
import re
from pathlib import Path
from typing import Any

from .canonical import (
    atomic_write_bytes,
    atomic_write_json,
    canonical_json_bytes,
    load_json,
    seal_document,
    sha256_bytes,
    utc_now,
    validate_sha256,
    verify_sealed_document,
)
from .errors import IntegrityError, StateConflict, ValidationError
from .manifests import validate_data_version
from .results import verify_summary

REGISTRY_FIELDS = [
    "experiment_id",
    "created_at_utc",
    "status",
    "stage",
    "preregistration_id",
    "preregistration_sha256",
    "hypothesis",
    "task",
    "policy",
    "data_version",
    "data_manifest_sha256",
    "git_commit",
    "config_path",
    "config_sha256",
    "seed",
    "hardware",
    "primary_metric",
    "result",
    "conclusion",
    "stop_reason",
]

REQUIRED_SPEC_FIELDS = {
    "stage",
    "preregistration_id",
    "preregistration_sha256",
    "hypothesis",
    "task",
    "policy",
    "data_version",
    "data_manifest_sha256",
    "git_commit",
    "config_path",
    "config_sha256",
    "seed",
    "hardware",
    "max_steps",
    "max_wallclock_seconds",
    "eval_interval_steps",
    "patience_evaluations",
    "stop_metric",
}


def _slug(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "-", value).strip("-").upper()
    if not slug:
        raise ValidationError("task and policy must contain letters or digits")
    return slug[:24]


def _validate_spec(spec: dict[str, Any]) -> None:
    generated = {"experiment_id", "spec_sha256", "created_at_utc", "status"}
    overlap = generated.intersection(spec)
    if overlap:
        raise ValidationError(f"input contains generated fields: {sorted(overlap)}")
    missing = sorted(REQUIRED_SPEC_FIELDS - set(spec))
    if missing:
        raise ValidationError(f"missing experiment fields: {missing}")
    for field in ("stage", "preregistration_id", "hypothesis", "task", "policy", "hardware", "stop_metric"):
        if not isinstance(spec[field], str) or not spec[field].strip():
            raise ValidationError(f"{field} must be a non-empty string")
    validate_sha256(spec["preregistration_sha256"], "preregistration_sha256")
    validate_sha256(spec["data_manifest_sha256"], "data_manifest_sha256")
    validate_sha256(spec["config_sha256"], "config_sha256")
    validate_data_version(spec["data_version"])
    if not re.fullmatch(r"[0-9a-f]{40}", spec["git_commit"]):
        raise ValidationError("git_commit must be a full lowercase 40-character SHA-1")
    config_path = Path(spec["config_path"])
    if config_path.is_absolute() or ".." in config_path.parts or "\\" in spec["config_path"]:
        raise ValidationError("config_path must be a repository-relative POSIX path")
    if not isinstance(spec["seed"], int):
        raise ValidationError("seed must be an integer")
    for field in ("max_steps", "max_wallclock_seconds", "eval_interval_steps", "patience_evaluations"):
        if not isinstance(spec[field], int) or spec[field] <= 0:
            raise ValidationError(f"{field} must be a positive integer")


def materialize_experiment_spec(spec: dict[str, Any]) -> dict[str, Any]:
    _validate_spec(spec)
    stable = dict(spec)
    spec_hash = sha256_bytes(canonical_json_bytes(stable))
    experiment_id = f"EXP-{_slug(spec['task'])}-{_slug(spec['policy'])}-{spec_hash[-12:].upper()}"
    body = {
        "schema_version": "1.0",
        "experiment_id": experiment_id,
        "created_at_utc": utc_now(),
        "status": "registered",
        **stable,
    }
    sealed = seal_document(body, "spec_sha256")
    # The sealed document hash includes generated timestamps. Identity remains the stable hash suffix.
    sealed["identity_sha256"] = spec_hash
    return sealed


def verify_experiment_spec(spec: dict[str, Any]) -> None:
    identity_hash = spec.get("identity_sha256")
    validate_sha256(identity_hash, "identity_sha256")
    document = dict(spec)
    document.pop("identity_sha256", None)
    verify_sealed_document(document, "spec_sha256")
    stable = {
        key: value
        for key, value in spec.items()
        if key not in {"schema_version", "experiment_id", "created_at_utc", "status", "spec_sha256", "identity_sha256"}
    }
    _validate_spec(stable)
    actual_identity = sha256_bytes(canonical_json_bytes(stable))
    if actual_identity != identity_hash:
        raise StateConflict("experiment identity does not match immutable spec fields")
    expected_id = f"EXP-{_slug(stable['task'])}-{_slug(stable['policy'])}-{identity_hash[-12:].upper()}"
    if spec.get("experiment_id") != expected_id:
        raise StateConflict("experiment_id does not match identity hash")


def _summary_map(summaries_dir: str | Path | None) -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    if summaries_dir is None or not Path(summaries_dir).exists():
        return latest
    for path in sorted(Path(summaries_dir).glob("**/summary.json")):
        summary = load_json(path)
        verify_summary(summary)
        experiment_id = summary["experiment_id"]
        previous = latest.get(experiment_id)
        ordering = (summary["ended_at_utc"], summary["attempt_id"])
        if previous is None or ordering > (previous["ended_at_utc"], previous["attempt_id"]):
            latest[experiment_id] = summary
    return latest


def _registry_bytes(
    specs: list[dict[str, Any]], summaries: dict[str, dict[str, Any]] | None = None
) -> bytes:
    summaries = summaries or {}
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=REGISTRY_FIELDS, lineterminator="\n")
    writer.writeheader()
    for spec in sorted(specs, key=lambda item: item["experiment_id"]):
        row = {field: spec.get(field, "") for field in REGISTRY_FIELDS}
        summary = summaries.get(spec["experiment_id"])
        if summary is None:
            row.update({"primary_metric": "", "result": "", "conclusion": "", "stop_reason": ""})
        else:
            metric = summary.get("primary_metric") or {}
            value = metric.get("value", "")
            if isinstance(value, (dict, list)):
                value = canonical_json_bytes(value).decode("utf-8")
            row.update(
                {
                    "status": summary["status"],
                    "primary_metric": metric.get("name", ""),
                    "result": value,
                    "conclusion": summary.get("conclusion", ""),
                    "stop_reason": summary.get("stop_reason", ""),
                }
            )
        writer.writerow(row)
    return output.getvalue().encode("utf-8")


def rebuild_registry(
    specs_dir: str | Path,
    registry_path: str | Path,
    summaries_dir: str | Path | None = None,
) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    for path in sorted(Path(specs_dir).glob("*.json")):
        spec = load_json(path)
        verify_experiment_spec(spec)
        specs.append(spec)
    summaries = _summary_map(summaries_dir)
    unknown = sorted(set(summaries) - {spec["experiment_id"] for spec in specs})
    if unknown:
        raise IntegrityError(f"result summaries reference unregistered experiments: {unknown}")
    atomic_write_bytes(registry_path, _registry_bytes(specs, summaries))
    return specs


def verify_registry(
    specs_dir: str | Path,
    registry_path: str | Path,
    summaries_dir: str | Path | None = None,
) -> None:
    specs: list[dict[str, Any]] = []
    for path in sorted(Path(specs_dir).glob("*.json")):
        spec = load_json(path)
        verify_experiment_spec(spec)
        specs.append(spec)
    summaries = _summary_map(summaries_dir)
    unknown = sorted(set(summaries) - {spec["experiment_id"] for spec in specs})
    if unknown:
        raise IntegrityError(f"result summaries reference unregistered experiments: {unknown}")
    expected = _registry_bytes(specs, summaries)
    try:
        actual = Path(registry_path).read_bytes()
    except OSError as exc:
        raise ValidationError(f"cannot read experiment registry: {exc}") from exc
    if actual != expected:
        raise IntegrityError("experiment registry does not match immutable specs and summaries")


def register_experiment(
    source: dict[str, Any],
    *,
    specs_dir: str | Path,
    registry_path: str | Path,
    summaries_dir: str | Path | None = None,
) -> dict[str, Any]:
    materialized = materialize_experiment_spec(source)
    destination = Path(specs_dir) / f"{materialized['experiment_id']}.json"
    if destination.exists():
        existing = load_json(destination)
        verify_experiment_spec(existing)
        if existing.get("identity_sha256") != materialized.get("identity_sha256"):
            raise StateConflict(f"experiment ID collision: {destination}")
        materialized = existing
    else:
        atomic_write_json(destination, materialized, immutable=True)
    rebuild_registry(specs_dir, registry_path, summaries_dir)
    return materialized


def attempt_id(experiment_id: str, number: int) -> str:
    if not re.fullmatch(r"EXP-[A-Z0-9-]+-[A-F0-9]{12}", experiment_id):
        raise ValidationError("invalid experiment_id")
    if not isinstance(number, int) or number <= 0 or number > 999:
        raise ValidationError("attempt number must be between 1 and 999")
    return f"{experiment_id}-A{number:03d}"
