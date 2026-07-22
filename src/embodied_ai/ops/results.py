"""Immutable per-attempt summaries and generated JSONL index."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .canonical import (
    atomic_write_bytes,
    atomic_write_json,
    canonical_json_bytes,
    load_json,
    seal_document,
    validate_sha256,
    verify_sealed_document,
)
from .errors import IntegrityError, ValidationError

TERMINAL_STATUSES = {"completed", "failed", "stopped"}
REQUIRED_FIELDS = {
    "experiment_id",
    "attempt_id",
    "status",
    "started_at_utc",
    "ended_at_utc",
    "git_commit",
    "config_sha256",
    "data_version",
    "data_manifest_sha256",
    "preregistration_id",
    "preregistration_sha256",
    "seed",
    "conclusion",
}


def _validate_summary_body(summary: dict[str, Any]) -> None:
    missing = sorted(REQUIRED_FIELDS - set(summary))
    if missing:
        raise ValidationError(f"missing result fields: {missing}")
    status = summary.get("status")
    if status not in TERMINAL_STATUSES:
        raise ValidationError(f"result status must be one of {sorted(TERMINAL_STATUSES)}")
    for field in ("config_sha256", "data_manifest_sha256", "preregistration_sha256"):
        validate_sha256(summary.get(field), field)
    for field in ("checkpoint_sha256", "artifact_manifest_sha256", "freeze_sha256"):
        if summary.get(field) is not None:
            validate_sha256(summary[field], field)
    if status == "completed":
        metric = summary.get("primary_metric")
        if not isinstance(metric, dict) or not isinstance(metric.get("name"), str) or "value" not in metric:
            raise ValidationError("completed results require primary_metric with name and value")
    elif not isinstance(summary.get("stop_reason"), str) or not summary["stop_reason"].strip():
        raise ValidationError("failed and stopped results require stop_reason")


def materialize_summary(source: dict[str, Any]) -> dict[str, Any]:
    if "summary_sha256" in source:
        raise ValidationError("input must not contain generated summary_sha256")
    _validate_summary_body(source)
    return seal_document(dict(source), "summary_sha256")


def verify_summary(summary: dict[str, Any]) -> None:
    verify_sealed_document(summary, "summary_sha256")
    body = dict(summary)
    del body["summary_sha256"]
    _validate_summary_body(body)


def finalize_result(source: dict[str, Any], summaries_dir: str | Path) -> Path:
    summary = materialize_summary(source)
    destination = (
        Path(summaries_dir)
        / summary["experiment_id"]
        / summary["attempt_id"]
        / "summary.json"
    )
    atomic_write_json(destination, summary, immutable=True)
    return destination


def _collect_summaries(summaries_dir: str | Path) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    seen_attempts: set[str] = set()
    for path in sorted(Path(summaries_dir).glob("**/summary.json")):
        summary = load_json(path)
        verify_summary(summary)
        attempt = summary["attempt_id"]
        if attempt in seen_attempts:
            raise IntegrityError(f"duplicate attempt_id in summaries: {attempt}")
        seen_attempts.add(attempt)
        summaries.append(summary)
    summaries.sort(key=lambda item: (item["experiment_id"], item["attempt_id"]))
    return summaries


def _index_bytes(summaries: list[dict[str, Any]]) -> bytes:
    if not summaries:
        return b""
    return b"\n".join(canonical_json_bytes(summary) for summary in summaries) + b"\n"


def rebuild_results_index(summaries_dir: str | Path, index_path: str | Path) -> list[dict[str, Any]]:
    summaries = _collect_summaries(summaries_dir)
    atomic_write_bytes(index_path, _index_bytes(summaries))
    return summaries


def verify_results_index(summaries_dir: str | Path, index_path: str | Path) -> None:
    summaries = _collect_summaries(summaries_dir)
    expected = _index_bytes(summaries)
    try:
        actual = Path(index_path).read_bytes()
    except OSError as exc:
        raise ValidationError(f"cannot read result index: {exc}") from exc
    if actual != expected:
        raise IntegrityError("result index does not match immutable summaries")

