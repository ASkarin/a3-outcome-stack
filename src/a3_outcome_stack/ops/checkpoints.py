"""Atomic, identity-bound PyTorch checkpoints with exact-resume RNG state."""

from __future__ import annotations

import os
import random
import uuid
from pathlib import Path
from typing import Any

from .canonical import atomic_write_json, load_json, sha256_file, utc_now, validate_sha256
from .errors import IntegrityError, StateConflict, ValidationError

IDENTITY_FIELDS = {
    "experiment_id",
    "attempt_id",
    "git_commit",
    "config_sha256",
    "data_version",
    "data_manifest_sha256",
    "seed",
}

EXACT_STATE_FIELDS = {
    "model_state_dict",
    "optimizer_state_dict",
    "scheduler_state_dict",
    "scaler_state_dict",
    "global_step",
    "sampler_state",
}


def _torch():
    try:
        import torch
    except ImportError as exc:
        raise ValidationError("PyTorch is required for checkpoint operations") from exc
    return torch


def _numpy():
    try:
        import numpy as np
    except ImportError as exc:
        raise ValidationError("NumPy is required for exact RNG recovery") from exc
    return np


def _validate_identity(identity: dict[str, Any]) -> None:
    missing = sorted(IDENTITY_FIELDS - set(identity))
    if missing:
        raise ValidationError(f"missing checkpoint identity fields: {missing}")
    for field in ("config_sha256", "data_manifest_sha256"):
        validate_sha256(identity[field], field)
    if not isinstance(identity["seed"], int):
        raise ValidationError("checkpoint seed must be an integer")


def _capture_rng() -> dict[str, Any]:
    torch = _torch()
    np = _numpy()
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def _restore_rng(state: dict[str, Any]) -> None:
    torch = _torch()
    np = _numpy()
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if state.get("torch_cuda"):
        if not torch.cuda.is_available():
            raise StateConflict("checkpoint contains CUDA RNG state but CUDA is unavailable")
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def verify_checkpoint(checkpoint_path: str | Path, metadata_path: str | Path | None = None) -> dict[str, Any]:
    checkpoint = Path(checkpoint_path)
    metadata_file = Path(metadata_path) if metadata_path else checkpoint.with_suffix(checkpoint.suffix + ".metadata.json")
    metadata = load_json(metadata_file)
    validate_sha256(metadata.get("checkpoint_sha256"), "checkpoint_sha256")
    actual = sha256_file(checkpoint)
    if actual != metadata["checkpoint_sha256"]:
        raise IntegrityError(f"checkpoint hash mismatch: expected {metadata['checkpoint_sha256']}, got {actual}")
    payload = _torch().load(checkpoint, map_location="cpu", weights_only=False)
    if payload.get("identity") != metadata.get("identity"):
        raise IntegrityError("checkpoint payload identity differs from metadata")
    _validate_identity(payload.get("identity", {}))
    if payload.get("state", {}).get("global_step") != metadata.get("global_step"):
        raise IntegrityError("checkpoint step differs from metadata")
    return metadata


class CheckpointManager:
    def __init__(self, directory: str | Path, identity: dict[str, Any]):
        _validate_identity(identity)
        self.directory = Path(directory)
        self.identity = dict(identity)

    @property
    def latest_path(self) -> Path:
        return self.directory / "latest.json"

    def save(self, state: dict[str, Any], *, exact_resume: bool = True) -> dict[str, Any]:
        missing = sorted(EXACT_STATE_FIELDS - set(state))
        if missing:
            raise ValidationError(f"missing checkpoint state fields: {missing}")
        if not isinstance(state["global_step"], int) or state["global_step"] < 0:
            raise ValidationError("global_step must be a non-negative integer")
        if exact_resume and state.get("sampler_state") is None:
            raise ValidationError("exact resume requires sampler_state; refusing silent best-effort recovery")

        self.directory.mkdir(parents=True, exist_ok=True)
        checkpoint = self.directory / f"step-{state['global_step']:012d}.pt"
        if checkpoint.exists():
            raise StateConflict(f"checkpoint step already exists: {checkpoint}")
        temporary = checkpoint.with_name(f".{checkpoint.name}.{uuid.uuid4().hex}.tmp")
        payload = {
            "schema_version": "1.0",
            "identity": self.identity,
            "exact_resume": exact_resume,
            "state": state,
            "rng_state": _capture_rng(),
        }
        torch = _torch()
        try:
            torch.save(payload, temporary)
            with temporary.open("rb") as handle:
                os.fsync(handle.fileno())
            os.replace(temporary, checkpoint)
        finally:
            if temporary.exists():
                temporary.unlink()

        metadata = {
            "schema_version": "1.0",
            "checkpoint": checkpoint.name,
            "checkpoint_sha256": sha256_file(checkpoint),
            "identity": self.identity,
            "global_step": state["global_step"],
            "exact_resume": exact_resume,
            "created_at_utc": utc_now(),
        }
        metadata_path = checkpoint.with_suffix(checkpoint.suffix + ".metadata.json")
        atomic_write_json(metadata_path, metadata, immutable=True)
        atomic_write_json(
            self.latest_path,
            {
                "checkpoint": checkpoint.name,
                "metadata": metadata_path.name,
                "checkpoint_sha256": metadata["checkpoint_sha256"],
            },
        )
        return metadata

    def load_latest(self, *, restore_rng: bool = True) -> dict[str, Any]:
        pointer = load_json(self.latest_path)
        checkpoint = self.directory / pointer.get("checkpoint", "")
        metadata_path = self.directory / pointer.get("metadata", "")
        metadata = verify_checkpoint(checkpoint, metadata_path)
        if metadata.get("identity") != self.identity:
            raise StateConflict("checkpoint identity mismatch; use a new experiment and explicit warm_start")
        if pointer.get("checkpoint_sha256") != metadata.get("checkpoint_sha256"):
            raise IntegrityError("latest pointer hash differs from checkpoint metadata")
        payload = _torch().load(checkpoint, map_location="cpu", weights_only=False)
        if payload.get("exact_resume") and payload.get("state", {}).get("sampler_state") is None:
            raise IntegrityError("checkpoint claims exact resume but has no sampler_state")
        if restore_rng:
            _restore_rng(payload["rng_state"])
        return payload["state"]


def load_warm_start(checkpoint_path: str | Path, expected_sha256: str) -> dict[str, Any]:
    """Load model weights only; never restores optimizer, progress, or RNG."""

    validate_sha256(expected_sha256, "expected_sha256")
    actual = sha256_file(checkpoint_path)
    if actual != expected_sha256:
        raise IntegrityError(f"warm-start hash mismatch: expected {expected_sha256}, got {actual}")
    payload = _torch().load(checkpoint_path, map_location="cpu", weights_only=False)
    model_state = payload.get("state", {}).get("model_state_dict")
    if model_state is None:
        raise ValidationError("checkpoint has no model_state_dict")
    return {"model_state_dict": model_state, "parent_checkpoint_sha256": actual, "mode": "warm_start"}

