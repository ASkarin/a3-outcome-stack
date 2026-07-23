"""Small atomic run-state machine; terminal states cannot be reopened."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .canonical import atomic_write_json, load_json, utc_now
from .errors import StateConflict, ValidationError

TRANSITIONS = {
    "registered": {"running"},
    "running": {"completed", "stopped", "failed"},
    "completed": set(),
    "stopped": set(),
    "failed": set(),
}


def create_run_state(path: str | Path, experiment_id: str, attempt_id: str) -> dict[str, Any]:
    if Path(path).exists():
        raise StateConflict(f"run state already exists: {path}")
    now = utc_now()
    state = {
        "schema_version": "1.0",
        "experiment_id": experiment_id,
        "attempt_id": attempt_id,
        "status": "registered",
        "updated_at_utc": now,
        "history": [{"from": None, "to": "registered", "at_utc": now}],
    }
    atomic_write_json(path, state)
    return state


def transition_run_state(path: str | Path, new_status: str) -> dict[str, Any]:
    state = load_json(path)
    current = state.get("status")
    if current not in TRANSITIONS:
        raise ValidationError(f"unknown current run status: {current}")
    if new_status not in TRANSITIONS[current]:
        raise StateConflict(f"illegal run transition: {current} -> {new_status}")
    now = utc_now()
    state["status"] = new_status
    state["updated_at_utc"] = now
    state.setdefault("history", []).append({"from": current, "to": new_status, "at_utc": now})
    atomic_write_json(path, state)
    return state

