"""Strict backend for replaying a finalized trace without hardware."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from embodied_ai.ops.canonical import canonical_json_bytes
from embodied_ai.ops.errors import IntegrityError, StateConflict

from .safety import StopReason
from .trace import VerifiedTrace, verify_trace
from .types import ActionEnvelope, ActionReceipt, Observation


class ReplayBackend:
    def __init__(self, trace: str | Path | VerifiedTrace, *, strict: bool = True):
        self.trace = verify_trace(trace) if isinstance(trace, (str, Path)) else trace
        self.strict = strict
        self._connected = False
        self._enabled = False
        self._index = 0
        self._observation_read = False
        self._healthy = True

    @property
    def observation_features(self) -> Mapping[str, Any]:
        return self.trace.meta["observation_features"]

    @property
    def action_features(self) -> Mapping[str, Any]:
        return self.trace.meta["action_features"]

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def is_healthy(self) -> bool:
        return self._healthy

    @property
    def hardware_verified(self) -> bool:
        return False

    @property
    def consumed_records(self) -> int:
        return self._index

    def connect(self) -> None:
        if self._connected:
            raise StateConflict("replay backend is already connected")
        self._connected = True

    def disconnect(self) -> None:
        self._enabled = False
        self._connected = False

    def enable(self) -> None:
        if not self._connected:
            raise StateConflict("replay backend is not connected")
        self._enabled = True

    def disable(self) -> None:
        self._enabled = False

    def _current(self) -> dict[str, Any]:
        if self._index >= len(self.trace.records):
            raise StateConflict("replay trace is exhausted")
        return self.trace.records[self._index]

    def get_observation(self) -> Observation:
        if not self._connected:
            raise StateConflict("replay backend is not connected")
        self._observation_read = True
        return Observation.from_dict(self._current()["observation"])

    def send_action(self, action: ActionEnvelope) -> ActionReceipt:
        if not self._connected or not self._enabled:
            raise StateConflict("replay backend must be connected and enabled")
        current = self._current()
        if self.strict and not self._observation_read:
            raise IntegrityError(
                "strict replay requires observation read before action"
            )
        expected = ActionEnvelope.from_dict(current["action"])
        if self.strict and canonical_json_bytes(
            action.to_dict()
        ) != canonical_json_bytes(expected.to_dict()):
            self._healthy = False
            raise IntegrityError(
                f"strict replay action mismatch at record {self._index}"
            )
        receipt = ActionReceipt.from_dict(current["receipt"])
        receipt.validate(expected)
        self._index += 1
        self._observation_read = False
        return receipt

    def request_safe_stop(self, reason: StopReason) -> None:
        self._enabled = False

    def emergency_stop(self) -> None:
        self._enabled = False
