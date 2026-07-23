"""Latched safety state machine and command watchdog."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from a3_outcome_stack.ops.errors import StateConflict, ValidationError

from .clock import Clock


class SafetyState(str, Enum):
    DISCONNECTED = "DISCONNECTED"
    DISABLED = "DISABLED"
    READY = "READY"
    ACTIVE = "ACTIVE"
    SAFE_STOP = "SAFE_STOP"
    FAULT = "FAULT"
    E_STOP = "E_STOP"


class StopReason(str, Enum):
    OPERATOR_REQUEST = "operator_request"
    WATCHDOG_TIMEOUT = "watchdog_timeout"
    STALE_ACTION = "stale_action"
    OUT_OF_ORDER = "out_of_order"
    NONFINITE = "nonfinite"
    SHAPE_MISMATCH = "shape_mismatch"
    LIMIT_VIOLATION = "limit_violation"
    CLOCK_REGRESSION = "clock_regression"
    CLOCK_DOMAIN_MISMATCH = "clock_domain_mismatch"
    FEEDBACK_TIMEOUT = "feedback_timeout"
    BACKEND_EXCEPTION = "backend_exception"
    DEVICE_FAULT = "device_fault"
    E_STOP = "e_stop"
    REJECTED_ACTION = "rejected_action"


_ALLOWED = {
    SafetyState.DISCONNECTED: {SafetyState.DISABLED},
    SafetyState.DISABLED: {
        SafetyState.READY,
        SafetyState.DISCONNECTED,
        SafetyState.SAFE_STOP,
        SafetyState.FAULT,
        SafetyState.E_STOP,
    },
    SafetyState.READY: {
        SafetyState.ACTIVE,
        SafetyState.DISABLED,
        SafetyState.DISCONNECTED,
        SafetyState.SAFE_STOP,
        SafetyState.FAULT,
        SafetyState.E_STOP,
    },
    SafetyState.ACTIVE: {
        SafetyState.READY,
        SafetyState.DISABLED,
        SafetyState.DISCONNECTED,
        SafetyState.SAFE_STOP,
        SafetyState.FAULT,
        SafetyState.E_STOP,
    },
    SafetyState.SAFE_STOP: {
        SafetyState.DISABLED,
        SafetyState.DISCONNECTED,
        SafetyState.FAULT,
        SafetyState.E_STOP,
    },
    SafetyState.FAULT: {
        SafetyState.DISABLED,
        SafetyState.DISCONNECTED,
        SafetyState.E_STOP,
    },
    SafetyState.E_STOP: {SafetyState.DISABLED, SafetyState.DISCONNECTED},
}


class SafetySupervisor:
    def __init__(self, clock: Clock):
        self._clock = clock
        self._state = SafetyState.DISCONNECTED
        self._transitions: list[dict[str, Any]] = []
        self._last_reason: str | None = None
        self._last_transition_ns: int | None = None

    @property
    def state(self) -> SafetyState:
        return self._state

    @property
    def last_reason(self) -> str | None:
        return self._last_reason

    @property
    def transitions(self) -> list[dict[str, Any]]:
        return [dict(item) for item in self._transitions]

    def transition(self, target: SafetyState, reason: str) -> None:
        if target not in _ALLOWED[self._state]:
            raise StateConflict(
                f"illegal safety transition {self._state.value} -> {target.value}"
            )
        source = self._state
        self._state = target
        self._last_reason = reason
        observed_ns = self._clock.now_ns()
        recorded_ns = observed_ns
        transition = {
            "transition_index": len(self._transitions),
            "from": source.value,
            "to": target.value,
            "reason": reason,
            "monotonic_ns": recorded_ns,
            "clock_domain_id": self._clock.domain_id,
        }
        if (
            self._last_transition_ns is not None
            and observed_ns < self._last_transition_ns
        ):
            recorded_ns = self._last_transition_ns
            transition["monotonic_ns"] = recorded_ns
            transition["observed_regressed_monotonic_ns"] = observed_ns
        self._last_transition_ns = recorded_ns
        self._transitions.append(transition)

    def connected(self) -> None:
        self.transition(SafetyState.DISABLED, "connected")

    def ready(self) -> None:
        self.transition(SafetyState.READY, "explicit_enable")

    def active(self) -> None:
        self.transition(SafetyState.ACTIVE, "first_valid_action")

    def safe_stop(self, reason: StopReason) -> None:
        if self._state == SafetyState.SAFE_STOP:
            return
        if self._state in {
            SafetyState.FAULT,
            SafetyState.E_STOP,
            SafetyState.DISCONNECTED,
        }:
            raise StateConflict(f"cannot enter SAFE_STOP from {self._state.value}")
        self.transition(SafetyState.SAFE_STOP, reason.value)

    def fault(self, reason: StopReason = StopReason.DEVICE_FAULT) -> None:
        if self._state == SafetyState.FAULT:
            return
        if self._state == SafetyState.DISCONNECTED:
            raise StateConflict("cannot latch FAULT while disconnected")
        self.transition(SafetyState.FAULT, reason.value)

    def emergency_stop(self) -> None:
        if self._state == SafetyState.E_STOP:
            return
        if self._state == SafetyState.DISCONNECTED:
            raise StateConflict("cannot latch E_STOP while disconnected")
        self.transition(SafetyState.E_STOP, StopReason.E_STOP.value)

    def reset_to_disabled(self, *, healthy: bool, operator_acknowledged: bool) -> None:
        if self._state not in {
            SafetyState.SAFE_STOP,
            SafetyState.FAULT,
            SafetyState.E_STOP,
        }:
            raise StateConflict("reset is allowed only from a latched stop state")
        if not healthy or not operator_acknowledged:
            raise StateConflict(
                "reset requires healthy backend and explicit operator acknowledgement"
            )
        self.transition(SafetyState.DISABLED, "explicit_operator_reset")

    def disconnected(self) -> None:
        if self._state == SafetyState.DISCONNECTED:
            return
        self.transition(SafetyState.DISCONNECTED, "disconnected")


@dataclass
class Watchdog:
    timeout_ns: int
    last_feed_ns: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.timeout_ns, int) or self.timeout_ns <= 0:
            raise ValidationError("watchdog timeout must be a positive integer")

    def feed(self, now_ns: int) -> None:
        if self.last_feed_ns is not None and now_ns < self.last_feed_ns:
            raise ValidationError("watchdog clock regression")
        self.last_feed_ns = now_ns

    def expired(self, now_ns: int) -> bool:
        if self.last_feed_ns is None:
            return False
        if now_ns < self.last_feed_ns:
            raise ValidationError("watchdog clock regression")
        return now_ns - self.last_feed_ns > self.timeout_ns
