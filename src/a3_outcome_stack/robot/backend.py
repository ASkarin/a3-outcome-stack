"""Backend protocol and the safety-enforcing public robot facade."""

from __future__ import annotations

import math
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable

from a3_outcome_stack.ops.errors import IntegrityError, StateConflict, ValidationError

from .clock import Clock
from .safety import SafetyState, SafetySupervisor, StopReason, Watchdog
from .types import ActionEnvelope, ActionReceipt, Observation


@runtime_checkable
class RobotBackend(Protocol):
    @property
    def observation_features(self) -> Mapping[str, Any]: ...

    @property
    def action_features(self) -> Mapping[str, Any]: ...

    @property
    def is_connected(self) -> bool: ...

    @property
    def is_healthy(self) -> bool: ...

    @property
    def hardware_verified(self) -> bool: ...

    def connect(self) -> None: ...

    def disconnect(self) -> None: ...

    def enable(self) -> None: ...

    def disable(self) -> None: ...

    def get_observation(self) -> Observation: ...

    def send_action(self, action: ActionEnvelope) -> ActionReceipt: ...

    def request_safe_stop(self, reason: StopReason) -> None: ...

    def emergency_stop(self) -> None: ...


class SafeRobot:
    """The only actuator-facing entry point exposed to project code."""

    def __init__(
        self,
        backend: RobotBackend,
        clock: Clock,
        *,
        watchdog_timeout_ns: int,
        joint_lower: Sequence[float],
        joint_upper: Sequence[float],
    ):
        if len(joint_lower) != 7 or len(joint_upper) != 7:
            raise ValidationError(
                "SafeRobot requires seven lower and upper joint limits"
            )
        self.backend = backend
        self.clock = clock
        self.supervisor = SafetySupervisor(clock)
        self.watchdog = Watchdog(watchdog_timeout_ns)
        self._joint_lower = tuple(float(value) for value in joint_lower)
        self._joint_upper = tuple(float(value) for value in joint_upper)
        if not all(
            math.isfinite(value) for value in self._joint_lower + self._joint_upper
        ):
            raise ValidationError("joint limits must be finite")
        if not all(
            lower < upper for lower, upper in zip(self._joint_lower, self._joint_upper)
        ):
            raise ValidationError(
                "each lower joint limit must be below its upper limit"
            )
        self._last_sequence_id = -1
        self._last_clock_ns: int | None = None

    @property
    def observation_features(self) -> Mapping[str, Any]:
        return self.backend.observation_features

    @property
    def action_features(self) -> Mapping[str, Any]:
        return self.backend.action_features

    @property
    def state(self) -> SafetyState:
        return self.supervisor.state

    def _now(self) -> int:
        value = self.clock.now_ns()
        if self._last_clock_ns is not None and value < self._last_clock_ns:
            self._safe_stop(StopReason.CLOCK_REGRESSION)
            raise StateConflict("monotonic clock regressed")
        self._last_clock_ns = value
        return value

    def connect(self) -> None:
        if self.state != SafetyState.DISCONNECTED:
            raise StateConflict("connect requires DISCONNECTED state")
        self.backend.connect()
        if not self.backend.is_connected:
            raise StateConflict("backend did not become connected")
        self._now()
        self.supervisor.connected()

    def enable(self) -> None:
        if self.state != SafetyState.DISABLED:
            raise StateConflict("enable requires DISABLED state")
        self.backend.enable()
        self._now()
        self.supervisor.ready()

    def disable(self) -> None:
        if self.state not in {SafetyState.READY, SafetyState.ACTIVE}:
            raise StateConflict("disable requires READY or ACTIVE state")
        self.backend.disable()
        self.supervisor.transition(SafetyState.DISABLED, "explicit_disable")

    def disconnect(self) -> None:
        if self.backend.is_connected:
            self.backend.disconnect()
        self.supervisor.disconnected()

    def _safe_stop(self, reason: StopReason) -> None:
        if self.state in {SafetyState.SAFE_STOP, SafetyState.FAULT, SafetyState.E_STOP}:
            return
        self.backend.request_safe_stop(reason)
        self.supervisor.safe_stop(reason)

    def request_safe_stop(self, reason: StopReason = StopReason.OPERATOR_REQUEST) -> None:
        """Expose the latched safe-stop path without exposing the backend directly."""

        if self.state == SafetyState.DISCONNECTED:
            raise StateConflict("cannot safe-stop a disconnected backend")
        self._safe_stop(reason)

    def _validation_stop_reason(self, exc: ValidationError) -> StopReason:
        message = str(exc).lower()
        if "non-finite" in message:
            return StopReason.NONFINITE
        if "exactly 7" in message or "shape" in message or "seven" in message:
            return StopReason.SHAPE_MISMATCH
        return StopReason.REJECTED_ACTION

    def send_action(self, action: ActionEnvelope) -> ActionReceipt:
        if self.state not in {SafetyState.READY, SafetyState.ACTIVE}:
            raise StateConflict(f"actions are not allowed in {self.state.value}")
        try:
            action.validate()
        except ValidationError as exc:
            self._safe_stop(self._validation_stop_reason(exc))
            raise

        now = self._now()
        if action.clock_domain_id != self.clock.domain_id:
            self._safe_stop(StopReason.CLOCK_DOMAIN_MISMATCH)
            raise ValidationError("action clock domain mismatch")
        if action.created_monotonic_ns > now or now > action.deadline_monotonic_ns:
            self._safe_stop(StopReason.STALE_ACTION)
            raise ValidationError("action is future-dated or stale")
        if action.sequence_id != self._last_sequence_id + 1:
            self._safe_stop(StopReason.OUT_OF_ORDER)
            raise ValidationError("action sequence is not contiguous")
        if any(
            value < lower or value > upper
            for value, lower, upper in zip(
                action.values, self._joint_lower, self._joint_upper
            )
        ):
            self._safe_stop(StopReason.LIMIT_VIOLATION)
            raise ValidationError("action violates configured joint limits")

        try:
            receipt = self.backend.send_action(action)
            receipt.validate(action)
        except (IntegrityError, StateConflict, ValidationError, RuntimeError):
            self._safe_stop(StopReason.BACKEND_EXCEPTION)
            raise
        if not receipt.accepted:
            self._safe_stop(StopReason.REJECTED_ACTION)
            raise StateConflict("backend rejected action")

        self._last_sequence_id = action.sequence_id
        self.watchdog.feed(now)
        if self.state == SafetyState.READY:
            self.supervisor.active()
        return receipt

    def get_observation(self) -> Observation:
        if self.state == SafetyState.DISCONNECTED:
            raise StateConflict("cannot read observation while disconnected")
        try:
            observation = self.backend.get_observation()
            observation.validate()
        except (IntegrityError, StateConflict, ValidationError, RuntimeError) as exc:
            reason = (
                StopReason.FEEDBACK_TIMEOUT
                if "feedback timeout" in str(exc).lower()
                else StopReason.BACKEND_EXCEPTION
            )
            self._safe_stop(reason)
            raise
        now = self._now()
        if observation.timestamps.clock_domain_id != self.clock.domain_id:
            self._safe_stop(StopReason.CLOCK_DOMAIN_MISMATCH)
            raise ValidationError("observation clock domain mismatch")
        if observation.timestamps.receive_monotonic_ns > now:
            self._safe_stop(StopReason.CLOCK_REGRESSION)
            raise ValidationError("observation receive time is in the future")
        if any(observation.status.joint_faults):
            self.backend.request_safe_stop(StopReason.DEVICE_FAULT)
            self.supervisor.fault(StopReason.DEVICE_FAULT)
            raise StateConflict("device feedback contains a joint fault")
        status = observation.status.with_safety_state(self.state.value)
        return Observation(
            joint_position=observation.joint_position,
            joint_velocity=observation.joint_velocity,
            joint_effort=observation.joint_effort,
            tcp_position=observation.tcp_position,
            tcp_orientation_xyzw=observation.tcp_orientation_xyzw,
            status=status,
            timestamps=observation.timestamps,
            camera_refs=observation.camera_refs,
        )

    def check_watchdog(self) -> bool:
        now = self._now()
        if self.state == SafetyState.ACTIVE and self.watchdog.expired(now):
            self._safe_stop(StopReason.WATCHDOG_TIMEOUT)
            return False
        return True

    def emergency_stop(self) -> None:
        if self.state == SafetyState.DISCONNECTED:
            raise StateConflict("cannot emergency-stop a disconnected backend")
        self.backend.emergency_stop()
        self.supervisor.emergency_stop()

    def reset(self, *, operator_acknowledged: bool) -> None:
        if not self.backend.is_healthy:
            raise StateConflict("backend remains unhealthy")
        self.backend.disable()
        self.supervisor.reset_to_disabled(
            healthy=self.backend.is_healthy,
            operator_acknowledged=operator_acknowledged,
        )
        self.watchdog.last_feed_ns = None
        self._last_sequence_id = -1
