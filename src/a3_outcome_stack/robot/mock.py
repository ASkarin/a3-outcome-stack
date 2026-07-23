"""Deterministic, explicitly non-physical robot backend for tests and replay."""

from __future__ import annotations

from typing import Any, Mapping

from a3_outcome_stack.ops.errors import StateConflict, ValidationError

from .clock import Clock
from .safety import StopReason
from .types import (
    ActionEnvelope,
    ActionReceipt,
    Observation,
    RobotStatus,
    SensorTimestamps,
    action_features,
    observation_features,
)


class MockBackend:
    def __init__(self, clock: Clock, config: Mapping[str, Any]):
        if config.get("backend") != "mock" or config.get("simulation_only") is not True:
            raise ValidationError(
                "MockBackend requires an explicitly simulation-only mock config"
            )
        self.clock = clock
        self.config = dict(config)
        self._observation_features = observation_features(
            config.get("camera_features", {})
        )
        self._action_features = action_features()
        self._connected = False
        self._enabled = False
        self._healthy = True
        self._positions = (0.0,) * 7
        self._velocities = (0.0,) * 7
        self._efforts = (0.0,) * 7
        self._joint_faults = [0] * 7
        self._fail_next_feedback = False
        self._last_action: ActionEnvelope | None = None
        self._stop_reasons: list[str] = []

    @property
    def observation_features(self) -> Mapping[str, Any]:
        return self._observation_features

    @property
    def action_features(self) -> Mapping[str, Any]:
        return self._action_features

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def is_healthy(self) -> bool:
        return self._healthy and not any(self._joint_faults)

    @property
    def hardware_verified(self) -> bool:
        return False

    @property
    def last_action(self) -> ActionEnvelope | None:
        return self._last_action

    @property
    def stop_reasons(self) -> tuple[str, ...]:
        return tuple(self._stop_reasons)

    def connect(self) -> None:
        if self._connected:
            raise StateConflict("mock backend is already connected")
        self._connected = True

    def disconnect(self) -> None:
        self._enabled = False
        self._connected = False

    def enable(self) -> None:
        if not self._connected:
            raise StateConflict("mock backend is not connected")
        if not self.is_healthy:
            raise StateConflict("mock backend is unhealthy")
        self._enabled = True

    def disable(self) -> None:
        self._enabled = False

    def get_observation(self) -> Observation:
        if not self._connected:
            raise StateConflict("mock backend is not connected")
        if self._fail_next_feedback:
            self._fail_next_feedback = False
            raise RuntimeError("mock feedback timeout")
        capture = self.clock.now_ns()
        receive = self.clock.now_ns()
        return Observation(
            joint_position=self._positions,
            joint_velocity=self._velocities,
            joint_effort=self._efforts,
            tcp_position=(0.0, 0.0, 0.0),
            tcp_orientation_xyzw=(0.0, 0.0, 0.0, 1.0),
            status=RobotStatus(
                vendor_state="MOCK_ENABLED" if self._enabled else "MOCK_DISABLED",
                enabled=self._enabled,
                joint_faults=tuple(self._joint_faults),
                safety_state="UNBOUND",
                tcp_pose_valid=False,
            ),
            timestamps=SensorTimestamps(capture, receive, self.clock.domain_id),
            camera_refs={},
        )

    def send_action(self, action: ActionEnvelope) -> ActionReceipt:
        if not self._connected or not self._enabled:
            raise StateConflict("mock backend must be connected and enabled")
        action.validate()
        sent = self.clock.now_ns()
        previous = self._positions
        self._positions = tuple(float(value) for value in action.values)
        self._velocities = tuple(
            new - old for old, new in zip(previous, self._positions)
        )
        self._efforts = (0.0,) * 7
        self._last_action = action
        ack = self.clock.now_ns()
        return ActionReceipt(action.sequence_id, sent, ack, True)

    def request_safe_stop(self, reason: StopReason) -> None:
        self._enabled = False
        self._velocities = (0.0,) * 7
        self._stop_reasons.append(reason.value)

    def emergency_stop(self) -> None:
        self._enabled = False
        self._velocities = (0.0,) * 7
        self._stop_reasons.append(StopReason.E_STOP.value)

    def fail_next_feedback(self) -> None:
        self._fail_next_feedback = True

    def set_joint_fault(self, joint_index: int, fault_code: int) -> None:
        if joint_index < 0 or joint_index >= 7 or fault_code < 0:
            raise ValidationError("invalid mock joint fault")
        self._joint_faults[joint_index] = fault_code
        self._healthy = fault_code == 0 and not any(self._joint_faults)

    def clear_faults(self) -> None:
        self._joint_faults = [0] * 7
        self._healthy = True
