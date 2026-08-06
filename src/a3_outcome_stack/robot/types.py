"""Internal observation, action, status, and timing types."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any, Mapping, Sequence

from a3_outcome_stack.ops.errors import ValidationError

JOINT_NAMES = ("L1", "L2", "L3", "L4", "L5", "L6", "L7")


def _finite_tuple(values: Sequence[Any], length: int, field: str) -> tuple[float, ...]:
    if isinstance(values, (str, bytes)) or len(values) != length:
        raise ValidationError(f"{field} must contain exactly {length} values")
    converted = tuple(float(value) for value in values)
    if not all(math.isfinite(value) for value in converted):
        raise ValidationError(f"{field} contains a non-finite value")
    return converted


@dataclass(frozen=True)
class SensorTimestamps:
    capture_monotonic_ns: int
    receive_monotonic_ns: int
    clock_domain_id: str

    def validate(self) -> None:
        if not self.clock_domain_id:
            raise ValidationError("sensor timestamps require clock_domain_id")
        if not isinstance(self.capture_monotonic_ns, int) or not isinstance(
            self.receive_monotonic_ns, int
        ):
            raise ValidationError("sensor timestamps must be integer nanoseconds")
        if self.capture_monotonic_ns < 0 or self.receive_monotonic_ns < self.capture_monotonic_ns:
            raise ValidationError("sensor timestamps must satisfy 0 <= capture <= receive")


@dataclass(frozen=True)
class RobotStatus:
    vendor_state: str
    enabled: bool
    joint_faults: tuple[int, ...]
    safety_state: str
    tcp_pose_valid: bool

    def validate(self) -> None:
        if not self.vendor_state or not self.safety_state:
            raise ValidationError("robot status requires vendor_state and safety_state")
        if len(self.joint_faults) != 7 or not all(
            isinstance(value, int) and value >= 0 for value in self.joint_faults
        ):
            raise ValidationError("joint_faults must contain seven non-negative integers")

    def with_safety_state(self, state: str) -> "RobotStatus":
        return replace(self, safety_state=state)


@dataclass(frozen=True)
class Observation:
    joint_position: tuple[float, ...]
    joint_velocity: tuple[float, ...]
    joint_effort: tuple[float, ...]
    tcp_position: tuple[float, ...]
    tcp_orientation_xyzw: tuple[float, ...]
    status: RobotStatus
    timestamps: SensorTimestamps
    camera_refs: Mapping[str, Mapping[str, Any]]

    def validate(self) -> None:
        _finite_tuple(self.joint_position, 7, "joint_position")
        _finite_tuple(self.joint_velocity, 7, "joint_velocity")
        _finite_tuple(self.joint_effort, 7, "joint_effort")
        _finite_tuple(self.tcp_position, 3, "tcp_position")
        quaternion = _finite_tuple(self.tcp_orientation_xyzw, 4, "tcp_orientation_xyzw")
        norm = math.sqrt(sum(value * value for value in quaternion))
        if not math.isclose(norm, 1.0, rel_tol=1e-5, abs_tol=1e-5):
            raise ValidationError("tcp_orientation_xyzw must be a unit quaternion")
        self.status.validate()
        self.timestamps.validate()
        for name, reference in self.camera_refs.items():
            if not isinstance(name, str) or not isinstance(reference, Mapping):
                raise ValidationError("camera_refs must map names to objects")


@dataclass(frozen=True)
class ActionEnvelope:
    values: tuple[float, ...]
    sequence_id: int
    created_monotonic_ns: int
    deadline_monotonic_ns: int
    clock_domain_id: str

    def validate(self) -> None:
        _finite_tuple(self.values, 7, "action.values")
        if not isinstance(self.sequence_id, int) or self.sequence_id < 0:
            raise ValidationError("sequence_id must be a non-negative integer")
        if not self.clock_domain_id:
            raise ValidationError("action requires clock_domain_id")
        if not isinstance(self.created_monotonic_ns, int) or not isinstance(
            self.deadline_monotonic_ns, int
        ):
            raise ValidationError("action timestamps must be integer nanoseconds")
        if self.created_monotonic_ns < 0 or self.deadline_monotonic_ns < self.created_monotonic_ns:
            raise ValidationError("action timestamps must satisfy 0 <= created <= deadline")


@dataclass(frozen=True)
class ActionReceipt:
    sequence_id: int
    sent_monotonic_ns: int
    ack_monotonic_ns: int
    accepted: bool

    def validate(self, action: ActionEnvelope | None = None) -> None:
        if not isinstance(self.sequence_id, int) or self.sequence_id < 0:
            raise ValidationError("receipt sequence_id must be non-negative")
        if self.sent_monotonic_ns < 0 or self.ack_monotonic_ns < self.sent_monotonic_ns:
            raise ValidationError("receipt timestamps must satisfy 0 <= sent <= ack")
        if action is not None:
            if self.sequence_id != action.sequence_id:
                raise ValidationError("receipt sequence does not match action")
            if self.sent_monotonic_ns < action.created_monotonic_ns:
                raise ValidationError("receipt was sent before action creation")


def rpy_to_quaternion_xyzw(
    roll: float, pitch: float, yaw: float
) -> tuple[float, float, float, float]:
    values = _finite_tuple((roll, pitch, yaw), 3, "rpy")
    cr, sr = math.cos(values[0] / 2), math.sin(values[0] / 2)
    cp, sp = math.cos(values[1] / 2), math.sin(values[1] / 2)
    cy, sy = math.cos(values[2] / 2), math.sin(values[2] / 2)
    quaternion = (
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
        cr * cp * cy + sr * sp * sy,
    )
    norm = math.sqrt(sum(value * value for value in quaternion))
    return tuple(value / norm for value in quaternion)  # type: ignore[return-value]
