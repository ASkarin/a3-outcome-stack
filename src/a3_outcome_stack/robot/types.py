"""Versioned observation, action, status, and feature contracts."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

from a3_outcome_stack.ops.canonical import load_json, validate_sha256
from a3_outcome_stack.ops.errors import ValidationError

JOINT_NAMES = ("L1", "L2", "L3", "L4", "L5", "L6", "L7")
ACTION_MODE = "joint_position_abs"


def _finite_tuple(values: Sequence[Any], length: int, field: str) -> tuple[float, ...]:
    if isinstance(values, (str, bytes)) or len(values) != length:
        raise ValidationError(f"{field} must contain exactly {length} values")
    converted = tuple(float(value) for value in values)
    if not all(math.isfinite(value) for value in converted):
        raise ValidationError(f"{field} contains a non-finite value")
    return converted


def observation_features(
    camera_features: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    features: dict[str, Any] = {
        "joint_position": {
            "dtype": "float32",
            "shape": [7],
            "unit": "rad",
            "policy_input": True,
        },
        "joint_velocity": {
            "dtype": "float32",
            "shape": [7],
            "unit": "rad/s",
            "policy_input": True,
        },
        "joint_effort": {
            "dtype": "float32",
            "shape": [7],
            "unit": "N*m",
            "policy_input": False,
        },
        "tcp_position": {
            "dtype": "float32",
            "shape": [3],
            "unit": "m",
            "frame": "base",
            "policy_input": False,
        },
        "tcp_orientation_xyzw": {
            "dtype": "float32",
            "shape": [4],
            "unit": "unit_quaternion",
            "frame": "base",
            "policy_input": False,
        },
    }
    for name, feature in sorted((camera_features or {}).items()):
        if not isinstance(name, str) or not name:
            raise ValidationError("camera feature names must be non-empty strings")
        features[f"camera.{name}"] = dict(feature)
    validate_feature_map(features)
    return features


def action_features() -> dict[str, Any]:
    return {
        "joint_position_abs": {
            "dtype": "float32",
            "shape": [7],
            "unit": "rad",
            "joint_order": list(JOINT_NAMES),
        }
    }


def validate_feature_map(features: Mapping[str, Any]) -> None:
    for name, feature in features.items():
        if not isinstance(name, str) or not isinstance(feature, Mapping):
            raise ValidationError(
                "feature maps require string names and object definitions"
            )
        dtype = feature.get("dtype")
        shape = feature.get("shape")
        unit = feature.get("unit")
        if dtype not in {"float32", "uint8", "bool", "int32"}:
            raise ValidationError(f"unsupported dtype for {name}: {dtype}")
        if (
            not isinstance(shape, list)
            or not shape
            or not all(isinstance(v, int) and v > 0 for v in shape)
        ):
            raise ValidationError(f"invalid shape for {name}")
        if not isinstance(unit, str) or not unit:
            raise ValidationError(f"missing unit for {name}")


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
        if (
            self.capture_monotonic_ns < 0
            or self.receive_monotonic_ns < self.capture_monotonic_ns
        ):
            raise ValidationError(
                "sensor timestamps must satisfy 0 <= capture <= receive"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "capture_monotonic_ns": self.capture_monotonic_ns,
            "receive_monotonic_ns": self.receive_monotonic_ns,
            "clock_domain_id": self.clock_domain_id,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "SensorTimestamps":
        result = cls(
            capture_monotonic_ns=value.get("capture_monotonic_ns"),
            receive_monotonic_ns=value.get("receive_monotonic_ns"),
            clock_domain_id=value.get("clock_domain_id"),
        )
        result.validate()
        return result


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
            raise ValidationError(
                "joint_faults must contain seven non-negative integers"
            )

    def with_safety_state(self, state: str) -> "RobotStatus":
        return replace(self, safety_state=state)

    def to_dict(self) -> dict[str, Any]:
        return {
            "vendor_state": self.vendor_state,
            "enabled": self.enabled,
            "joint_faults": list(self.joint_faults),
            "safety_state": self.safety_state,
            "tcp_pose_valid": self.tcp_pose_valid,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "RobotStatus":
        result = cls(
            vendor_state=value.get("vendor_state"),
            enabled=value.get("enabled"),
            joint_faults=tuple(value.get("joint_faults", [])),
            safety_state=value.get("safety_state"),
            tcp_pose_valid=value.get("tcp_pose_valid"),
        )
        result.validate()
        return result


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
            if "sha256" in reference:
                validate_sha256(reference["sha256"], f"camera_refs.{name}.sha256")

    def to_dict(self) -> dict[str, Any]:
        return {
            "joint_position": list(self.joint_position),
            "joint_velocity": list(self.joint_velocity),
            "joint_effort": list(self.joint_effort),
            "tcp_position": list(self.tcp_position),
            "tcp_orientation_xyzw": list(self.tcp_orientation_xyzw),
            "status": self.status.to_dict(),
            "timestamps": self.timestamps.to_dict(),
            "camera_refs": {
                name: dict(value) for name, value in sorted(self.camera_refs.items())
            },
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "Observation":
        result = cls(
            joint_position=_finite_tuple(
                value.get("joint_position", []), 7, "joint_position"
            ),
            joint_velocity=_finite_tuple(
                value.get("joint_velocity", []), 7, "joint_velocity"
            ),
            joint_effort=_finite_tuple(
                value.get("joint_effort", []), 7, "joint_effort"
            ),
            tcp_position=_finite_tuple(
                value.get("tcp_position", []), 3, "tcp_position"
            ),
            tcp_orientation_xyzw=_finite_tuple(
                value.get("tcp_orientation_xyzw", []), 4, "tcp_orientation_xyzw"
            ),
            status=RobotStatus.from_dict(value.get("status", {})),
            timestamps=SensorTimestamps.from_dict(value.get("timestamps", {})),
            camera_refs=value.get("camera_refs", {}),
        )
        result.validate()
        return result


@dataclass(frozen=True)
class ActionEnvelope:
    values: tuple[float, ...]
    sequence_id: int
    created_monotonic_ns: int
    deadline_monotonic_ns: int
    clock_domain_id: str
    mode: str = ACTION_MODE

    def validate(self) -> None:
        _finite_tuple(self.values, 7, "action.values")
        if self.mode != ACTION_MODE:
            raise ValidationError(f"unsupported action mode: {self.mode}")
        if not isinstance(self.sequence_id, int) or self.sequence_id < 0:
            raise ValidationError("sequence_id must be a non-negative integer")
        if not self.clock_domain_id:
            raise ValidationError("action requires clock_domain_id")
        if not isinstance(self.created_monotonic_ns, int) or not isinstance(
            self.deadline_monotonic_ns, int
        ):
            raise ValidationError("action timestamps must be integer nanoseconds")
        if (
            self.created_monotonic_ns < 0
            or self.deadline_monotonic_ns < self.created_monotonic_ns
        ):
            raise ValidationError(
                "action timestamps must satisfy 0 <= created <= deadline"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "values": list(self.values),
            "sequence_id": self.sequence_id,
            "created_monotonic_ns": self.created_monotonic_ns,
            "deadline_monotonic_ns": self.deadline_monotonic_ns,
            "clock_domain_id": self.clock_domain_id,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ActionEnvelope":
        result = cls(
            mode=value.get("mode", ACTION_MODE),
            values=tuple(value.get("values", [])),
            sequence_id=value.get("sequence_id"),
            created_monotonic_ns=value.get("created_monotonic_ns"),
            deadline_monotonic_ns=value.get("deadline_monotonic_ns"),
            clock_domain_id=value.get("clock_domain_id"),
        )
        result.validate()
        return result


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

    def to_dict(self) -> dict[str, Any]:
        return {
            "sequence_id": self.sequence_id,
            "sent_monotonic_ns": self.sent_monotonic_ns,
            "ack_monotonic_ns": self.ack_monotonic_ns,
            "accepted": self.accepted,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ActionReceipt":
        result = cls(
            sequence_id=value.get("sequence_id"),
            sent_monotonic_ns=value.get("sent_monotonic_ns"),
            ack_monotonic_ns=value.get("ack_monotonic_ns"),
            accepted=value.get("accepted"),
        )
        result.validate()
        return result


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


def verify_contract_file(path: str | Path) -> dict[str, Any]:
    contract = load_json(path)
    if contract.get("schema_version") != "a3-contract-v1":
        raise ValidationError("unsupported A3 contract schema")
    if contract.get("joint_order") != list(JOINT_NAMES):
        raise ValidationError("A3 contract joint_order mismatch")
    expected_observation = observation_features(contract.get("camera_features", {}))
    if contract.get("observation_features") != expected_observation:
        raise ValidationError("A3 observation feature contract mismatch")
    if contract.get("action_features") != action_features():
        raise ValidationError("A3 action feature contract mismatch")
    return contract
