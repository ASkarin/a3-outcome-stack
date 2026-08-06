"""Guarded adapter for the pinned official ELA3Interface API."""

from __future__ import annotations

import importlib
import math
from typing import Any, Callable, Mapping

from a3_outcome_stack.ops.errors import StateConflict, ValidationError

from .clock import Clock
from .safety import StopReason
from .types import (
    JOINT_NAMES,
    ActionEnvelope,
    ActionReceipt,
    Observation,
    RobotStatus,
    SensorTimestamps,
    rpy_to_quaternion_xyzw,
)


def _finite_list(value: Any, field: str, length: int = 7) -> list[float]:
    if not isinstance(value, list) or len(value) != length:
        raise ValidationError(f"{field} must contain {length} values")
    result = []
    for item in value:
        if item is None:
            raise ValidationError(f"{field} remains unfrozen")
        converted = float(item)
        if not math.isfinite(converted):
            raise ValidationError(f"{field} contains a non-finite value")
        result.append(converted)
    return result


def validate_calibration(calibration: Mapping[str, Any]) -> dict[str, Any]:
    if calibration.get("schema_version") != "a3-calibration-v1":
        raise ValidationError("unsupported calibration schema")
    if calibration.get("status") != "frozen":
        raise StateConflict("A3 calibration is not frozen")
    if calibration.get("joint_order") != list(JOINT_NAMES):
        raise ValidationError("calibration joint order mismatch")
    if not isinstance(calibration.get("robot_serial"), str) or not calibration["robot_serial"]:
        raise ValidationError("frozen calibration requires robot_serial")
    for field in ("hardware_revision", "calibrated_at_utc", "operator", "method"):
        if not isinstance(calibration.get(field), str) or not calibration[field]:
            raise ValidationError(f"frozen calibration requires {field}")
    directions = _finite_list(calibration.get("direction_sign"), "direction_sign")
    if any(value not in {-1.0, 1.0} for value in directions):
        raise ValidationError("direction_sign values must be -1 or 1")
    offsets = _finite_list(calibration.get("zero_offset_rad"), "zero_offset_rad")
    home = _finite_list(calibration.get("home_position_rad"), "home_position_rad")
    return {"directions": directions, "offsets": offsets, "home": home}


def validate_safety(safety: Mapping[str, Any]) -> dict[str, Any]:
    if safety.get("schema_version") != "a3-safety-v1":
        raise ValidationError("unsupported safety schema")
    if safety.get("approval_status") != "frozen" or safety.get("simulation_only") is not False:
        raise StateConflict("A3 hardware safety configuration is not frozen")
    if safety.get("physical_estop_verified") is not True:
        raise StateConflict("physical emergency stop is not verified")
    if not isinstance(safety.get("workspace_limits"), Mapping) or not safety["workspace_limits"]:
        raise ValidationError("frozen safety configuration requires workspace_limits")
    lower = _finite_list(safety.get("joint_position_lower_rad"), "joint_position_lower_rad")
    upper = _finite_list(safety.get("joint_position_upper_rad"), "joint_position_upper_rad")
    if not all(lo < hi for lo, hi in zip(lower, upper)):
        raise ValidationError("hardware joint lower limits must be below upper limits")
    scalar_fields = (
        "watchdog_timeout_ns",
        "max_velocity_rad_s",
        "max_acceleration_rad_s2",
        "velocity_limit_rad_s",
        "limit_margin_rad",
        "limit_stop_margin_rad",
        "limit_decel_factor",
    )
    scalars: dict[str, float] = {}
    for field in scalar_fields:
        raw = safety.get(field)
        if (
            raw is None
            or not isinstance(raw, (int, float))
            or not math.isfinite(float(raw))
            or float(raw) <= 0
        ):
            raise ValidationError(f"{field} must be a frozen positive finite value")
        scalars[field] = float(raw)
    if scalars["limit_decel_factor"] > 1.0:
        raise ValidationError("limit_decel_factor must not exceed 1")
    if scalars["limit_stop_margin_rad"] >= scalars["limit_margin_rad"]:
        raise ValidationError("limit_stop_margin_rad must be below limit_margin_rad")
    return {
        "lower": lower,
        "upper": upper,
        **scalars,
    }


def validate_hardware_ready(
    calibration: Mapping[str, Any], safety: Mapping[str, Any]
) -> dict[str, Any]:
    return {**validate_calibration(calibration), **validate_safety(safety)}


def validate_hardware_gate(
    gate: Mapping[str, Any], *, calibration_sha256: str, safety_sha256: str
) -> None:
    if gate.get("schema_version") != "a3-hardware-gate-v1":
        raise ValidationError("unsupported A3 hardware gate schema")
    required = (
        "hardware_available",
        "hardware_tests_executed",
        "motor_enable_executed",
        "real_can_traffic_executed",
        "hardware_verified",
    )
    closed = [field for field in required if gate.get(field) is not True]
    if closed:
        raise StateConflict(f"A3 hardware gate remains closed: {closed}")
    if gate.get("calibration_sha256") != calibration_sha256:
        raise StateConflict("A3 hardware gate is not bound to the active calibration")
    if gate.get("safety_sha256") != safety_sha256:
        raise StateConflict("A3 hardware gate is not bound to the active safety configuration")


class A3SdkBackend:
    """Direct mapping to the pinned official ELA3Interface API."""

    def __init__(
        self,
        clock: Clock,
        config: Mapping[str, Any],
        calibration: Mapping[str, Any],
        safety: Mapping[str, Any],
        *,
        vendor_factory: Callable[..., Any] | None = None,
        apply_safety_limits: bool = False,
    ):
        self.clock = clock
        self.config = dict(config)
        self.calibration = dict(calibration)
        self.safety = dict(safety)
        self._vendor_factory = vendor_factory
        self._apply_safety_limits = apply_safety_limits
        self._vendor: Any = None
        self._connected = False
        self._enabled = False
        self._healthy = True
        self._last_faults = (0,) * 7

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def is_healthy(self) -> bool:
        return self._healthy and not any(self._last_faults)

    def _factory(self) -> Callable[..., Any]:
        if self._vendor_factory is not None:
            return self._vendor_factory
        module = importlib.import_module("el_a3_sdk")
        return module.ELA3Interface

    def _vendor_kwargs(self) -> dict[str, Any]:
        calibration = validate_calibration(self.calibration)
        kwargs: dict[str, Any] = {
            "can_name": self.config.get("can_interface", "can0"),
            "start_sdk_joint_limit": False,
        }
        kwargs.update(
            {
                "joint_directions": {
                    index + 1: value for index, value in enumerate(calibration["directions"])
                },
                "joint_offsets": {
                    index + 1: value for index, value in enumerate(calibration["offsets"])
                },
            }
        )
        if not self._apply_safety_limits:
            return kwargs
        frozen = validate_safety(self.safety)
        kwargs.update(
            {
                "joint_limits": {
                    index + 1: (frozen["lower"][index], frozen["upper"][index])
                    for index in range(7)
                },
                "start_sdk_joint_limit": True,
                "max_velocity": frozen["max_velocity_rad_s"],
                "max_acceleration": frozen["max_acceleration_rad_s2"],
                "velocity_limit": frozen["velocity_limit_rad_s"],
                "limit_margin": frozen["limit_margin_rad"],
                "limit_stop_margin": frozen["limit_stop_margin_rad"],
                "limit_decel_factor": frozen["limit_decel_factor"],
            }
        )
        return kwargs

    def connect(self) -> None:
        if self._connected:
            raise StateConflict("A3 SDK backend is already connected")
        vendor_kwargs = self._vendor_kwargs()
        self._vendor = self._factory()(**vendor_kwargs)
        if not self._vendor.ConnectPort():
            raise StateConflict("ELA3Interface.ConnectPort returned false")
        self._connected = True

    def disconnect(self) -> None:
        if self._vendor is not None:
            self._vendor.DisconnectPort()
        self._enabled = False
        self._connected = False

    def enable(self) -> None:
        if not self._apply_safety_limits:
            raise StateConflict("A3 SDK enable requires explicit safety-limit mode")
        validate_hardware_ready(self.calibration, self.safety)
        if not self._connected:
            raise StateConflict("A3 SDK backend is not connected")
        if not self._vendor.EnableArm():
            raise StateConflict("ELA3Interface.EnableArm returned false")
        self._enabled = True

    def disable(self) -> None:
        if self._connected and self._vendor is not None and not self._vendor.DisableArm():
            raise StateConflict("ELA3Interface.DisableArm returned false")
        self._enabled = False

    def get_observation(self) -> Observation:
        if not self._connected:
            raise StateConflict("A3 SDK backend is not connected")
        capture = self.clock.now_ns()
        positions = self._vendor.GetArmJointMsgs().to_list(include_gripper=True)
        velocities = self._vendor.GetArmJointVelocities().to_list(include_gripper=True)
        efforts = self._vendor.GetArmJointEfforts().to_list(include_gripper=True)
        end_pose = self._vendor.GetArmEndPoseMsgs()
        status = self._vendor.GetArmStatus()
        receive = self.clock.now_ns()
        faults = tuple(int(value) for value in status.joint_faults)
        self._last_faults = faults
        self._healthy = not any(faults)
        vendor_state = getattr(getattr(self._vendor, "arm_state", None), "name", "UNKNOWN")
        tcp_valid = self.config.get("tcp_pose_source") == "pinocchio_validated"
        return Observation(
            joint_position=tuple(positions),
            joint_velocity=tuple(velocities),
            joint_effort=tuple(efforts),
            tcp_position=(float(end_pose.x), float(end_pose.y), float(end_pose.z)),
            tcp_orientation_xyzw=rpy_to_quaternion_xyzw(end_pose.rx, end_pose.ry, end_pose.rz),
            status=RobotStatus(
                vendor_state=vendor_state,
                enabled=bool(status.all_enabled),
                joint_faults=faults,
                safety_state="UNBOUND",
                tcp_pose_valid=tcp_valid,
            ),
            timestamps=SensorTimestamps(capture, receive, self.clock.domain_id),
            camera_refs={},
        )

    def send_action(self, action: ActionEnvelope) -> ActionReceipt:
        if not self._connected or not self._enabled:
            raise StateConflict("A3 SDK backend must be connected and enabled")
        action.validate()
        sent = self.clock.now_ns()
        if not self._vendor.JointCtrlList(list(action.values)):
            raise StateConflict("ELA3Interface.JointCtrlList returned false")
        ack = self.clock.now_ns()
        return ActionReceipt(action.sequence_id, sent, ack, True)

    def request_safe_stop(self, reason: StopReason) -> None:
        if self._connected and self._vendor is not None:
            self._vendor.DisableArm()
        self._enabled = False

    def emergency_stop(self) -> None:
        if not self._connected or self._vendor is None or not self._vendor.EmergencyStop():
            raise StateConflict("ELA3Interface.EmergencyStop failed")
        self._enabled = False
