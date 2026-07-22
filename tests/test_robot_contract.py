from __future__ import annotations

import math
from pathlib import Path

import pytest

from embodied_ai.ops.canonical import load_json
from embodied_ai.ops.errors import StateConflict
from embodied_ai.robot.a3_sdk import A3SdkBackend, validate_hardware_ready
from embodied_ai.robot.clock import ManualClock
from embodied_ai.robot.mock import MockBackend
from embodied_ai.robot.types import (
    ActionEnvelope,
    action_features,
    rpy_to_quaternion_xyzw,
    verify_contract_file,
)

ROOT = Path(__file__).parents[1]


class _Joints:
    def __init__(self, values):
        self.values = values

    def to_list(self, include_gripper=True):
        return list(self.values if include_gripper else self.values[:6])


class _Pose:
    x, y, z = 0.1, 0.2, 0.3
    rx, ry, rz = 0.0, 0.0, math.pi / 2


class _Status:
    all_enabled = True
    joint_faults = [0] * 7


class _ArmState:
    name = "ENABLED"


class FakeVendor:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.arm_state = _ArmState()
        self.calls = []

    def ConnectPort(self):
        self.calls.append("ConnectPort")
        return True

    def DisconnectPort(self):
        self.calls.append("DisconnectPort")

    def EnableArm(self):
        self.calls.append("EnableArm")
        return True

    def DisableArm(self):
        self.calls.append("DisableArm")
        return True

    def EmergencyStop(self):
        self.calls.append("EmergencyStop")
        return True

    def JointCtrlList(self, values):
        self.calls.append(("JointCtrlList", list(values)))
        return True

    def GetArmJointMsgs(self):
        return _Joints([0.1] * 7)

    def GetArmJointVelocities(self):
        return _Joints([0.2] * 7)

    def GetArmJointEfforts(self):
        return _Joints([0.3] * 7)

    def GetArmEndPoseMsgs(self):
        return _Pose()

    def GetArmStatus(self):
        return _Status()


def _frozen_configs():
    calibration = load_json(ROOT / "configs/robot/a3_calibration.template.json")
    calibration.update(
        {
            "status": "frozen",
            "robot_serial": "TEST-ONLY",
            "hardware_revision": "TEST-ONLY",
            "direction_sign": [1.0] * 7,
            "zero_offset_rad": [0.0] * 7,
            "home_position_rad": [0.0] * 7,
            "calibrated_at_utc": "2026-07-22T00:00:00Z",
            "operator": "test-fixture",
            "method": "synthetic-test-only",
        }
    )
    safety = load_json(ROOT / "configs/robot/a3_safety.template.json")
    safety.update(
        {
            "approval_status": "frozen",
            "joint_position_lower_rad": [-1.0] * 7,
            "joint_position_upper_rad": [1.0] * 7,
            "watchdog_timeout_ns": 100,
            "max_velocity_rad_s": 1.0,
            "max_acceleration_rad_s2": 1.0,
            "velocity_limit_rad_s": 1.0,
            "limit_margin_rad": 0.1,
            "limit_stop_margin_rad": 0.01,
            "limit_decel_factor": 0.5,
            "workspace_limits": {"source": "synthetic-test-only"},
            "physical_estop_verified": True,
        }
    )
    return calibration, safety


def test_contract_and_features_are_available_before_connect():
    contract = verify_contract_file(ROOT / "configs/robot/a3_contract_v1.json")
    config = load_json(ROOT / "configs/robot/a3_mock_test.json")
    backend = MockBackend(ManualClock(domain_id=config["clock_domain_id"]), config)
    assert contract["joint_order"] == ["L1", "L2", "L3", "L4", "L5", "L6", "L7"]
    assert backend.observation_features["joint_position"]["shape"] == [7]
    assert backend.observation_features["camera.mock_rgb"]["shape"] == [8, 8, 3]
    assert backend.action_features == action_features()
    assert not backend.is_connected


def test_rpy_is_converted_to_normalized_xyzw_quaternion():
    quaternion = rpy_to_quaternion_xyzw(0.0, 0.0, math.pi / 2)
    assert math.sqrt(sum(value * value for value in quaternion)) == pytest.approx(1.0)
    assert quaternion == pytest.approx((0.0, 0.0, math.sqrt(0.5), math.sqrt(0.5)))


def test_sdk_backend_refuses_unfrozen_enable_but_allows_fake_readonly_connect():
    calibration = load_json(ROOT / "configs/robot/a3_calibration.template.json")
    safety = load_json(ROOT / "configs/robot/a3_safety.template.json")
    clock = ManualClock(current_ns=10, domain_id="sdk-test")
    created = []

    def factory(**kwargs):
        vendor = FakeVendor(**kwargs)
        created.append(vendor)
        return vendor

    backend = A3SdkBackend(
        clock,
        {"can_interface": "can-test"},
        calibration,
        safety,
        vendor_factory=factory,
    )
    backend.connect()
    assert created[0].kwargs["start_sdk_joint_limit"] is False
    with pytest.raises(StateConflict, match="calibration is not frozen"):
        backend.enable()
    assert "EnableArm" not in created[0].calls
    assert backend.hardware_verified is False
    backend.disconnect()


def test_sdk_backend_maps_pinned_vendor_contract_with_test_only_frozen_values():
    calibration, safety = _frozen_configs()
    frozen = validate_hardware_ready(calibration, safety)
    assert frozen["directions"] == [1.0] * 7
    clock = ManualClock(current_ns=1000, domain_id="sdk-test")
    created = []

    def factory(**kwargs):
        vendor = FakeVendor(**kwargs)
        created.append(vendor)
        return vendor

    backend = A3SdkBackend(
        clock,
        {"can_interface": "can-test", "tcp_pose_source": "pinocchio_validated"},
        calibration,
        safety,
        vendor_factory=factory,
    )
    backend.connect()
    assert created[0].kwargs["start_sdk_joint_limit"] is True
    backend.enable()
    observation = backend.get_observation()
    assert observation.joint_position == pytest.approx((0.1,) * 7)
    assert observation.joint_velocity == pytest.approx((0.2,) * 7)
    assert observation.joint_effort == pytest.approx((0.3,) * 7)
    assert observation.status.tcp_pose_valid is True
    action = ActionEnvelope((0.25,) * 7, 0, 1000, 1100, "sdk-test")
    receipt = backend.send_action(action)
    assert receipt.accepted
    assert ("JointCtrlList", [0.25] * 7) in created[0].calls
    backend.emergency_stop()
    assert "EmergencyStop" in created[0].calls
    backend.disconnect()
