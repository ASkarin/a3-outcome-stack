from __future__ import annotations

import json
import math
from importlib import metadata
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("lerobot")

from lerobot.robots.utils import make_robot_from_config  # noqa: E402
from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402
from lerobot.cameras.opencv import OpenCVCameraConfig  # noqa: E402
from lerobot.utils.constants import ACTION, OBS_STR  # noqa: E402
from lerobot.utils.feature_utils import (  # noqa: E402
    build_dataset_frame,
    hw_to_dataset_features,
)
from lerobot.utils.import_utils import register_third_party_plugins  # noqa: E402

from a3_outcome_stack.ops.errors import StateConflict, ValidationError  # noqa: E402
from a3_outcome_stack.ops.canonical import sha256_file  # noqa: E402
from a3_outcome_stack.robot.clock import ManualClock  # noqa: E402
from lerobot_robot_a3 import A3Robot, A3RobotConfig  # noqa: E402


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
        self.feedback_exception = False

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
        if self.feedback_exception:
            raise RuntimeError("fake SDK feedback exception")
        return _Joints([0.1] * 7)

    def GetArmJointVelocities(self):
        return _Joints([0.2] * 7)

    def GetArmJointEfforts(self):
        return _Joints([0.3] * 7)

    def GetArmEndPoseMsgs(self):
        return _Pose()

    def GetArmStatus(self):
        return _Status()


class FakeCamera:
    def __init__(self):
        self.is_connected = False

    def connect(self):
        self.is_connected = True

    def async_read(self):
        return np.zeros((48, 64, 3), dtype=np.uint8)

    def disconnect(self):
        self.is_connected = False


def _write_documents(root: Path, *, frozen: bool = True, gate_open: bool = True):
    root.mkdir(parents=True, exist_ok=True)
    calibration = {
        "schema_version": "a3-calibration-v1",
        "status": "frozen" if frozen else "draft",
        "robot_serial": "TEST-ONLY",
        "hardware_revision": "TEST-ONLY",
        "joint_order": [f"L{index}" for index in range(1, 8)],
        "direction_sign": [1.0] * 7,
        "zero_offset_rad": [0.0] * 7,
        "home_position_rad": [0.0] * 7,
        "calibrated_at_utc": "2026-08-06T00:00:00Z",
        "operator": "test-fixture",
        "method": "synthetic-test-only",
    }
    safety = {
        "schema_version": "a3-safety-v1",
        "approval_status": "frozen",
        "simulation_only": False,
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
    paths = []
    for name, value in (("calibration", calibration), ("safety", safety)):
        path = root / f"{name}.json"
        path.write_text(json.dumps(value), encoding="utf-8")
        paths.append(path)
    gate = {
        "schema_version": "a3-hardware-gate-v1",
        "hardware_available": gate_open,
        "hardware_tests_executed": gate_open,
        "motor_enable_executed": gate_open,
        "real_can_traffic_executed": gate_open,
        "hardware_verified": gate_open,
        "calibration_sha256": sha256_file(paths[0]),
        "safety_sha256": sha256_file(paths[1]),
    }
    path = root / "gate.json"
    path.write_text(json.dumps(gate), encoding="utf-8")
    paths.append(path)
    return tuple(paths)


def _robot(
    tmp_path: Path,
    *,
    mode: str,
    frozen: bool = True,
    gate_open: bool = True,
    cameras=None,
    camera_factory=None,
):
    calibration, safety, gate = _write_documents(tmp_path, frozen=frozen, gate_open=gate_open)
    config = A3RobotConfig(
        id="fixture",
        can_interface="can-test",
        calibration_path=calibration,
        safety_path=safety,
        hardware_gate_path=gate,
        execution_mode=mode,
        cameras=cameras or {},
    )
    created = []

    def factory(**kwargs):
        vendor = FakeVendor(**kwargs)
        created.append(vendor)
        return vendor

    clock = ManualClock(current_ns=1_000, domain_id="plugin-test")
    robot = A3Robot(
        config,
        vendor_factory=factory,
        clock=clock,
        camera_factory=camera_factory,
    )
    return robot, clock, created


def test_plugin_distribution_is_auto_discoverable():
    try:
        distribution = metadata.distribution("lerobot_robot_a3")
    except metadata.PackageNotFoundError:
        pytest.skip("workspace plugin distribution is not installed in this test environment")
    assert distribution.metadata["Name"] == "lerobot_robot_a3"
    register_third_party_plugins()
    config = A3RobotConfig(id="factory-test")
    created = make_robot_from_config(config)
    assert isinstance(created, A3Robot)


def test_read_only_connect_observation_and_action_refusal(tmp_path: Path):
    robot, _, created = _robot(tmp_path, mode="read_only")
    assert set(robot.action_features) == {f"L{index}.pos" for index in range(1, 8)}
    assert set(robot.observation_features) == {
        *{f"L{index}.pos" for index in range(1, 8)},
        *{f"L{index}.vel" for index in range(1, 8)},
    }
    robot.connect()
    assert "EnableArm" not in created[0].calls
    observation = robot.get_observation()
    assert observation["L1.pos"] == pytest.approx(0.1)
    assert observation["L7.vel"] == pytest.approx(0.2)
    with pytest.raises(StateConflict, match="execution_mode=motion"):
        robot.send_action({f"L{index}.pos": 0.0 for index in range(1, 8)})
    robot.disconnect()


def test_connect_refuses_unfrozen_calibration_without_touching_sdk(tmp_path: Path):
    robot, _, created = _robot(tmp_path, mode="read_only", frozen=False)
    with pytest.raises(StateConflict, match="calibration is not frozen"):
        robot.connect()
    assert created == []


def test_configured_camera_is_connected_observed_and_disconnected(tmp_path: Path):
    camera = FakeCamera()
    robot, _, _ = _robot(
        tmp_path,
        mode="read_only",
        cameras={
            "wrist": OpenCVCameraConfig(
                index_or_path=0,
                fps=30,
                width=64,
                height=48,
            )
        },
        camera_factory=lambda _: {"wrist": camera},
    )
    assert robot.observation_features["wrist"] == (48, 64, 3)
    robot.connect()
    assert robot.get_observation()["wrist"].shape == (48, 64, 3)
    robot.disconnect()
    assert camera.is_connected is False


def test_motion_requires_gate_and_converts_actions(tmp_path: Path):
    closed, _, created = _robot(tmp_path / "closed", mode="motion", gate_open=False)
    with pytest.raises(StateConflict, match="hardware gate remains closed"):
        closed.connect()
    assert created == []

    robot, clock, created = _robot(tmp_path / "open", mode="motion")
    robot.connect()
    assert "EnableArm" in created[0].calls
    action = {f"L{index}.pos": 0.25 for index in range(1, 8)}
    assert robot.send_action(action) == action
    assert ("JointCtrlList", [0.25] * 7) in created[0].calls
    clock.advance_ns(101)
    assert robot.check_watchdog() is False
    assert "DisableArm" in created[0].calls
    robot.disconnect()


def test_motion_gate_is_bound_to_active_calibration_and_safety(tmp_path: Path):
    robot, _, created = _robot(tmp_path, mode="motion")
    calibration = json.loads(robot.config.calibration_path.read_text(encoding="utf-8"))
    calibration["method"] = "tampered-after-gate-approval"
    robot.config.calibration_path.write_text(json.dumps(calibration), encoding="utf-8")
    with pytest.raises(StateConflict, match="not bound to the active calibration"):
        robot.connect()
    assert created == []


def test_invalid_action_shape_disables_arm(tmp_path: Path):
    robot, _, created = _robot(tmp_path, mode="motion")
    robot.connect()
    with pytest.raises(ValidationError, match="keys mismatch"):
        robot.send_action({"L1.pos": 0.0})
    assert "DisableArm" in created[0].calls
    robot.disconnect()


def test_sdk_feedback_exception_disables_arm(tmp_path: Path):
    robot, _, created = _robot(tmp_path, mode="motion")
    robot.connect()
    created[0].feedback_exception = True
    with pytest.raises(RuntimeError, match="fake SDK feedback exception"):
        robot.get_observation()
    assert "DisableArm" in created[0].calls
    robot.disconnect()


def test_fake_a3_records_finalizes_reloads_and_replays_lerobot_dataset(tmp_path: Path):
    robot, _, _ = _robot(tmp_path / "record-config", mode="motion")
    robot.connect()
    features = {
        **hw_to_dataset_features(robot.observation_features, OBS_STR, use_video=False),
        **hw_to_dataset_features(robot.action_features, ACTION, use_video=False),
    }
    dataset_root = tmp_path / "dataset"
    dataset = LeRobotDataset.create(
        repo_id="local/a3-fake-plugin-smoke",
        fps=10,
        features=features,
        root=dataset_root,
        robot_type="a3",
        use_videos=False,
    )
    for step in range(3):
        action = {f"L{index}.pos": 0.01 * (step + 1) for index in range(1, 8)}
        frame = {
            **build_dataset_frame(features, robot.get_observation(), OBS_STR),
            **build_dataset_frame(features, robot.send_action(action), ACTION),
            "task": "synthetic fake A3 plugin smoke",
        }
        dataset.add_frame(frame)
    dataset.save_episode()
    dataset.finalize()
    robot.disconnect()

    reloaded = LeRobotDataset(repo_id="local/a3-fake-plugin-smoke", root=dataset_root)
    assert len(reloaded) == 3
    replay_robot, _, replay_vendors = _robot(tmp_path / "replay-config", mode="motion")
    replay_robot.connect()
    action_keys = list(replay_robot.action_features)
    for index in range(len(reloaded)):
        replay_robot.send_action(dict(zip(action_keys, reloaded[index][ACTION].tolist())))
    replay_robot.disconnect()
    sent = [call for call in replay_vendors[0].calls if isinstance(call, tuple)]
    assert len(sent) == 3
    for step, (method, values) in enumerate(sent, start=1):
        assert method == "JointCtrlList"
        assert values == pytest.approx([0.01 * step] * 7)
