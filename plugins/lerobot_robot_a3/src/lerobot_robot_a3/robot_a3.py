"""Direct, in-process LeRobot adapter for the official EduLite A3 SDK."""

from __future__ import annotations

import logging
import threading
from functools import cached_property
from pathlib import Path
from typing import Any, Callable, Mapping

from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.robots.robot import Robot

from a3_outcome_stack.ops.canonical import load_json, sha256_file
from a3_outcome_stack.ops.errors import StateConflict, ValidationError
from a3_outcome_stack.robot.a3_sdk import (
    A3SdkBackend,
    validate_calibration,
    validate_hardware_gate,
    validate_hardware_ready,
)
from a3_outcome_stack.robot.backend import SafeRobot
from a3_outcome_stack.robot.clock import Clock, MonotonicClock
from a3_outcome_stack.robot.safety import SafetyState, StopReason
from a3_outcome_stack.robot.types import JOINT_NAMES, ActionEnvelope

from .config_a3 import A3RobotConfig

logger = logging.getLogger(__name__)
READ_ONLY_WATCHDOG_TIMEOUT_NS = 1_000_000_000


class A3Robot(Robot):
    """LeRobot robot that the unique administrator runs from an immutable release."""

    config_class = A3RobotConfig
    name = "a3"

    def __init__(
        self,
        config: A3RobotConfig,
        *,
        vendor_factory: Callable[..., Any] | None = None,
        clock: Clock | None = None,
        camera_factory: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
        watchdog_poll_seconds: float = 0.01,
    ):
        super().__init__(config)
        if watchdog_poll_seconds <= 0:
            raise ValueError("watchdog_poll_seconds must be positive")
        self.config = config
        self._vendor_factory = vendor_factory
        self._clock = clock or MonotonicClock()
        self._camera_factory = camera_factory or make_cameras_from_configs
        self._watchdog_poll_seconds = watchdog_poll_seconds
        self._safe_robot: SafeRobot | None = None
        self._cameras: dict[str, Any] = {}
        self._sequence_id = 0
        self._watchdog_stop = threading.Event()
        self._watchdog_thread: threading.Thread | None = None

    @cached_property
    def observation_features(self) -> dict[str, type | tuple[int, ...]]:
        features: dict[str, type | tuple[int, ...]] = {
            **{f"{joint}.pos": float for joint in JOINT_NAMES},
            **{f"{joint}.vel": float for joint in JOINT_NAMES},
        }
        for name, camera in self.config.cameras.items():
            if getattr(camera, "use_rgb", True):
                features[name] = (camera.height, camera.width, 3)
            if getattr(camera, "use_depth", False):
                features[f"{name}_depth"] = (camera.height, camera.width, 1)
        return features

    @cached_property
    def action_features(self) -> dict[str, type]:
        return {f"{joint}.pos": float for joint in JOINT_NAMES}

    @property
    def is_connected(self) -> bool:
        robot_connected = self._safe_robot is not None and self._safe_robot.backend.is_connected
        return robot_connected and all(camera.is_connected for camera in self._cameras.values())

    @property
    def is_calibrated(self) -> bool:
        try:
            calibration = self._load_object(self.config.calibration_path, "calibration")
            validate_calibration(calibration)
        except (StateConflict, ValidationError):
            return False
        return True

    @staticmethod
    def _load_object(path: Path, label: str) -> Mapping[str, Any]:
        document = load_json(path)
        if not isinstance(document, Mapping):
            raise ValidationError(f"A3 {label} must be a JSON object")
        return document

    def calibrate(self) -> None:
        if not self.is_calibrated:
            raise StateConflict(
                "A3 calibration is absent or unfrozen; complete the approved external "
                "calibration workflow before connecting"
            )

    def configure(self) -> None:
        """The official SDK receives the frozen limits when the backend is constructed."""

    def _start_watchdog(self) -> None:
        self._watchdog_stop.clear()

        def monitor() -> None:
            while not self._watchdog_stop.wait(self._watchdog_poll_seconds):
                robot = self._safe_robot
                if robot is None:
                    return
                try:
                    if not robot.check_watchdog():
                        logger.error("A3 action watchdog expired; arm disabled")
                        return
                except Exception:
                    logger.exception("A3 watchdog failed; requesting a safe stop")
                    try:
                        robot.request_safe_stop()
                    except Exception:
                        logger.exception("A3 safe-stop request failed")
                    return

        self._watchdog_thread = threading.Thread(
            target=monitor,
            name="a3-action-watchdog",
            daemon=True,
        )
        self._watchdog_thread.start()

    def connect(self, calibrate: bool = True) -> None:
        if self._safe_robot is not None:
            raise StateConflict("A3 robot is already connected")
        # Calibration is deliberately non-interactive. The parameter is accepted only
        # for compatibility with LeRobot's base interface.
        del calibrate
        calibration = self._load_object(self.config.calibration_path, "calibration")
        safety = self._load_object(self.config.safety_path, "safety")
        if self.config.execution_mode == "motion":
            frozen = validate_hardware_ready(calibration, safety)
            watchdog_timeout_ns = int(frozen["watchdog_timeout_ns"])
            joint_lower = frozen["lower"]
            joint_upper = frozen["upper"]
            gate = self._load_object(self.config.hardware_gate_path, "hardware gate")
            validate_hardware_gate(
                gate,
                calibration_sha256=sha256_file(self.config.calibration_path),
                safety_sha256=sha256_file(self.config.safety_path),
            )
        else:
            validate_calibration(calibration)
            watchdog_timeout_ns = READ_ONLY_WATCHDOG_TIMEOUT_NS
            joint_lower = None
            joint_upper = None

        backend = A3SdkBackend(
            self._clock,
            {"can_interface": self.config.can_interface},
            calibration,
            safety,
            vendor_factory=self._vendor_factory,
            apply_safety_limits=self.config.execution_mode == "motion",
        )
        robot = SafeRobot(
            backend,
            self._clock,
            watchdog_timeout_ns=watchdog_timeout_ns,
            joint_lower=joint_lower,
            joint_upper=joint_upper,
        )
        cameras = self._camera_factory(self.config.cameras)
        try:
            robot.connect()
            for camera in cameras.values():
                camera.connect()
            self._safe_robot = robot
            self._cameras = cameras
            if self.config.execution_mode == "motion":
                robot.enable()
                self._start_watchdog()
        except Exception:
            self._watchdog_stop.set()
            for camera in cameras.values():
                if getattr(camera, "is_connected", False):
                    camera.disconnect()
            if backend.is_connected:
                backend.disconnect()
            self._safe_robot = None
            self._cameras = {}
            raise
        self._sequence_id = 0

    def _require_robot(self) -> SafeRobot:
        if self._safe_robot is None or not self._safe_robot.backend.is_connected:
            raise StateConflict("A3 robot is not connected")
        return self._safe_robot

    def get_observation(self) -> dict[str, Any]:
        robot = self._require_robot()
        observation = robot.get_observation()
        result: dict[str, Any] = {}
        for index, joint in enumerate(JOINT_NAMES):
            result[f"{joint}.pos"] = observation.joint_position[index]
            result[f"{joint}.vel"] = observation.joint_velocity[index]
        for name, camera in self._cameras.items():
            if getattr(self.config.cameras[name], "use_rgb", True):
                result[name] = camera.async_read()
            if getattr(self.config.cameras[name], "use_depth", False):
                result[f"{name}_depth"] = camera.async_read_depth()
        return result

    def send_action(self, action: dict[str, Any]) -> dict[str, float]:
        if self.config.execution_mode != "motion":
            raise StateConflict("A3 actions require execution_mode=motion")
        robot = self._require_robot()
        expected = set(self.action_features)
        if set(action) != expected:
            robot.request_safe_stop(StopReason.SHAPE_MISMATCH)
            raise ValidationError(
                f"A3 action keys mismatch: missing={sorted(expected - set(action))}, "
                f"extra={sorted(set(action) - expected)}"
            )
        try:
            values = tuple(float(action[f"{joint}.pos"]) for joint in JOINT_NAMES)
        except (TypeError, ValueError) as exc:
            robot.request_safe_stop(StopReason.REJECTED_ACTION)
            raise ValidationError("A3 action values must be numeric") from exc
        now = self._clock.now_ns()
        envelope = ActionEnvelope(
            values=values,
            sequence_id=self._sequence_id,
            created_monotonic_ns=now,
            deadline_monotonic_ns=now + robot.watchdog.timeout_ns,
            clock_domain_id=self._clock.domain_id,
        )
        robot.send_action(envelope)
        self._sequence_id += 1
        return {f"{joint}.pos": values[index] for index, joint in enumerate(JOINT_NAMES)}

    def check_watchdog(self) -> bool:
        return self._require_robot().check_watchdog()

    def disconnect(self) -> None:
        self._watchdog_stop.set()
        if self._watchdog_thread is not None:
            self._watchdog_thread.join(timeout=max(1.0, self._watchdog_poll_seconds * 2))
            self._watchdog_thread = None
        for camera in self._cameras.values():
            if getattr(camera, "is_connected", False):
                camera.disconnect()
        robot = self._safe_robot
        if robot is not None:
            if robot.state in {SafetyState.READY, SafetyState.ACTIVE}:
                robot.disable()
            robot.disconnect()
        self._cameras = {}
        self._safe_robot = None
