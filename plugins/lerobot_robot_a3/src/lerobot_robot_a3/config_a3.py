"""LeRobot configuration for a directly connected EduLite A3."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from lerobot.cameras.configs import CameraConfig
from lerobot.robots.config import RobotConfig


@RobotConfig.register_subclass("a3")
@dataclass(kw_only=True)
class A3RobotConfig(RobotConfig):
    """Configuration loaded by ``lerobot-record``, teleoperate, and replay."""

    id: str = "a3"
    can_interface: str = "can0"
    calibration_path: Path = Path("configs/robot/a3_calibration.template.json")
    safety_path: Path = Path("configs/robot/a3_safety.template.json")
    hardware_gate_path: Path = Path("configs/robot/a3_hardware_gate.template.json")
    execution_mode: Literal["read_only", "motion"] = "read_only"
    cameras: dict[str, CameraConfig] = field(default_factory=dict)

    def __post_init__(self) -> None:
        super().__post_init__()
        if not self.can_interface:
            raise ValueError("can_interface must be non-empty")
        if self.execution_mode not in {"read_only", "motion"}:
            raise ValueError("execution_mode must be read_only or motion")
