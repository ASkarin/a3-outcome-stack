"""Auto-discoverable LeRobot plugin for the EduLite A3."""

from .config_a3 import A3RobotConfig
from .robot_a3 import A3Robot

__all__ = ["A3Robot", "A3RobotConfig"]
