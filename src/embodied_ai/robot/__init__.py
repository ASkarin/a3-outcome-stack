"""Safe, backend-neutral robot interface for the EduLite A3 project."""

from .backend import RobotBackend, SafeRobot
from .clock import ManualClock, MonotonicClock
from .mock import MockBackend
from .replay import ReplayBackend
from .safety import SafetyState, StopReason
from .types import ActionEnvelope, ActionReceipt, Observation

__all__ = [
    "ActionEnvelope",
    "ActionReceipt",
    "ManualClock",
    "MockBackend",
    "MonotonicClock",
    "Observation",
    "ReplayBackend",
    "RobotBackend",
    "SafeRobot",
    "SafetyState",
    "StopReason",
]
