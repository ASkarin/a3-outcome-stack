from __future__ import annotations

import math
from pathlib import Path

import pytest

from embodied_ai.ops.canonical import load_json
from embodied_ai.ops.errors import StateConflict, ValidationError
from embodied_ai.robot.backend import SafeRobot
from embodied_ai.robot.clock import ManualClock
from embodied_ai.robot.mock import MockBackend
from embodied_ai.robot.safety import _ALLOWED, SafetyState, SafetySupervisor, StopReason
from embodied_ai.robot.types import ActionEnvelope

ROOT = Path(__file__).parents[1]


def make_robot():
    config = load_json(ROOT / "configs/robot/a3_mock_test.json")
    clock = ManualClock(
        current_ns=config["start_monotonic_ns"], domain_id=config["clock_domain_id"]
    )
    backend = MockBackend(clock, config)
    robot = SafeRobot(
        backend,
        clock,
        watchdog_timeout_ns=config["watchdog_timeout_ns"],
        joint_lower=config["joint_limits"]["lower"],
        joint_upper=config["joint_limits"]["upper"],
    )
    robot.connect()
    robot.enable()
    return config, clock, backend, robot


def action(clock, *, sequence=0, values=(0.1,) * 7, deadline_offset=100):
    now = clock.now_ns()
    return ActionEnvelope(
        tuple(values), sequence, now, now + deadline_offset, clock.domain_id
    )


def test_valid_action_enters_active_and_watchdog_latches_safe_stop():
    config, clock, backend, robot = make_robot()
    robot.send_action(action(clock, deadline_offset=config["watchdog_timeout_ns"]))
    assert robot.state == SafetyState.ACTIVE
    clock.advance_ns(config["watchdog_timeout_ns"] + 1)
    assert robot.check_watchdog() is False
    assert robot.state == SafetyState.SAFE_STOP
    assert backend.stop_reasons[-1] == StopReason.WATCHDOG_TIMEOUT.value
    with pytest.raises(StateConflict):
        robot.send_action(action(clock, sequence=1))
    with pytest.raises(StateConflict):
        robot.reset(operator_acknowledged=False)
    robot.reset(operator_acknowledged=True)
    assert robot.state == SafetyState.DISABLED


@pytest.mark.parametrize(
    ("candidate", "expected_reason"),
    [
        (
            ActionEnvelope(
                (math.nan,) * 7, 0, 1000000000, 1000000100, "stage1a-mock-clock-v1"
            ),
            StopReason.NONFINITE,
        ),
        (
            ActionEnvelope(
                (0.0,) * 6, 0, 1000000000, 1000000100, "stage1a-mock-clock-v1"
            ),
            StopReason.SHAPE_MISMATCH,
        ),
        (
            ActionEnvelope(
                (2.0,) * 7, 0, 1000000000, 1000000100, "stage1a-mock-clock-v1"
            ),
            StopReason.LIMIT_VIOLATION,
        ),
        (
            ActionEnvelope(
                (0.0,) * 7, 1, 1000000000, 1000000100, "stage1a-mock-clock-v1"
            ),
            StopReason.OUT_OF_ORDER,
        ),
        (
            ActionEnvelope((0.0,) * 7, 0, 1000000000, 1000000100, "wrong-clock"),
            StopReason.CLOCK_DOMAIN_MISMATCH,
        ),
    ],
)
def test_invalid_actions_latch_expected_safe_stop(candidate, expected_reason):
    _, _, backend, robot = make_robot()
    with pytest.raises((ValidationError, StateConflict)):
        robot.send_action(candidate)
    assert robot.state == SafetyState.SAFE_STOP
    assert backend.stop_reasons[-1] == expected_reason.value


def test_stale_action_latches_safe_stop():
    _, clock, backend, robot = make_robot()
    candidate = action(clock, deadline_offset=1)
    clock.advance_ns(2)
    with pytest.raises(ValidationError, match="stale"):
        robot.send_action(candidate)
    assert robot.state == SafetyState.SAFE_STOP
    assert backend.stop_reasons[-1] == StopReason.STALE_ACTION.value


def test_feedback_timeout_latches_safe_stop():
    _, _, backend, robot = make_robot()
    backend.fail_next_feedback()
    with pytest.raises(RuntimeError, match="feedback timeout"):
        robot.get_observation()
    assert robot.state == SafetyState.SAFE_STOP
    assert backend.stop_reasons[-1] == StopReason.FEEDBACK_TIMEOUT.value


def test_device_fault_latches_fault_and_requires_explicit_reset():
    _, _, backend, robot = make_robot()
    backend.set_joint_fault(2, 7)
    with pytest.raises(StateConflict, match="joint fault"):
        robot.get_observation()
    assert robot.state == SafetyState.FAULT
    with pytest.raises(StateConflict, match="unhealthy"):
        robot.reset(operator_acknowledged=True)
    backend.clear_faults()
    robot.reset(operator_acknowledged=True)
    assert robot.state == SafetyState.DISABLED


def test_emergency_stop_is_latched_and_not_automatic():
    _, _, backend, robot = make_robot()
    robot.emergency_stop()
    assert robot.state == SafetyState.E_STOP
    assert backend.stop_reasons[-1] == StopReason.E_STOP.value
    with pytest.raises(StateConflict):
        robot.enable()
    robot.reset(operator_acknowledged=True)
    assert robot.state == SafetyState.DISABLED


def test_clock_regression_latches_safe_stop():
    _, clock, backend, robot = make_robot()
    clock.current_ns -= 1
    with pytest.raises(StateConflict, match="regressed"):
        robot.send_action(action(clock))
    assert robot.state == SafetyState.SAFE_STOP
    assert backend.stop_reasons[-1] == StopReason.CLOCK_REGRESSION.value
    transition = robot.supervisor.transitions[-1]
    assert (
        transition["monotonic_ns"] >= robot.supervisor.transitions[-2]["monotonic_ns"]
    )
    assert "observed_regressed_monotonic_ns" in transition


def test_complete_safety_transition_table_accepts_only_registered_edges():
    for source in SafetyState:
        for target in SafetyState:
            supervisor = SafetySupervisor(ManualClock(domain_id="transition-table"))
            supervisor._state = source
            if target in _ALLOWED[source]:
                supervisor.transition(target, "table_test")
                assert supervisor.state == target
            else:
                with pytest.raises(StateConflict, match="illegal safety transition"):
                    supervisor.transition(target, "table_test")
