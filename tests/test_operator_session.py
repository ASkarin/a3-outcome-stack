from __future__ import annotations

import json
import os
import socket
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from a3_outcome_stack.ops.canonical import load_json, sha256_bytes
from a3_outcome_stack.ops.errors import IntegrityError, StateConflict, ValidationError
from a3_outcome_stack.robot.backend import SafeRobot
from a3_outcome_stack.robot.clock import ManualClock
from a3_outcome_stack.robot.control import ControlUnixServer
from a3_outcome_stack.robot.mock import MockBackend
from a3_outcome_stack.robot.safety import SafetyState, StopReason
from a3_outcome_stack.robot.session import (
    OperatorPermit,
    OperatorSession,
    build_operator_permit,
    load_operator_permit,
)

ROOT = Path(__file__).parents[1]
COMMIT = "1" * 40
CALIBRATION_HASH = sha256_bytes(b"calibration")
SAFETY_HASH = sha256_bytes(b"safety")


def _write_permit(
    path: Path,
    *,
    operator_uid: int,
    now: datetime,
    execution_mode: str = "simulation",
    expires_delta: timedelta = timedelta(minutes=5),
    heartbeat_timeout_ns: int = 100,
) -> dict:
    permit = build_operator_permit(
        session_id="TEST-SESSION",
        operator_uid=operator_uid,
        execution_mode=execution_mode,
        git_commit=COMMIT,
        calibration_sha256=CALIBRATION_HASH,
        safety_sha256=SAFETY_HASH,
        issued_at=now - timedelta(seconds=1),
        expires_at=now + expires_delta,
        heartbeat_timeout_ns=heartbeat_timeout_ns,
        real_enable_authorized=execution_mode == "real",
    )
    path.write_text(json.dumps(permit), encoding="utf-8")
    path.chmod(0o640)
    return permit


def _make_session(tmp_path: Path, *, now: datetime, permit_exists: bool = True):
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
    permit_path = tmp_path / "permit.json"
    if permit_exists:
        _write_permit(permit_path, operator_uid=1000, now=now)
    session = OperatorSession(
        robot,
        clock,
        permit_path=permit_path,
        git_commit=COMMIT,
        calibration_sha256=CALIBRATION_HASH,
        safety_sha256=SAFETY_HASH,
        require_root_permit=False,
        utc_now=lambda: now,
    )
    return clock, backend, robot, session, permit_path


def test_operator_permit_is_sealed_and_rejects_tampering(tmp_path: Path):
    now = datetime(2026, 8, 4, tzinfo=timezone.utc)
    permit_path = tmp_path / "permit.json"
    permit = _write_permit(permit_path, operator_uid=1000, now=now)
    loaded = load_operator_permit(permit_path, require_root_owner=False)
    assert loaded == OperatorPermit.from_dict(permit)

    permit["git_commit"] = "2" * 40
    permit_path.write_text(json.dumps(permit), encoding="utf-8")
    with pytest.raises(IntegrityError, match="permit_sha256 mismatch"):
        load_operator_permit(permit_path, require_root_owner=False)


def test_operator_permit_rejects_group_writable_file(tmp_path: Path):
    now = datetime(2026, 8, 4, tzinfo=timezone.utc)
    permit_path = tmp_path / "permit.json"
    _write_permit(permit_path, operator_uid=1000, now=now)
    permit_path.chmod(0o660)
    with pytest.raises(ValidationError, match="group/world writable"):
        load_operator_permit(permit_path, require_root_owner=False)


def test_missing_administrator_permit_refuses_enable(tmp_path: Path):
    now = datetime(2026, 8, 4, tzinfo=timezone.utc)
    _, _, robot, session, _ = _make_session(tmp_path, now=now, permit_exists=False)
    session.connect_client(1000)
    with pytest.raises(ValidationError, match="cannot open operator permit"):
        session.enable(1000)
    assert robot.state == SafetyState.DISABLED


def test_expired_permit_refuses_enable(tmp_path: Path):
    now = datetime(2026, 8, 4, tzinfo=timezone.utc)
    _, _, robot, session, permit_path = _make_session(tmp_path, now=now)
    _write_permit(
        permit_path,
        operator_uid=1000,
        now=now - timedelta(minutes=10),
        expires_delta=timedelta(minutes=1),
    )
    session.connect_client(1000)
    with pytest.raises(StateConflict, match="expired"):
        session.enable(1000)
    assert robot.state == SafetyState.DISABLED


def test_heartbeat_timeout_latches_safe_stop(tmp_path: Path):
    now = datetime(2026, 8, 4, tzinfo=timezone.utc)
    clock, backend, robot, session, _ = _make_session(tmp_path, now=now)
    session.connect_client(1000)
    session.enable(1000)
    clock.advance_ns(101)
    assert session.check_liveness() is False
    assert robot.state == SafetyState.SAFE_STOP
    assert backend.stop_reasons[-1] == StopReason.OPERATOR_HEARTBEAT_TIMEOUT.value


def test_disconnect_latches_safe_stop_and_operator_cannot_reset(tmp_path: Path):
    now = datetime(2026, 8, 4, tzinfo=timezone.utc)
    _, backend, robot, session, _ = _make_session(tmp_path, now=now)
    session.connect_client(1000)
    session.enable(1000)
    session.disconnect_client(1000)
    assert robot.state == SafetyState.SAFE_STOP
    assert backend.stop_reasons[-1] == StopReason.OPERATOR_DISCONNECT.value
    assert session.status()["reset_available_to_operator"] is False


def test_real_permit_cannot_enable_unverified_backend(tmp_path: Path):
    now = datetime(2026, 8, 4, tzinfo=timezone.utc)
    _, _, robot, session, permit_path = _make_session(tmp_path, now=now)
    _write_permit(
        permit_path,
        operator_uid=1000,
        now=now,
        execution_mode="real",
    )
    session.connect_client(1000)
    with pytest.raises(StateConflict, match="hardware_verified"):
        session.enable(1000)
    assert robot.state == SafetyState.DISABLED


@pytest.mark.skipif(
    os.name != "posix" or not hasattr(socket, "SO_PEERCRED"),
    reason="peer-credential Unix sockets require Linux",
)
def test_unix_socket_disconnect_latches_safe_stop(tmp_path: Path):
    now = datetime(2026, 8, 4, tzinfo=timezone.utc)
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
    permit_path = tmp_path / "permit.json"
    _write_permit(permit_path, operator_uid=os.getuid(), now=now)
    session = OperatorSession(
        robot,
        clock,
        permit_path=permit_path,
        git_commit=COMMIT,
        calibration_sha256=CALIBRATION_HASH,
        safety_sha256=SAFETY_HASH,
        require_root_permit=False,
        utc_now=lambda: now,
    )
    socket_path = tmp_path / "control.sock"
    server = ControlUnixServer(socket_path, session)
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01})
    thread.start()
    connection: socket.socket | None = None
    reader = None
    writer = None
    try:
        connection = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        connection.connect(str(socket_path))
        reader = connection.makefile("r", encoding="utf-8")
        writer = connection.makefile("w", encoding="utf-8")
        writer.write('{"operation":"enable"}\n')
        writer.flush()
        assert json.loads(reader.readline())["state"] == SafetyState.READY.value
        writer.close()
        reader.close()
        connection.close()
        deadline = time.monotonic() + 2
        while robot.state != SafetyState.SAFE_STOP and time.monotonic() < deadline:
            time.sleep(0.01)
        assert robot.state == SafetyState.SAFE_STOP
        assert backend.stop_reasons[-1] == StopReason.OPERATOR_DISCONNECT.value
    finally:
        if writer is not None and not writer.closed:
            writer.close()
        if reader is not None and not reader.closed:
            reader.close()
        if connection is not None:
            connection.close()
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


@pytest.mark.skipif(
    os.name != "posix" or not hasattr(socket, "SO_PEERCRED"),
    reason="peer-credential Unix sockets require Linux",
)
def test_unix_socket_idle_client_triggers_heartbeat_safe_stop(tmp_path: Path):
    now = datetime(2026, 8, 4, tzinfo=timezone.utc)
    clock, backend, robot, session, permit_path = _make_session(tmp_path, now=now)
    _write_permit(
        permit_path,
        operator_uid=os.getuid(),
        now=now,
        heartbeat_timeout_ns=100,
    )
    socket_path = tmp_path / "control.sock"
    server = ControlUnixServer(socket_path, session)
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01})
    thread.start()
    connection: socket.socket | None = None
    try:
        connection = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        connection.connect(str(socket_path))
        connection.sendall(b'{"operation":"enable"}\n')
        assert json.loads(connection.recv(4096))["state"] == SafetyState.READY.value
        clock.advance_ns(101)
        deadline = time.monotonic() + 2
        while robot.state != SafetyState.SAFE_STOP and time.monotonic() < deadline:
            time.sleep(0.01)
        assert robot.state == SafetyState.SAFE_STOP
        assert backend.stop_reasons[-1] == StopReason.OPERATOR_HEARTBEAT_TIMEOUT.value
    finally:
        if connection is not None:
            connection.close()
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
