"""Unix-domain control service for a single supervised operator session."""

from __future__ import annotations

import json
import os
import select
import socket
import socketserver
import stat
import struct
import sys
from pathlib import Path
from typing import Any, Mapping, TextIO

from a3_outcome_stack.ops.canonical import load_json, sha256_file
from a3_outcome_stack.ops.errors import OpsError, StateConflict, ValidationError

from .backend import SafeRobot
from .clock import MonotonicClock
from .mock import MockBackend
from .session import OperatorSession
from .types import ActionEnvelope

MAX_REQUEST_BYTES = 1024 * 1024


def _peer_uid(connection: socket.socket) -> int:
    if not hasattr(socket, "SO_PEERCRED"):
        raise StateConflict("Unix peer credentials are unavailable on this platform")
    credentials = connection.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, 12)
    _, uid, _ = struct.unpack("3i", credentials)
    return uid


def _dispatch(
    session: OperatorSession, operator_uid: int, request: Mapping[str, Any]
) -> dict[str, Any]:
    if not isinstance(request, Mapping):
        raise ValidationError("control request must be a JSON object")
    operation = request.get("operation")
    if operation == "status":
        return {"status": "ok", **session.status()}
    if operation == "heartbeat":
        session.heartbeat(operator_uid)
        return {"status": "ok", **session.status()}
    if operation == "enable":
        session.enable(operator_uid)
        return {"status": "ok", **session.status()}
    if operation == "disable":
        session.disable(operator_uid)
        return {"status": "ok", **session.status()}
    if operation == "safe_stop":
        session.request_safe_stop(operator_uid)
        return {"status": "ok", **session.status()}
    if operation == "observation":
        return {
            "status": "ok",
            "observation": session.get_observation(operator_uid).to_dict(),
        }
    if operation == "action":
        action_payload = request.get("action")
        if not isinstance(action_payload, Mapping):
            raise ValidationError("action operation requires an action object")
        action = ActionEnvelope.from_dict(action_payload)
        receipt = session.send_action(operator_uid, action)
        return {"status": "ok", "receipt": receipt.to_dict()}
    raise ValidationError("unsupported control operation")


class _ControlHandler(socketserver.BaseRequestHandler):
    def handle(self) -> None:
        control_server = self.server
        if not isinstance(control_server, ControlUnixServer):
            raise RuntimeError("invalid control server")
        connection = self.request
        if not isinstance(connection, socket.socket):
            raise RuntimeError("invalid Unix socket request")
        operator_uid = _peer_uid(connection)
        connected = False
        buffered = b""
        try:
            control_server.session.connect_client(operator_uid)
            connected = True
            while True:
                control_server.session.check_liveness()
                readable, _, _ = select.select([connection], [], [], 0.1)
                if not readable:
                    continue
                chunk = connection.recv(64 * 1024)
                if not chunk:
                    break
                buffered += chunk
                while True:
                    line_end = buffered.find(b"\n")
                    if line_end < 0:
                        if len(buffered) > MAX_REQUEST_BYTES:
                            raise ValidationError("control request exceeds size limit")
                        break
                    line = buffered[:line_end]
                    buffered = buffered[line_end + 1 :]
                    if len(line) > MAX_REQUEST_BYTES:
                        raise ValidationError("control request exceeds size limit")
                    try:
                        request = json.loads(line)
                        response = _dispatch(control_server.session, operator_uid, request)
                    except (json.JSONDecodeError, UnicodeDecodeError, OpsError) as exc:
                        response = {
                            "status": "error",
                            "error": type(exc).__name__,
                            "message": str(exc),
                        }
                    connection.sendall(
                        json.dumps(response, ensure_ascii=False, sort_keys=True).encode("utf-8")
                        + b"\n"
                    )
        finally:
            if connected:
                control_server.session.disconnect_client(operator_uid)


class ControlUnixServer(socketserver.UnixStreamServer):
    """One-client AF_UNIX server; it never listens on an IP interface."""

    allow_reuse_address = False

    def __init__(
        self,
        socket_path: str | Path,
        session: OperatorSession,
        *,
        socket_group: str | None = None,
    ):
        path = Path(socket_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        group_id: int | None = None
        if socket_group:
            import grp

            try:
                group_id = grp.getgrnam(socket_group).gr_gid
            except KeyError as exc:
                raise ValidationError(
                    f"control socket group does not exist: {socket_group}"
                ) from exc
            os.chown(path.parent, -1, group_id)
            os.chmod(path.parent, 0o750)
        if path.exists() or path.is_symlink():
            metadata = path.lstat()
            if not stat.S_ISSOCK(metadata.st_mode):
                raise StateConflict("control socket path exists and is not a socket")
            probe = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            try:
                probe.connect(str(path))
            except OSError:
                path.unlink()
            else:
                raise StateConflict("another control service is already listening")
            finally:
                probe.close()
        self.session = session
        self.socket_path = path
        super().__init__(str(path), _ControlHandler)
        if group_id is not None:
            os.chown(path, -1, group_id)
        os.chmod(path, 0o660)

    def service_actions(self) -> None:
        self.session.check_liveness()

    def server_close(self) -> None:
        super().server_close()
        if self.socket_path.exists() and stat.S_ISSOCK(self.socket_path.lstat().st_mode):
            self.socket_path.unlink()


def create_mock_control_server(
    *,
    config_path: str | Path,
    permit_path: str | Path,
    socket_path: str | Path,
    git_commit: str,
    calibration_path: str | Path,
    safety_path: str | Path,
    socket_group: str | None,
    require_root_permit: bool = True,
) -> ControlUnixServer:
    config = load_json(config_path)
    clock = MonotonicClock(domain_id=config.get("clock_domain_id"))
    backend = MockBackend(clock, config)
    robot = SafeRobot(
        backend,
        clock,
        watchdog_timeout_ns=config.get("watchdog_timeout_ns"),
        joint_lower=config.get("joint_limits", {}).get("lower", []),
        joint_upper=config.get("joint_limits", {}).get("upper", []),
    )
    robot.connect()
    session = OperatorSession(
        robot,
        clock,
        permit_path=permit_path,
        git_commit=git_commit,
        calibration_sha256=sha256_file(calibration_path),
        safety_sha256=sha256_file(safety_path),
        require_root_permit=require_root_permit,
    )
    return ControlUnixServer(socket_path, session, socket_group=socket_group)


def serve_mock_control(
    *,
    config_path: str | Path,
    permit_path: str | Path,
    socket_path: str | Path,
    git_commit: str,
    calibration_path: str | Path,
    safety_path: str | Path,
    socket_group: str | None,
) -> dict[str, Any]:
    server = create_mock_control_server(
        config_path=config_path,
        permit_path=permit_path,
        socket_path=socket_path,
        git_commit=git_commit,
        calibration_path=calibration_path,
        safety_path=safety_path,
        socket_group=socket_group,
    )
    try:
        server.serve_forever(poll_interval=0.1)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return {"status": "stopped", "hardware_verified": False}


def run_control_client(
    socket_path: str | Path,
    *,
    input_stream: TextIO = sys.stdin,
    output_stream: TextIO = sys.stdout,
) -> dict[str, Any]:
    path = Path(socket_path)
    connection = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    connection.connect(str(path))
    responses = 0
    try:
        reader = connection.makefile("r", encoding="utf-8")
        writer = connection.makefile("w", encoding="utf-8")
        try:
            for line in input_stream:
                if not line.strip():
                    continue
                try:
                    request = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValidationError(f"client input is not valid JSON: {exc}") from exc
                if not isinstance(request, dict):
                    raise ValidationError("client input must contain JSON objects")
                writer.write(json.dumps(request, separators=(",", ":")) + "\n")
                writer.flush()
                response_line = reader.readline()
                if not response_line:
                    raise StateConflict("control service closed the connection")
                output_stream.write(response_line)
                output_stream.flush()
                responses += 1
        finally:
            writer.close()
            reader.close()
    finally:
        connection.close()
    return {"status": "closed", "responses": responses}
