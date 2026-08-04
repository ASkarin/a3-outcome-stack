"""Root-authorized, time-bounded operator sessions for local robot control."""

from __future__ import annotations

import json
import os
import re
import stat
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping

from a3_outcome_stack.ops.canonical import (
    seal_document,
    validate_sha256,
    verify_sealed_document,
)
from a3_outcome_stack.ops.errors import StateConflict, ValidationError

from .backend import SafeRobot
from .clock import Clock
from .safety import SafetyState, StopReason
from .types import ActionEnvelope, ActionReceipt, Observation

PERMIT_SCHEMA = "a3-operator-permit-v1"
_SESSION_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}\Z")
_GIT_COMMIT = re.compile(r"[0-9a-f]{40}\Z")
_PERMIT_FIELDS = {
    "schema_version",
    "session_id",
    "operator_uid",
    "execution_mode",
    "git_commit",
    "calibration_sha256",
    "safety_sha256",
    "issued_at_utc",
    "expires_at_utc",
    "heartbeat_timeout_ns",
    "real_enable_authorized",
    "permit_sha256",
}


def _parse_utc(value: Any, field: str) -> datetime:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise ValidationError(f"{field} must be a UTC timestamp ending in Z")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise ValidationError(f"{field} is not a valid timestamp") from exc
    if parsed.utcoffset() is None or parsed.utcoffset().total_seconds() != 0:
        raise ValidationError(f"{field} must use UTC")
    return parsed.astimezone(timezone.utc)


def _format_utc(value: datetime) -> str:
    if value.tzinfo is None:
        raise ValidationError("operator permit timestamps must be timezone-aware")
    return value.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


@dataclass(frozen=True)
class OperatorPermit:
    session_id: str
    operator_uid: int
    execution_mode: str
    git_commit: str
    calibration_sha256: str
    safety_sha256: str
    issued_at_utc: str
    expires_at_utc: str
    heartbeat_timeout_ns: int
    real_enable_authorized: bool
    permit_sha256: str

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "OperatorPermit":
        if not isinstance(value, Mapping):
            raise ValidationError("operator permit must be a JSON object")
        if set(value) != _PERMIT_FIELDS:
            missing = sorted(_PERMIT_FIELDS - set(value))
            extra = sorted(set(value) - _PERMIT_FIELDS)
            raise ValidationError(
                f"operator permit fields mismatch: missing={missing}, extra={extra}"
            )
        document = dict(value)
        if document.get("schema_version") != PERMIT_SCHEMA:
            raise ValidationError("unsupported operator permit schema")
        verify_sealed_document(document, "permit_sha256")

        session_id = document["session_id"]
        if not isinstance(session_id, str) or not _SESSION_ID.fullmatch(session_id):
            raise ValidationError("invalid operator permit session_id")
        operator_uid = document["operator_uid"]
        if not isinstance(operator_uid, int) or isinstance(operator_uid, bool) or operator_uid < 0:
            raise ValidationError("operator_uid must be a non-negative integer")
        execution_mode = document["execution_mode"]
        if execution_mode not in {"simulation", "real"}:
            raise ValidationError("execution_mode must be simulation or real")
        git_commit = document["git_commit"]
        if not isinstance(git_commit, str) or not _GIT_COMMIT.fullmatch(git_commit):
            raise ValidationError("git_commit must be 40 lowercase hexadecimal characters")
        validate_sha256(document["calibration_sha256"], "calibration_sha256")
        validate_sha256(document["safety_sha256"], "safety_sha256")
        issued = _parse_utc(document["issued_at_utc"], "issued_at_utc")
        expires = _parse_utc(document["expires_at_utc"], "expires_at_utc")
        if expires <= issued:
            raise ValidationError("operator permit must expire after it is issued")
        heartbeat_timeout_ns = document["heartbeat_timeout_ns"]
        if (
            not isinstance(heartbeat_timeout_ns, int)
            or isinstance(heartbeat_timeout_ns, bool)
            or heartbeat_timeout_ns <= 0
        ):
            raise ValidationError("heartbeat_timeout_ns must be a positive integer")
        real_enable_authorized = document["real_enable_authorized"]
        if not isinstance(real_enable_authorized, bool):
            raise ValidationError("real_enable_authorized must be boolean")
        if execution_mode == "simulation" and real_enable_authorized:
            raise ValidationError("simulation permits cannot authorize real enable")
        if execution_mode == "real" and not real_enable_authorized:
            raise ValidationError("real permits require explicit real enable authorization")

        return cls(
            session_id=session_id,
            operator_uid=operator_uid,
            execution_mode=execution_mode,
            git_commit=git_commit,
            calibration_sha256=document["calibration_sha256"],
            safety_sha256=document["safety_sha256"],
            issued_at_utc=document["issued_at_utc"],
            expires_at_utc=document["expires_at_utc"],
            heartbeat_timeout_ns=heartbeat_timeout_ns,
            real_enable_authorized=real_enable_authorized,
            permit_sha256=document["permit_sha256"],
        )

    @property
    def issued_at(self) -> datetime:
        return _parse_utc(self.issued_at_utc, "issued_at_utc")

    @property
    def expires_at(self) -> datetime:
        return _parse_utc(self.expires_at_utc, "expires_at_utc")

    def assert_authorized(
        self,
        *,
        operator_uid: int,
        git_commit: str,
        calibration_sha256: str,
        safety_sha256: str,
        hardware_verified: bool,
        now_utc: datetime,
    ) -> None:
        if operator_uid != self.operator_uid:
            raise StateConflict("operator permit does not match the connected peer")
        if git_commit != self.git_commit:
            raise StateConflict("operator permit git commit mismatch")
        if calibration_sha256 != self.calibration_sha256:
            raise StateConflict("operator permit calibration hash mismatch")
        if safety_sha256 != self.safety_sha256:
            raise StateConflict("operator permit safety hash mismatch")
        if now_utc.tzinfo is None:
            raise ValidationError("operator session UTC clock must be timezone-aware")
        current = now_utc.astimezone(timezone.utc)
        if current < self.issued_at:
            raise StateConflict("operator permit is not active yet")
        if current >= self.expires_at:
            raise StateConflict("operator permit has expired")
        if self.execution_mode == "real" and not hardware_verified:
            raise StateConflict("real enable requires hardware_verified=true")
        if self.execution_mode == "simulation" and hardware_verified:
            raise StateConflict("simulation permit cannot authorize a real backend")


def build_operator_permit(
    *,
    session_id: str,
    operator_uid: int,
    execution_mode: str,
    git_commit: str,
    calibration_sha256: str,
    safety_sha256: str,
    issued_at: datetime,
    expires_at: datetime,
    heartbeat_timeout_ns: int,
    real_enable_authorized: bool,
) -> dict[str, Any]:
    body = {
        "schema_version": PERMIT_SCHEMA,
        "session_id": session_id,
        "operator_uid": operator_uid,
        "execution_mode": execution_mode,
        "git_commit": git_commit,
        "calibration_sha256": calibration_sha256,
        "safety_sha256": safety_sha256,
        "issued_at_utc": _format_utc(issued_at),
        "expires_at_utc": _format_utc(expires_at),
        "heartbeat_timeout_ns": heartbeat_timeout_ns,
        "real_enable_authorized": real_enable_authorized,
    }
    sealed = seal_document(body, "permit_sha256")
    OperatorPermit.from_dict(sealed)
    return sealed


def load_operator_permit(path: str | Path, *, require_root_owner: bool = True) -> OperatorPermit:
    permit_path = Path(path)
    flags = os.O_RDONLY
    flags |= getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(permit_path, flags)
    except OSError as exc:
        raise ValidationError(f"cannot open operator permit: {exc}") from exc
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise ValidationError("operator permit must be a regular file")
        if require_root_owner and metadata.st_uid != 0:
            raise ValidationError("operator permit must be owned by root")
        if metadata.st_mode & (stat.S_IWGRP | stat.S_IWOTH):
            raise ValidationError("operator permit must not be group/world writable")
        try:
            with os.fdopen(descriptor, "r", encoding="utf-8") as handle:
                descriptor = -1
                value = json.load(handle)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValidationError(f"cannot read operator permit: {exc}") from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    return OperatorPermit.from_dict(value)


class OperatorSession:
    """Binds one Unix peer and one root-owned permit to a SafeRobot instance."""

    def __init__(
        self,
        robot: SafeRobot,
        clock: Clock,
        *,
        permit_path: str | Path,
        git_commit: str,
        calibration_sha256: str,
        safety_sha256: str,
        require_root_permit: bool = True,
        utc_now: Callable[[], datetime] | None = None,
    ):
        if not _GIT_COMMIT.fullmatch(git_commit):
            raise ValidationError("operator session requires a 40-character git commit")
        validate_sha256(calibration_sha256, "calibration_sha256")
        validate_sha256(safety_sha256, "safety_sha256")
        self.robot = robot
        self.clock = clock
        self.permit_path = Path(permit_path)
        self.git_commit = git_commit
        self.calibration_sha256 = calibration_sha256
        self.safety_sha256 = safety_sha256
        self.require_root_permit = require_root_permit
        self.utc_now = utc_now or (lambda: datetime.now(timezone.utc))
        self._operator_uid: int | None = None
        self._last_heartbeat_ns: int | None = None
        self._active_permit: OperatorPermit | None = None

    def connect_client(self, operator_uid: int) -> None:
        if self._operator_uid is not None:
            raise StateConflict("an operator client is already connected")
        if not isinstance(operator_uid, int) or operator_uid < 0:
            raise ValidationError("connected peer uid must be non-negative")
        self._operator_uid = operator_uid
        self._last_heartbeat_ns = self.clock.now_ns()

    def _assert_peer(self, operator_uid: int) -> None:
        if self._operator_uid is None or operator_uid != self._operator_uid:
            raise StateConflict("request does not match the connected operator")

    def _load_authorized_permit(self, operator_uid: int) -> OperatorPermit:
        self._assert_peer(operator_uid)
        permit = load_operator_permit(self.permit_path, require_root_owner=self.require_root_permit)
        permit.assert_authorized(
            operator_uid=operator_uid,
            git_commit=self.git_commit,
            calibration_sha256=self.calibration_sha256,
            safety_sha256=self.safety_sha256,
            hardware_verified=self.robot.backend.hardware_verified,
            now_utc=self.utc_now(),
        )
        return permit

    def enable(self, operator_uid: int) -> None:
        permit = self._load_authorized_permit(operator_uid)
        self.robot.enable()
        self._active_permit = permit
        self._last_heartbeat_ns = self.clock.now_ns()

    def heartbeat(self, operator_uid: int) -> None:
        permit = self._load_authorized_permit(operator_uid)
        self._active_permit = permit
        self._last_heartbeat_ns = self.clock.now_ns()

    def _latch_safe_stop(self, reason: StopReason) -> None:
        if self.robot.state in {SafetyState.READY, SafetyState.ACTIVE}:
            self.robot.request_safe_stop(reason)

    def check_liveness(self) -> bool:
        if self.robot.state not in {SafetyState.READY, SafetyState.ACTIVE}:
            return True
        if self._operator_uid is None or self._last_heartbeat_ns is None:
            self._latch_safe_stop(StopReason.OPERATOR_DISCONNECT)
            return False
        try:
            permit = self._load_authorized_permit(self._operator_uid)
        except (StateConflict, ValidationError) as exc:
            reason = (
                StopReason.PERMIT_EXPIRED
                if "expired" in str(exc).lower()
                else StopReason.PERMIT_INVALID
            )
            self._latch_safe_stop(reason)
            return False
        now_ns = self.clock.now_ns()
        if now_ns - self._last_heartbeat_ns > permit.heartbeat_timeout_ns:
            self._latch_safe_stop(StopReason.OPERATOR_HEARTBEAT_TIMEOUT)
            return False
        self._active_permit = permit
        return self.robot.check_watchdog()

    def send_action(self, operator_uid: int, action: ActionEnvelope) -> ActionReceipt:
        self._assert_peer(operator_uid)
        if not self.check_liveness():
            raise StateConflict("operator session is no longer live")
        return self.robot.send_action(action)

    def get_observation(self, operator_uid: int) -> Observation:
        self._assert_peer(operator_uid)
        if self.robot.state in {SafetyState.READY, SafetyState.ACTIVE}:
            self.check_liveness()
        return self.robot.get_observation()

    def disable(self, operator_uid: int) -> None:
        self._assert_peer(operator_uid)
        self.robot.disable()
        self._active_permit = None

    def request_safe_stop(self, operator_uid: int) -> None:
        self._assert_peer(operator_uid)
        self._latch_safe_stop(StopReason.OPERATOR_REQUEST)

    def disconnect_client(self, operator_uid: int) -> None:
        self._assert_peer(operator_uid)
        self._latch_safe_stop(StopReason.OPERATOR_DISCONNECT)
        self._operator_uid = None
        self._last_heartbeat_ns = None
        self._active_permit = None

    def status(self) -> dict[str, Any]:
        permit = self._active_permit
        return {
            "state": self.robot.state.value,
            "client_connected": self._operator_uid is not None,
            "session_id": permit.session_id if permit else None,
            "execution_mode": permit.execution_mode if permit else None,
            "hardware_verified": self.robot.backend.hardware_verified,
            "reset_available_to_operator": False,
        }
