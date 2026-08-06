"""Parser and command handlers for stage-1A robot operations."""

from __future__ import annotations

import argparse
import importlib.util
import math
import os
import shutil
import stat
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from a3_outcome_stack.ops.canonical import (
    atomic_write_json,
    canonical_json_bytes,
    load_json,
    sha256_file,
    utc_now,
)
from a3_outcome_stack.ops.errors import IntegrityError, StateConflict, ValidationError

from .backend import SafeRobot
from .clock import ManualClock
from .control import run_control_client, serve_mock_control
from .mock import MockBackend
from .replay import ReplayBackend
from .session import build_operator_permit
from .trace import TraceWriter, verify_trace
from .types import ActionEnvelope, Observation, action_features, verify_contract_file
from .upstream import verify_upstream_lock


def add_robot_parser(commands: argparse._SubParsersAction) -> None:
    robot = commands.add_parser("robot")
    actions = robot.add_subparsers(dest="robot_action", required=True)

    doctor = actions.add_parser("doctor")
    doctor.add_argument("--root", default=".")
    doctor.add_argument("--upstream-checkout")

    contract = actions.add_parser("contract")
    contract_actions = contract.add_subparsers(dest="contract_action", required=True)
    contract_verify = contract_actions.add_parser("verify")
    contract_verify.add_argument(
        "--contract", default="configs/robot/a3_contract_v1.json"
    )

    mock = actions.add_parser("mock")
    mock_actions = mock.add_subparsers(dest="mock_action", required=True)
    mock_record = mock_actions.add_parser("record")
    mock_record.add_argument("--config", default="configs/robot/a3_mock_test.json")
    mock_record.add_argument("--output", required=True)
    mock_record.add_argument("--steps", type=int, default=3)

    trace = actions.add_parser("trace")
    trace_actions = trace.add_subparsers(dest="trace_action", required=True)
    trace_verify = trace_actions.add_parser("verify")
    trace_verify.add_argument("--trace", required=True)

    replay = actions.add_parser("replay")
    replay.add_argument("--trace", required=True)
    replay.add_argument("--strict", action="store_true")

    upstream = actions.add_parser("upstream")
    upstream_actions = upstream.add_subparsers(dest="upstream_action", required=True)
    upstream_verify = upstream_actions.add_parser("verify")
    upstream_verify.add_argument(
        "--lock", default="configs/upstream/edulite_a3.lock.json"
    )
    upstream_verify.add_argument("--checkout")

    control = actions.add_parser("control")
    control_actions = control.add_subparsers(dest="control_action", required=True)
    control_serve = control_actions.add_parser("serve-mock")
    control_serve.add_argument("--config", default="configs/robot/a3_mock_test.json")
    control_serve.add_argument("--permit", required=True)
    control_serve.add_argument(
        "--socket", default="/run/a3-outcome-stack/control.sock"
    )
    control_serve.add_argument("--git-commit", required=True)
    control_serve.add_argument(
        "--calibration", default="configs/robot/a3_calibration.template.json"
    )
    control_serve.add_argument(
        "--safety", default="configs/robot/a3_safety.template.json"
    )
    control_serve.add_argument("--socket-group")
    control_client = control_actions.add_parser("client")
    control_client.add_argument(
        "--socket", default="/run/a3-outcome-stack/control.sock"
    )
    permit_create = control_actions.add_parser("permit-create")
    permit_create.add_argument("--output", required=True)
    permit_create.add_argument("--session-id", required=True)
    permit_create.add_argument("--operator-uid", required=True, type=int)
    permit_create.add_argument(
        "--execution-mode", required=True, choices=("simulation", "real")
    )
    permit_create.add_argument("--git-commit", required=True)
    permit_create.add_argument("--calibration", required=True)
    permit_create.add_argument("--safety", required=True)
    permit_create.add_argument("--duration-seconds", required=True, type=int)
    permit_create.add_argument("--heartbeat-timeout-ns", required=True, type=int)
    permit_create.add_argument("--authorize-real-enable", action="store_true")
    permit_create.add_argument("--reader-group")


def _mock_robot(config: dict[str, Any]) -> tuple[ManualClock, MockBackend, SafeRobot]:
    clock = ManualClock(
        current_ns=config.get("start_monotonic_ns", 0),
        domain_id=config.get("clock_domain_id", "stage1a-mock-clock-v1"),
    )
    backend = MockBackend(clock, config)
    limits = config.get("joint_limits", {})
    robot = SafeRobot(
        backend,
        clock,
        watchdog_timeout_ns=config.get("watchdog_timeout_ns"),
        joint_lower=limits.get("lower", []),
        joint_upper=limits.get("upper", []),
    )
    return clock, backend, robot


def _record_mock(config_path: str, output: str, steps: int) -> dict[str, Any]:
    if steps <= 0:
        raise ValidationError("mock record steps must be positive")
    config = load_json(config_path)
    clock, backend, robot = _mock_robot(config)
    writer = TraceWriter(
        output,
        {
            "trace_id": Path(output).name,
            "purpose": "stage1a_deterministic_mock_fixture",
            "created_at_utc": utc_now(),
            "clock_domain_id": clock.domain_id,
            "hardware_verified": False,
            "observation_features": dict(robot.observation_features),
            "action_features": action_features(),
        },
    )
    try:
        robot.connect()
        robot.enable()
        interval = config.get("test_step_interval_ns")
        step_rad = config.get("test_action_step_rad")
        if (
            not isinstance(interval, int)
            or interval <= 0
            or not isinstance(step_rad, (int, float))
        ):
            raise ValidationError(
                "mock test interval and action step must be configured"
            )
        for index in range(steps):
            created = clock.now_ns()
            action = ActionEnvelope(
                values=tuple(float(step_rad) * (index + 1) for _ in range(7)),
                sequence_id=index,
                created_monotonic_ns=created,
                deadline_monotonic_ns=created + config["watchdog_timeout_ns"],
                clock_domain_id=clock.domain_id,
            )
            receipt = robot.send_action(action)
            clock.advance_ns(interval)
            observation = robot.get_observation()
            camera_refs = {}
            for camera_name, feature in sorted(
                config.get("camera_features", {}).items()
            ):
                shape = feature.get("shape", [])
                blob = bytes([index % 256]) * math.prod(shape)
                camera_refs[camera_name] = writer.add_blob(
                    f"{camera_name}/frame-{index:06d}.rgb8",
                    blob,
                )
            if camera_refs:
                observation = Observation(
                    joint_position=observation.joint_position,
                    joint_velocity=observation.joint_velocity,
                    joint_effort=observation.joint_effort,
                    tcp_position=observation.tcp_position,
                    tcp_orientation_xyzw=observation.tcp_orientation_xyzw,
                    status=observation.status,
                    timestamps=observation.timestamps,
                    camera_refs=camera_refs,
                )
            writer.append(
                {
                    "record_index": index,
                    "observation": observation.to_dict(),
                    "action": action.to_dict(),
                    "receipt": receipt.to_dict(),
                    "next_clock_ns": clock.now_ns(),
                    "safety_transitions": robot.supervisor.transitions,
                }
            )
        complete = writer.finalize()
        robot.disable()
        robot.disconnect()
    except Exception:
        writer.close_incomplete()
        raise
    return {
        "status": "ok",
        "trace": str(Path(output).resolve()),
        "records": steps,
        "trace_sha256": complete["trace_sha256"],
        "hardware_verified": backend.hardware_verified,
    }


def _strict_replay(path: str, strict: bool) -> dict[str, Any]:
    verified = verify_trace(path)
    backend = ReplayBackend(verified, strict=strict)
    backend.connect()
    backend.enable()
    for index, record in enumerate(verified.records):
        observation = backend.get_observation()
        expected_observation = Observation.from_dict(record["observation"])
        if canonical_json_bytes(observation.to_dict()) != canonical_json_bytes(
            expected_observation.to_dict()
        ):
            raise IntegrityError(f"replay observation mismatch at record {index}")
        action = ActionEnvelope.from_dict(record["action"])
        receipt = backend.send_action(action)
        if canonical_json_bytes(receipt.to_dict()) != canonical_json_bytes(
            record["receipt"]
        ):
            raise IntegrityError(f"replay receipt mismatch at record {index}")
    backend.disable()
    backend.disconnect()
    return {
        "status": "ok",
        "trace": str(verified.path),
        "records_replayed": backend.consumed_records,
        "strict": strict,
        "trace_sha256": verified.complete["trace_sha256"],
        "hardware_verified": False,
    }


def _create_operator_permit(args: argparse.Namespace) -> dict[str, Any]:
    if os.name != "posix" or os.geteuid() != 0:
        raise StateConflict("operator permits must be created by the local administrator")
    if args.duration_seconds <= 0:
        raise ValidationError("duration-seconds must be positive")
    group_id: int | None = None
    if args.reader_group:
        import grp

        try:
            group_id = grp.getgrnam(args.reader_group).gr_gid
        except KeyError as exc:
            raise ValidationError(
                f"operator permit reader group does not exist: {args.reader_group}"
            ) from exc
    now = datetime.now(timezone.utc).replace(microsecond=0)
    document = build_operator_permit(
        session_id=args.session_id,
        operator_uid=args.operator_uid,
        execution_mode=args.execution_mode,
        git_commit=args.git_commit,
        calibration_sha256=sha256_file(args.calibration),
        safety_sha256=sha256_file(args.safety),
        issued_at=now,
        expires_at=now + timedelta(seconds=args.duration_seconds),
        heartbeat_timeout_ns=args.heartbeat_timeout_ns,
        real_enable_authorized=args.authorize_real_enable,
    )
    output = Path(args.output)
    atomic_write_json(output, document, immutable=True)
    os.chmod(output, 0o640)
    if group_id is not None:
        os.chown(output, 0, group_id)
    return {
        "status": "ok",
        "session_id": document["session_id"],
        "permit_sha256": document["permit_sha256"],
        "expires_at_utc": document["expires_at_utc"],
        "hardware_verified": False,
    }


def _robot_doctor(root: str, upstream_checkout: str | None) -> dict[str, Any]:
    root_path = Path(root).resolve()
    contract = verify_contract_file(root_path / "configs/robot/a3_contract_v1.json")
    upstream = verify_upstream_lock(
        root_path / "configs/upstream/edulite_a3.lock.json",
        upstream_checkout,
    )
    can_interfaces = []
    network_root = Path("/sys/class/net")
    if network_root.is_dir():
        can_interfaces = sorted(
            path.name for path in network_root.iterdir() if path.name.startswith("can")
        )
    input_devices = []
    input_root = Path("/dev/input")
    if input_root.is_dir():
        input_devices = sorted(path.name for path in input_root.iterdir())
    serial_devices = sorted(
        path.name
        for pattern in ("ttyACM*", "ttyUSB*")
        for path in Path("/dev").glob(pattern)
    )

    group_names: set[str] = set()
    role_groups = {
        "collaborator": os.environ.get("A3_LOCAL_COLLAB_GROUP", "a3-collab"),
        "operator": os.environ.get("A3_LOCAL_OPERATOR_GROUP", "a3-operator"),
        "hardware_service": os.environ.get(
            "A3_LOCAL_HARDWARE_GROUP", "a3-hardware"
        ),
    }
    group_presence = {role: False for role in role_groups}
    if os.name == "posix":
        import grp

        for group_id in os.getgroups():
            try:
                group_names.add(grp.getgrgid(group_id).gr_name)
            except KeyError:
                continue
        for role, group_name in role_groups.items():
            try:
                grp.getgrnam(group_name)
            except KeyError:
                continue
            group_presence[role] = True

    if (hasattr(os, "geteuid") and os.geteuid() == 0) or "sudo" in group_names:
        execution_role = "administrator"
    elif role_groups["operator"] in group_names:
        execution_role = "controlled_operator"
    elif role_groups["collaborator"] in group_names:
        execution_role = "collaborator"
    else:
        execution_role = "unassigned"

    deployment_root = Path(
        os.environ.get("A3_LOCAL_DEPLOYMENT_ROOT", "/opt/a3-outcome-stack")
    )
    session_socket = Path(
        os.environ.get(
            "A3_LOCAL_CONTROL_SOCKET", "/run/a3-outcome-stack/control.sock"
        )
    )
    session_active = False
    try:
        session_active = stat.S_ISSOCK(session_socket.lstat().st_mode)
    except OSError:
        pass

    gpu_inventory = []
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
        for line in completed.stdout.splitlines():
            fields = [field.strip() for field in line.split(",")]
            if len(fields) == 3:
                gpu_inventory.append(
                    {
                        "name": fields[0],
                        "driver_version": fields[1],
                        "memory_mib": int(fields[2]),
                    }
                )
    except (OSError, ValueError, subprocess.SubprocessError):
        pass
    return {
        "status": "ok",
        "contract_schema": contract["schema_version"],
        "upstream": upstream,
        "dependencies": {
            "ros2_cli": shutil.which("ros2") is not None,
            "lerobot": importlib.util.find_spec("lerobot") is not None,
            "pinocchio": importlib.util.find_spec("pinocchio") is not None,
            "el_a3_sdk": importlib.util.find_spec("el_a3_sdk") is not None,
        },
        "can_interfaces": can_interfaces,
        "serial_devices": serial_devices,
        "input_devices": input_devices,
        "gpu_inventory": gpu_inventory,
        "execution_role": execution_role,
        "role_boundary": {
            "groups_configured": group_presence,
            "raw_hardware_access_granted_to_humans": False,
            "operator_reset_allowed": False,
        },
        "deployment": {
            "exists": deployment_root.is_dir(),
            "writable_by_current_process": deployment_root.is_dir()
            and os.access(deployment_root, os.W_OK),
        },
        "operator_session": {
            "unix_socket_active": session_active,
            "network_listener_configured": False,
        },
        "hardware_available": False,
        "hardware_tests_executed": False,
        "motor_enable_executed": False,
        "real_can_traffic_executed": False,
        "hardware_verified": False,
        "hardware_branch": "not_executed",
    }


def run_robot_command(args: argparse.Namespace) -> Any:
    if args.robot_action == "doctor":
        return _robot_doctor(args.root, args.upstream_checkout)
    if args.robot_action == "contract" and args.contract_action == "verify":
        contract = verify_contract_file(args.contract)
        return {
            "status": "ok",
            "contract": args.contract,
            "schema_version": contract["schema_version"],
        }
    if args.robot_action == "mock" and args.mock_action == "record":
        return _record_mock(args.config, args.output, args.steps)
    if args.robot_action == "trace" and args.trace_action == "verify":
        trace = verify_trace(args.trace)
        return {
            "status": "ok",
            "trace": str(trace.path),
            "records": len(trace.records),
            "trace_sha256": trace.complete["trace_sha256"],
        }
    if args.robot_action == "replay":
        return _strict_replay(args.trace, args.strict)
    if args.robot_action == "upstream" and args.upstream_action == "verify":
        return verify_upstream_lock(args.lock, args.checkout)
    if args.robot_action == "control" and args.control_action == "serve-mock":
        return serve_mock_control(
            config_path=args.config,
            permit_path=args.permit,
            socket_path=args.socket,
            git_commit=args.git_commit,
            calibration_path=args.calibration,
            safety_path=args.safety,
            socket_group=args.socket_group,
        )
    if args.robot_action == "control" and args.control_action == "client":
        return run_control_client(args.socket)
    if args.robot_action == "control" and args.control_action == "permit-create":
        return _create_operator_permit(args)
    raise ValidationError("unsupported robot command")
