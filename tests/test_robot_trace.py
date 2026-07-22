from __future__ import annotations

from pathlib import Path

import pytest

from embodied_ai.ops.canonical import load_json
from embodied_ai.ops.errors import IntegrityError, ValidationError
from embodied_ai.robot.backend import SafeRobot
from embodied_ai.robot.clock import ManualClock
from embodied_ai.robot.mock import MockBackend
from embodied_ai.robot.replay import ReplayBackend
from embodied_ai.robot.trace import TraceWriter, verify_trace
from embodied_ai.robot.types import ActionEnvelope, action_features

ROOT = Path(__file__).parents[1]


def build_trace(destination: Path, *, add_blob=False):
    config = load_json(ROOT / "configs/robot/a3_mock_test.json")
    clock = ManualClock(config["start_monotonic_ns"], config["clock_domain_id"])
    backend = MockBackend(clock, config)
    robot = SafeRobot(
        backend,
        clock,
        watchdog_timeout_ns=config["watchdog_timeout_ns"],
        joint_lower=config["joint_limits"]["lower"],
        joint_upper=config["joint_limits"]["upper"],
    )
    writer = TraceWriter(
        destination,
        {
            "trace_id": destination.name,
            "clock_domain_id": clock.domain_id,
            "hardware_verified": False,
            "observation_features": dict(robot.observation_features),
            "action_features": action_features(),
        },
    )
    blob = writer.add_blob("mock/frame.bin", b"fixture") if add_blob else None
    robot.connect()
    robot.enable()
    for index in range(2):
        now = clock.now_ns()
        candidate = ActionEnvelope(
            (0.1 * (index + 1),) * 7, index, now, now + 100, clock.domain_id
        )
        receipt = robot.send_action(candidate)
        clock.advance_ns(10)
        observation = robot.get_observation()
        camera_refs = {"mock_rgb": blob} if blob is not None else {}
        observation = type(observation)(
            observation.joint_position,
            observation.joint_velocity,
            observation.joint_effort,
            observation.tcp_position,
            observation.tcp_orientation_xyzw,
            observation.status,
            observation.timestamps,
            camera_refs,
        )
        writer.append(
            {
                "record_index": index,
                "observation": observation.to_dict(),
                "action": candidate.to_dict(),
                "receipt": receipt.to_dict(),
                "next_clock_ns": clock.now_ns(),
                "safety_transitions": robot.supervisor.transitions,
            }
        )
    complete = writer.finalize()
    return complete


def test_trace_roundtrip_and_strict_replay(tmp_path):
    trace_path = tmp_path / "trace"
    complete = build_trace(trace_path, add_blob=True)
    verified = verify_trace(trace_path)
    assert len(verified.records) == 2
    assert verified.complete["trace_sha256"] == complete["trace_sha256"]
    backend = ReplayBackend(verified, strict=True)
    backend.connect()
    backend.enable()
    for record in verified.records:
        assert backend.get_observation().to_dict() == record["observation"]
        backend.send_action(ActionEnvelope.from_dict(record["action"]))
    assert backend.consumed_records == 2


def test_strict_replay_rejects_mismatched_action(tmp_path):
    trace_path = tmp_path / "trace"
    build_trace(trace_path)
    backend = ReplayBackend(verify_trace(trace_path), strict=True)
    backend.connect()
    backend.enable()
    record = backend.trace.records[0]
    backend.get_observation()
    expected = ActionEnvelope.from_dict(record["action"])
    wrong = ActionEnvelope(
        (0.9,) * 7,
        expected.sequence_id,
        expected.created_monotonic_ns,
        expected.deadline_monotonic_ns,
        expected.clock_domain_id,
    )
    with pytest.raises(IntegrityError, match="action mismatch"):
        backend.send_action(wrong)
    assert not backend.is_healthy


def test_trace_tamper_and_missing_blob_are_detected(tmp_path):
    trace_path = tmp_path / "trace"
    build_trace(trace_path, add_blob=True)
    records = trace_path / "records.jsonl"
    records.write_bytes(records.read_bytes().replace(b"0.1", b"0.2", 1))
    with pytest.raises(IntegrityError, match="records.jsonl hash mismatch"):
        verify_trace(trace_path)

    second = tmp_path / "trace-2"
    build_trace(second, add_blob=True)
    (second / "blobs/mock/frame.bin").unlink()
    with pytest.raises(IntegrityError, match="blob set mismatch"):
        verify_trace(second)


def test_incomplete_trace_and_duplicate_index_are_rejected(tmp_path):
    config = load_json(ROOT / "configs/robot/a3_mock_test.json")
    writer = TraceWriter(
        tmp_path / "incomplete",
        {
            "clock_domain_id": config["clock_domain_id"],
            "observation_features": MockBackend(
                ManualClock(domain_id=config["clock_domain_id"]), config
            ).observation_features,
            "action_features": action_features(),
        },
    )
    writer.close_incomplete()
    with pytest.raises(ValidationError, match="finalized"):
        verify_trace(tmp_path / "incomplete.partial")

    golden = verify_trace(ROOT / "tests/fixtures/traces/a3_mock_v1")
    duplicate_writer = TraceWriter(
        tmp_path / "duplicate",
        {
            "clock_domain_id": golden.meta["clock_domain_id"],
            "observation_features": golden.meta["observation_features"],
            "action_features": golden.meta["action_features"],
        },
    )
    duplicate_writer.append(golden.records[0])
    with pytest.raises(IntegrityError, match="index mismatch"):
        duplicate_writer.append(golden.records[0])
    duplicate_writer.close_incomplete()


def test_unsealed_camera_reference_is_rejected(tmp_path):
    trace_path = tmp_path / "trace"
    build_trace(trace_path)
    records_path = trace_path / "records.jsonl"
    records = records_path.read_text(encoding="utf-8").splitlines()
    import json
    from embodied_ai.ops.canonical import (
        atomic_write_json,
        canonical_json_bytes,
        seal_document,
        sha256_file,
    )

    first = json.loads(records[0])
    first["observation"]["camera_refs"] = {
        "missing": {
            "path": "blobs/missing.bin",
            "size_bytes": 1,
            "sha256": "sha256:" + "0" * 64,
        }
    }
    records[0] = canonical_json_bytes(first).decode("utf-8")
    records_path.write_text("\n".join(records) + "\n", encoding="utf-8")
    complete_path = trace_path / "complete.json"
    old_complete = load_json(complete_path)
    body = {key: value for key, value in old_complete.items() if key != "trace_sha256"}
    body["records_sha256"] = sha256_file(records_path)
    atomic_write_json(complete_path, seal_document(body, "trace_sha256"))
    with pytest.raises(IntegrityError, match="no sealed blob"):
        verify_trace(trace_path)


def test_committed_golden_trace_is_valid():
    trace = verify_trace(ROOT / "tests/fixtures/traces/a3_mock_v1")
    assert trace.complete["record_count"] == 3
    assert trace.meta["hardware_verified"] is False
