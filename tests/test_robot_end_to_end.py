from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def run_cli(root: Path, *args: str) -> subprocess.CompletedProcess:
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(root / "src")
    return subprocess.run(
        [sys.executable, "-m", "embodied_ai.ops", *args],
        cwd=root,
        env=environment,
        text=True,
        capture_output=True,
    )


def test_cli_mock_write_verify_and_strict_replay(tmp_path):
    root = Path(__file__).parents[1]
    trace = tmp_path / "cli-trace"
    record = run_cli(
        root,
        "robot",
        "mock",
        "record",
        "--config",
        str(root / "configs/robot/a3_mock_test.json"),
        "--output",
        str(trace),
        "--steps",
        "3",
    )
    assert record.returncode == 0, record.stderr
    verify = run_cli(root, "robot", "trace", "verify", "--trace", str(trace))
    assert verify.returncode == 0, verify.stderr
    replay = run_cli(root, "robot", "replay", "--trace", str(trace), "--strict")
    assert replay.returncode == 0, replay.stderr

    records = trace / "records.jsonl"
    records.write_bytes(records.read_bytes() + b"\n")
    tampered = run_cli(root, "robot", "trace", "verify", "--trace", str(trace))
    assert tampered.returncode == 3
