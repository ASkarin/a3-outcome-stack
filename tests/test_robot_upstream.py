from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from a3_outcome_stack.ops.canonical import atomic_write_json, seal_document, sha256_file
from a3_outcome_stack.ops.errors import IntegrityError
from a3_outcome_stack.robot.upstream import verify_upstream_lock

ROOT = Path(__file__).parents[1]


def test_real_upstream_lock_is_self_consistent():
    result = verify_upstream_lock(ROOT / "configs/upstream/edulite_a3.lock.json")
    assert result["commit"] == "ea7231f784ebb37e4c4120f7be8e3670514dc9ee"
    assert result["selected_files"] == 7
    assert result["source_files_verified"] is False


def test_upstream_file_tamper_is_detected(tmp_path):
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    source = checkout / "interface.py"
    source.write_text("pinned", encoding="utf-8")
    lock = seal_document(
        {
            "schema_version": "upstream-source-lock-v1",
            "repository": "https://example.invalid/upstream",
            "commit": "1" * 40,
            "files": [{"path": "interface.py", "sha256": sha256_file(source)}],
        },
        "lock_sha256",
    )
    lock_path = tmp_path / "lock.json"
    atomic_write_json(lock_path, lock)
    assert verify_upstream_lock(lock_path, checkout)["source_files_verified"] is True
    source.write_text("tampered", encoding="utf-8")
    with pytest.raises(IntegrityError, match="hash mismatch"):
        verify_upstream_lock(lock_path, checkout)


def test_upstream_git_commit_mismatch_is_detected(tmp_path):
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    subprocess.run(["git", "init", "-q", str(checkout)], check=True)
    subprocess.run(
        ["git", "-C", str(checkout), "config", "user.name", "Fixture"], check=True
    )
    subprocess.run(
        ["git", "-C", str(checkout), "config", "user.email", "fixture@example.invalid"],
        check=True,
    )
    source = checkout / "interface.py"
    source.write_text("pinned", encoding="utf-8")
    subprocess.run(["git", "-C", str(checkout), "add", "interface.py"], check=True)
    subprocess.run(
        ["git", "-C", str(checkout), "commit", "-q", "-m", "fixture"], check=True
    )
    actual = subprocess.check_output(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True
    ).strip()
    different = "0" * 40 if actual != "0" * 40 else "1" * 40
    lock = seal_document(
        {
            "schema_version": "upstream-source-lock-v1",
            "repository": "https://example.invalid/upstream",
            "commit": different,
            "files": [{"path": "interface.py", "sha256": sha256_file(source)}],
        },
        "lock_sha256",
    )
    lock_path = tmp_path / "lock.json"
    atomic_write_json(lock_path, lock)
    with pytest.raises(IntegrityError, match="commit mismatch"):
        verify_upstream_lock(lock_path, checkout)
