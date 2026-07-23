from __future__ import annotations

import pytest

from a3_outcome_stack.ops.errors import IntegrityError, StateConflict, ValidationError
from a3_outcome_stack.ops.manifests import (
    build_asset_manifest,
    build_dataset_manifest,
    verify_asset_manifest,
    verify_dataset_manifest,
    write_manifest_immutable,
)

PREREG_HASH = "sha256:" + "a" * 64


def test_asset_hash_is_stable_and_detects_tampering(tmp_path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    (first / "b.txt").write_text("beta", encoding="utf-8")
    (first / "a.txt").write_text("alpha", encoding="utf-8")
    (second / "a.txt").write_text("alpha", encoding="utf-8")
    (second / "b.txt").write_text("beta", encoding="utf-8")

    manifest_one = build_asset_manifest(first, asset_id="fixture", kind="test")
    manifest_two = build_asset_manifest(second, asset_id="fixture", kind="test")
    assert manifest_one["content_sha256"] == manifest_two["content_sha256"]
    verify_asset_manifest(first, manifest_one)

    (first / "a.txt").write_text("changed", encoding="utf-8")
    with pytest.raises(IntegrityError):
        verify_asset_manifest(first, manifest_one)


def test_asset_manifest_detects_missing_and_extra_files(tmp_path):
    root = tmp_path / "asset"
    root.mkdir()
    (root / "one.bin").write_bytes(b"1")
    manifest = build_asset_manifest(root, asset_id="fixture", kind="test")
    (root / "two.bin").write_bytes(b"2")
    with pytest.raises(IntegrityError):
        verify_asset_manifest(root, manifest)
    (root / "two.bin").unlink()
    (root / "one.bin").unlink()
    with pytest.raises(IntegrityError):
        verify_asset_manifest(root, manifest)


def _episode(relative_path, episode_id="ep-1", session_id="session-1", split="train", source="source-1"):
    return {
        "episode_id": episode_id,
        "session_id": session_id,
        "source_trajectory_id": source,
        "task": "A",
        "split": split,
        "relative_path": relative_path,
    }


def test_dataset_manifest_and_split_leakage(tmp_path):
    root = tmp_path / "dataset"
    (root / "ep-1").mkdir(parents=True)
    (root / "ep-2").mkdir()
    (root / "ep-1" / "data.bin").write_bytes(b"one")
    (root / "ep-2" / "data.bin").write_bytes(b"two")

    manifest = build_dataset_manifest(
        root,
        data_version="a3-core-v0.1.0",
        episodes=[_episode("ep-1")],
        preregistration_id="PR-20260722-01",
        preregistration_sha256=PREREG_HASH,
    )
    verify_dataset_manifest(root, manifest)
    (root / "ep-1" / "data.bin").write_bytes(b"tampered")
    with pytest.raises(IntegrityError):
        verify_dataset_manifest(root, manifest)

    leaking = [
        _episode("ep-1", "ep-1", "same-session", "train", "source-1"),
        _episode("ep-2", "ep-2", "same-session", "validation", "source-2"),
    ]
    with pytest.raises(ValidationError, match="split leakage"):
        build_dataset_manifest(
            root,
            data_version="a3-core-v0.1.0",
            episodes=leaking,
            preregistration_id="PR-20260722-01",
            preregistration_sha256=PREREG_HASH,
        )


def test_data_version_cannot_be_rebound(tmp_path):
    manifests = tmp_path / "manifests"
    root_one = tmp_path / "one"
    root_two = tmp_path / "two"
    (root_one / "ep").mkdir(parents=True)
    (root_two / "ep").mkdir(parents=True)
    (root_one / "ep" / "data.bin").write_bytes(b"one")
    (root_two / "ep" / "data.bin").write_bytes(b"two")
    kwargs = {
        "data_version": "a3-core-v0.1.0",
        "episodes": [_episode("ep")],
        "preregistration_id": "PR-20260722-01",
        "preregistration_sha256": PREREG_HASH,
    }
    first = build_dataset_manifest(root_one, **kwargs)
    second = build_dataset_manifest(root_two, **kwargs)
    write_manifest_immutable(first, manifests / "first.json", kind="dataset")
    with pytest.raises(StateConflict, match="already bound"):
        write_manifest_immutable(second, manifests / "second.json", kind="dataset")

