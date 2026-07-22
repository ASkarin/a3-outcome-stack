"""Immutable dataset and asset manifests."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from .canonical import (
    atomic_write_json,
    canonical_json_bytes,
    directory_content_sha256,
    directory_entries,
    load_json,
    resolve_relative,
    seal_document,
    sha256_bytes,
    utc_now,
    validate_sha256,
    verify_directory_entries,
    verify_sealed_document,
)
from .errors import IntegrityError, StateConflict, ValidationError

DATA_VERSION_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,63}-v(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)$")
SPLITS = {"train", "development", "validation", "final_test"}


def validate_data_version(version: str) -> None:
    if not isinstance(version, str) or DATA_VERSION_RE.fullmatch(version) is None:
        raise ValidationError("data version must match <dataset_slug>-vMAJOR.MINOR.PATCH")


def build_asset_manifest(
    root: str | Path,
    *,
    asset_id: str,
    kind: str,
    logical_uri: str | None = None,
) -> dict[str, Any]:
    if not asset_id or not kind:
        raise ValidationError("asset_id and kind are required")
    entries = directory_entries(root)
    body = {
        "schema_version": "1.0",
        "asset_id": asset_id,
        "kind": kind,
        "logical_uri": logical_uri or asset_id,
        "created_at_utc": utc_now(),
        "files": entries,
        "content_sha256": directory_content_sha256(entries),
    }
    return seal_document(body, "manifest_sha256")


def verify_asset_manifest(root: str | Path, manifest: dict[str, Any]) -> None:
    verify_sealed_document(manifest, "manifest_sha256")
    validate_sha256(manifest.get("content_sha256"), "content_sha256")
    expected = manifest.get("files")
    if not isinstance(expected, list):
        raise ValidationError("asset manifest files must be a list")
    verify_directory_entries(root, expected)
    actual_content = directory_content_sha256(expected)
    if actual_content != manifest["content_sha256"]:
        raise IntegrityError("asset content_sha256 does not match file entries")


def _validate_episode_groups(episodes: list[dict[str, Any]]) -> None:
    episode_ids: set[str] = set()
    group_splits: dict[tuple[str, str], str] = {}
    for episode in episodes:
        episode_id = episode.get("episode_id")
        session_id = episode.get("session_id")
        split = episode.get("split")
        if not isinstance(episode_id, str) or not episode_id:
            raise ValidationError("every episode requires episode_id")
        if episode_id in episode_ids:
            raise ValidationError(f"duplicate episode_id: {episode_id}")
        episode_ids.add(episode_id)
        if not isinstance(session_id, str) or not session_id:
            raise ValidationError(f"episode {episode_id} requires session_id")
        if split not in SPLITS:
            raise ValidationError(f"episode {episode_id} has invalid split {split}")
        for group_name in ("session_id", "source_trajectory_id"):
            group_value = episode.get(group_name)
            if group_value is None:
                continue
            if not isinstance(group_value, str) or not group_value:
                raise ValidationError(f"episode {episode_id} has invalid {group_name}")
            key = (group_name, group_value)
            previous = group_splits.setdefault(key, split)
            if previous != split:
                raise ValidationError(
                    f"split leakage: {group_name}={group_value} appears in {previous} and {split}"
                )


def build_dataset_manifest(
    root: str | Path,
    *,
    data_version: str,
    episodes: list[dict[str, Any]],
    preregistration_id: str,
    preregistration_sha256: str,
) -> dict[str, Any]:
    validate_data_version(data_version)
    validate_sha256(preregistration_sha256, "preregistration_sha256")
    if not preregistration_id:
        raise ValidationError("preregistration_id is required")
    if not isinstance(episodes, list) or not episodes:
        raise ValidationError("episodes must be a non-empty list")
    _validate_episode_groups(episodes)

    built: list[dict[str, Any]] = []
    for source in episodes:
        relative_path = source.get("relative_path")
        episode_root = resolve_relative(root, relative_path)
        entries = directory_entries(episode_root)
        item = {
            key: value
            for key, value in source.items()
            if key in {"episode_id", "session_id", "source_trajectory_id", "task", "split", "relative_path"}
            and value is not None
        }
        item["files"] = entries
        item["content_sha256"] = directory_content_sha256(entries)
        built.append(item)
    built.sort(key=lambda item: item["episode_id"])

    stable_content = {
        "data_version": data_version,
        "preregistration_id": preregistration_id,
        "preregistration_sha256": preregistration_sha256,
        "episodes": built,
    }
    body = {
        "schema_version": "1.0",
        **stable_content,
        "created_at_utc": utc_now(),
        "content_sha256": sha256_bytes(canonical_json_bytes(stable_content)),
    }
    return seal_document(body, "manifest_sha256")


def verify_dataset_manifest(root: str | Path, manifest: dict[str, Any]) -> None:
    verify_sealed_document(manifest, "manifest_sha256")
    validate_data_version(manifest.get("data_version"))
    validate_sha256(manifest.get("preregistration_sha256"), "preregistration_sha256")
    validate_sha256(manifest.get("content_sha256"), "content_sha256")
    episodes = manifest.get("episodes")
    if not isinstance(episodes, list) or not episodes:
        raise ValidationError("dataset manifest episodes must be non-empty")
    _validate_episode_groups(episodes)

    for episode in episodes:
        episode_root = resolve_relative(root, episode.get("relative_path"))
        verify_directory_entries(episode_root, episode.get("files"))
        actual_episode_hash = directory_content_sha256(episode["files"])
        if actual_episode_hash != episode.get("content_sha256"):
            raise IntegrityError(f"episode hash mismatch: {episode.get('episode_id')}")

    stable_content = {
        "data_version": manifest["data_version"],
        "preregistration_id": manifest["preregistration_id"],
        "preregistration_sha256": manifest["preregistration_sha256"],
        "episodes": episodes,
    }
    actual_content = sha256_bytes(canonical_json_bytes(stable_content))
    if actual_content != manifest["content_sha256"]:
        raise IntegrityError("dataset content_sha256 mismatch")


def write_manifest_immutable(manifest: dict[str, Any], output: str | Path, *, kind: str) -> None:
    if kind not in {"asset", "dataset"}:
        raise ValidationError(f"unknown manifest kind: {kind}")
    verify_sealed_document(manifest, "manifest_sha256")
    destination = Path(output)

    if kind == "dataset":
        version = manifest.get("data_version")
        validate_data_version(version)
        for candidate in destination.parent.glob("*.json"):
            try:
                existing = load_json(candidate)
            except ValidationError:
                continue
            if existing.get("data_version") != version:
                continue
            if existing.get("content_sha256") != manifest.get("content_sha256"):
                raise StateConflict(
                    f"data version {version} is already bound to different content in {candidate}"
                )

    atomic_write_json(destination, manifest, immutable=True)

