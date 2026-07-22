"""Read-only structural and dependency verification for the project root."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from .canonical import load_json, sha256_file
from .errors import IntegrityError, ValidationError
from .experiments import verify_registry
from .results import verify_results_index
from embodied_ai.robot.types import verify_contract_file
from embodied_ai.robot.upstream import verify_upstream_lock

REQUIRED_DIRECTORIES = [
    "src/embodied_ai/ops",
    "src/embodied_ai/robot",
    "configs",
    "configs/robot",
    "configs/upstream",
    "metadata/datasets",
    "metadata/assets",
    "metadata/freezes",
    "experiments/specs",
    "runs",
    "results/summaries",
    "docs/preregistration",
    "tests",
    "evidence",
]


def doctor_project(root: str | Path) -> dict[str, Any]:
    root_path = Path(root).resolve()
    missing = [
        relative
        for relative in REQUIRED_DIRECTORIES
        if not (root_path / relative).is_dir()
    ]
    if missing:
        raise ValidationError(f"missing project directories: {missing}")

    project = load_json(root_path / "configs/project.json")
    prereg = project.get("preregistration", {})
    snapshot = root_path / prereg.get("path", "")
    if not snapshot.is_file():
        raise ValidationError(f"missing preregistration snapshot: {snapshot}")
    actual_prereg_hash = sha256_file(snapshot)
    if actual_prereg_hash != prereg.get("sha256"):
        raise IntegrityError(
            f"preregistration snapshot mismatch: expected {prereg.get('sha256')}, got {actual_prereg_hash}"
        )

    registry_path = root_path / project.get("experiment_registry", "")
    verify_registry(
        root_path / "experiments/specs", registry_path, root_path / "results/summaries"
    )
    result_index = root_path / project.get("result_index", "")
    verify_results_index(root_path / "results/summaries", result_index)
    for line_number, line in enumerate(
        result_index.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if line.strip():
            try:
                json.loads(line)
            except json.JSONDecodeError as exc:
                raise IntegrityError(
                    f"invalid result index JSON at line {line_number}"
                ) from exc

    contract = verify_contract_file(root_path / "configs/robot/a3_contract_v1.json")
    upstream = verify_upstream_lock(root_path / "configs/upstream/edulite_a3.lock.json")

    try:
        branch = subprocess.check_output(
            ["git", "-C", str(root_path), "branch", "--show-current"], text=True
        ).strip()
        commit = subprocess.check_output(
            ["git", "-C", str(root_path), "rev-parse", "HEAD"], text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ValidationError(
            "project root must be a committed Git repository"
        ) from exc
    if branch != "main":
        raise IntegrityError(f"expected Git branch main, got {branch}")

    try:
        import torch
    except ImportError as exc:
        raise ValidationError(
            "PyTorch is not importable in the active environment"
        ) from exc

    return {
        "status": "ok",
        "project_root": str(root_path),
        "git_branch": branch,
        "git_commit": commit,
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "preregistration_id": prereg.get("id"),
        "preregistration_sha256": actual_prereg_hash,
        "robot_contract_schema": contract["schema_version"],
        "a3_upstream_commit": upstream["commit"],
        "hardware_verified": False,
    }
