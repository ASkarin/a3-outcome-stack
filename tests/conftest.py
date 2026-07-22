from __future__ import annotations

import copy

import pytest

HASH_A = "sha256:" + "a" * 64
HASH_B = "sha256:" + "b" * 64
HASH_C = "sha256:" + "c" * 64


@pytest.fixture
def experiment_source():
    return {
        "stage": "2",
        "preregistration_id": "PR-20260722-01",
        "preregistration_sha256": HASH_A,
        "hypothesis": "ACT forms a repeatable Task A baseline",
        "task": "A",
        "policy": "ACT",
        "data_version": "a3-core-v0.1.0",
        "data_manifest_sha256": HASH_B,
        "git_commit": "1" * 40,
        "config_path": "configs/act/task_a.json",
        "config_sha256": HASH_C,
        "seed": 7,
        "hardware": "laboratory:3xRTX3090",
        "max_steps": 100,
        "max_wallclock_seconds": 3600,
        "eval_interval_steps": 10,
        "patience_evaluations": 5,
        "stop_metric": "development_success_rate",
    }


@pytest.fixture
def freeze_source():
    return {
        "freeze_id": "FRZ-20261231-CORE",
        "created_at_utc": "2026-12-31T00:00:00Z",
        "preregistration": {"id": "PR-20260722-01", "sha256": HASH_A},
        "split_protocol": {"version": "0.2", "sha256": HASH_B},
        "code": {"git_commit": "1" * 40},
        "environment": {"lock_sha256": HASH_C},
        "configs": [{"path": "configs/act.json", "sha256": HASH_A}],
        "seeds": [1, 2, 3],
        "data": {"version": "a3-core-v1.0.0", "manifest_sha256": HASH_B},
        "checkpoints": [{"experiment_id": "EXP-A-ACT-AAAAAAAAAAAA", "sha256": HASH_C}],
        "normalization": {"sha256": HASH_A},
        "calibration_assets": [{"id": "calibration-v1", "sha256": HASH_B}],
        "camera_assets": [{"id": "camera-layout-v1", "sha256": HASH_C}],
        "action_chunk": {"act": 20, "smolvla": 20},
        "thresholds": {"failure": 0.7},
        "evaluation_conditions": ["id", "new_position", "new_object"],
        "episode_budget": {"per_cell": 20},
        "run_order": ["id", "new_position", "new_object"],
        "randomization_seed": 20261231,
        "metrics": ["success_rate", "harmful_failure_rate"],
        "ci_method": "wilson_and_stratified_bootstrap",
        "failure_taxonomy": ["miss", "drop", "wrong_target"],
        "exclusion_rules": ["external_power_failure"],
        "final_test_manifest_sha256": HASH_C,
    }

