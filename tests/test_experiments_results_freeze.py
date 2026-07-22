from __future__ import annotations

import csv
import json

import pytest

from embodied_ai.ops.errors import IntegrityError, StateConflict, ValidationError
from embodied_ai.ops.experiments import REGISTRY_FIELDS, materialize_experiment_spec, register_experiment
from embodied_ai.ops.freeze import create_freeze, verify_freeze_file
from embodied_ai.ops.results import finalize_result, rebuild_results_index, verify_results_index
from embodied_ai.ops.runstate import create_run_state, transition_run_state


def test_experiment_id_is_stable_and_identity_sensitive(experiment_source):
    first = materialize_experiment_spec(experiment_source)
    second = materialize_experiment_spec(experiment_source)
    assert first["experiment_id"] == second["experiment_id"]
    changed = dict(experiment_source)
    changed["seed"] = experiment_source["seed"] + 1
    assert materialize_experiment_spec(changed)["experiment_id"] != first["experiment_id"]


def test_registration_generates_registry(experiment_source, tmp_path):
    specs = tmp_path / "experiments" / "specs"
    registry = tmp_path / "experiments" / "registry.csv"
    registered = register_experiment(experiment_source, specs_dir=specs, registry_path=registry)
    same = register_experiment(experiment_source, specs_dir=specs, registry_path=registry)
    assert same["experiment_id"] == registered["experiment_id"]
    with registry.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert list(rows[0]) == REGISTRY_FIELDS
    assert len(rows) == 1
    assert rows[0]["experiment_id"] == registered["experiment_id"]


def test_run_state_rejects_terminal_reopen(tmp_path):
    path = tmp_path / "state.json"
    create_run_state(path, "EXP-A-ACT-AAAAAAAAAAAA", "EXP-A-ACT-AAAAAAAAAAAA-A001")
    transition_run_state(path, "running")
    transition_run_state(path, "completed")
    with pytest.raises(StateConflict, match="illegal run transition"):
        transition_run_state(path, "running")


def _summary(status="completed"):
    result = {
        "experiment_id": "EXP-A-ACT-AAAAAAAAAAAA",
        "attempt_id": "EXP-A-ACT-AAAAAAAAAAAA-A001",
        "status": status,
        "started_at_utc": "2026-10-01T00:00:00Z",
        "ended_at_utc": "2026-10-01T01:00:00Z",
        "git_commit": "1" * 40,
        "config_sha256": "sha256:" + "a" * 64,
        "data_version": "a3-core-v1.0.0",
        "data_manifest_sha256": "sha256:" + "b" * 64,
        "preregistration_id": "PR-20260722-01",
        "preregistration_sha256": "sha256:" + "c" * 64,
        "seed": 7,
        "conclusion": "fixture",
    }
    if status == "completed":
        result["primary_metric"] = {"name": "success_rate", "value": 0.75}
    else:
        result["stop_reason"] = "fixture stop"
    return result


def test_results_are_immutable_and_index_is_generated(tmp_path):
    summaries = tmp_path / "summaries"
    index = tmp_path / "index.jsonl"
    destination = finalize_result(_summary(), summaries)
    rebuild_results_index(summaries, index)
    verify_results_index(summaries, index)
    assert len(index.read_text(encoding="utf-8").splitlines()) == 1

    document = json.loads(destination.read_text(encoding="utf-8"))
    document["primary_metric"]["value"] = 1.0
    destination.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(IntegrityError):
        rebuild_results_index(summaries, index)


def test_failed_result_requires_stop_reason(tmp_path):
    invalid = _summary("failed")
    del invalid["stop_reason"]
    with pytest.raises(ValidationError, match="stop_reason"):
        finalize_result(invalid, tmp_path)


def test_freeze_seal_detects_tampering(freeze_source, tmp_path):
    path = tmp_path / "freeze.json"
    create_freeze(freeze_source, path)
    verify_freeze_file(path)
    tampered = json.loads(path.read_text(encoding="utf-8"))
    tampered["thresholds"]["failure"] = 0.9
    path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(IntegrityError):
        verify_freeze_file(path)

