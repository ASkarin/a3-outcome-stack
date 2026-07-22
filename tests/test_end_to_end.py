from __future__ import annotations

import torch

from embodied_ai.ops.checkpoints import CheckpointManager
from embodied_ai.ops.experiments import attempt_id, register_experiment
from embodied_ai.ops.freeze import create_freeze, verify_freeze_file
from embodied_ai.ops.manifests import build_dataset_manifest, verify_dataset_manifest, write_manifest_immutable
from embodied_ai.ops.results import finalize_result, rebuild_results_index, verify_results_index
from embodied_ai.ops.runstate import create_run_state, transition_run_state


def test_stage0_evidence_chain(experiment_source, freeze_source, tmp_path):
    data_root = tmp_path / "data"
    episode_root = data_root / "episode-0001"
    episode_root.mkdir(parents=True)
    (episode_root / "observation.bin").write_bytes(b"fixture")
    dataset = build_dataset_manifest(
        data_root,
        data_version="a3-core-v0.1.0",
        episodes=[
            {
                "episode_id": "episode-0001",
                "session_id": "session-0001",
                "source_trajectory_id": "trajectory-0001",
                "task": "A",
                "split": "train",
                "relative_path": "episode-0001",
            }
        ],
        preregistration_id="PR-20260722-01",
        preregistration_sha256=experiment_source["preregistration_sha256"],
    )
    dataset_path = tmp_path / "metadata" / "datasets" / "a3-core-v0.1.0.json"
    write_manifest_immutable(dataset, dataset_path, kind="dataset")
    verify_dataset_manifest(data_root, dataset)

    source = dict(experiment_source)
    source["data_manifest_sha256"] = dataset["manifest_sha256"]
    registered = register_experiment(
        source,
        specs_dir=tmp_path / "experiments" / "specs",
        registry_path=tmp_path / "experiments" / "registry.csv",
    )
    attempt = attempt_id(registered["experiment_id"], 1)
    run_state = tmp_path / "runs" / registered["experiment_id"] / attempt / "state.json"
    create_run_state(run_state, registered["experiment_id"], attempt)
    transition_run_state(run_state, "running")

    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    manager = CheckpointManager(
        run_state.parent / "checkpoints",
        {
            "experiment_id": registered["experiment_id"],
            "attempt_id": attempt,
            "git_commit": source["git_commit"],
            "config_sha256": source["config_sha256"],
            "data_version": source["data_version"],
            "data_manifest_sha256": source["data_manifest_sha256"],
            "seed": source["seed"],
        },
    )
    checkpoint = manager.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": None,
            "scaler_state_dict": None,
            "global_step": 0,
            "sampler_state": {"position": 0},
        }
    )

    summary_path = finalize_result(
        {
            "experiment_id": registered["experiment_id"],
            "attempt_id": attempt,
            "status": "completed",
            "started_at_utc": "2026-07-22T00:00:00Z",
            "ended_at_utc": "2026-07-22T00:01:00Z",
            "git_commit": source["git_commit"],
            "config_sha256": source["config_sha256"],
            "data_version": source["data_version"],
            "data_manifest_sha256": source["data_manifest_sha256"],
            "preregistration_id": source["preregistration_id"],
            "preregistration_sha256": source["preregistration_sha256"],
            "seed": source["seed"],
            "checkpoint_sha256": checkpoint["checkpoint_sha256"],
            "primary_metric": {"name": "fixture_loss", "value": 0.0},
            "conclusion": "infrastructure fixture only",
        },
        tmp_path / "results" / "summaries",
    )
    transition_run_state(run_state, "completed")
    index = tmp_path / "results" / "index.jsonl"
    rebuild_results_index(tmp_path / "results" / "summaries", index)
    verify_results_index(tmp_path / "results" / "summaries", index)

    freeze = dict(freeze_source)
    freeze["data"] = {
        "version": source["data_version"],
        "manifest_sha256": source["data_manifest_sha256"],
    }
    freeze["checkpoints"] = [
        {"experiment_id": registered["experiment_id"], "sha256": checkpoint["checkpoint_sha256"]}
    ]
    freeze_path = tmp_path / "metadata" / "freezes" / "FRZ-20261231-CORE.json"
    create_freeze(freeze, freeze_path)
    verify_freeze_file(freeze_path)
    assert summary_path.is_file()

