from __future__ import annotations

import random

import numpy as np
import pytest
import torch

from embodied_ai.ops.checkpoints import CheckpointManager, verify_checkpoint
from embodied_ai.ops.errors import IntegrityError, StateConflict, ValidationError


def _identity(attempt="EXP-A-ACT-AAAAAAAAAAAA-A001"):
    return {
        "experiment_id": "EXP-A-ACT-AAAAAAAAAAAA",
        "attempt_id": attempt,
        "git_commit": "1" * 40,
        "config_sha256": "sha256:" + "a" * 64,
        "data_version": "a3-core-v0.1.0",
        "data_manifest_sha256": "sha256:" + "b" * 64,
        "seed": 123,
    }


def _seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _new_training_state():
    model = torch.nn.Linear(3, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9)
    return model, optimizer


def _train(model, optimizer, start, end):
    for _ in range(start, end):
        inputs = torch.randn(4, 3)
        targets = torch.randn(4, 1)
        scale = random.random() + float(np.random.random())
        optimizer.zero_grad()
        loss = ((model(inputs) - targets) ** 2).mean() * scale
        loss.backward()
        optimizer.step()


def _state(model, optimizer, step):
    return {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": None,
        "scaler_state_dict": None,
        "global_step": step,
        "sampler_state": {"position": step},
    }


def test_interrupted_run_matches_uninterrupted_run(tmp_path):
    _seed_everything(123)
    full_model, full_optimizer = _new_training_state()
    _train(full_model, full_optimizer, 0, 8)
    full_next_rng = (random.random(), float(np.random.random()), torch.rand(3))

    _seed_everything(123)
    partial_model, partial_optimizer = _new_training_state()
    _train(partial_model, partial_optimizer, 0, 3)
    manager = CheckpointManager(tmp_path / "checkpoints", _identity())
    metadata = manager.save(_state(partial_model, partial_optimizer, 3))
    verify_checkpoint(tmp_path / "checkpoints" / metadata["checkpoint"])

    # Simulate a fresh process whose initialization consumes random values.
    resumed_model, resumed_optimizer = _new_training_state()
    resumed = manager.load_latest(restore_rng=True)
    resumed_model.load_state_dict(resumed["model_state_dict"])
    resumed_optimizer.load_state_dict(resumed["optimizer_state_dict"])
    _train(resumed_model, resumed_optimizer, resumed["global_step"], 8)
    resumed_next_rng = (random.random(), float(np.random.random()), torch.rand(3))

    for name, expected in full_model.state_dict().items():
        torch.testing.assert_close(resumed_model.state_dict()[name], expected, rtol=0, atol=0)
    assert resumed_next_rng[0] == full_next_rng[0]
    assert resumed_next_rng[1] == full_next_rng[1]
    torch.testing.assert_close(resumed_next_rng[2], full_next_rng[2], rtol=0, atol=0)


def test_exact_resume_refuses_missing_sampler_state(tmp_path):
    _seed_everything(1)
    model, optimizer = _new_training_state()
    state = _state(model, optimizer, 0)
    state["sampler_state"] = None
    manager = CheckpointManager(tmp_path, _identity())
    with pytest.raises(ValidationError, match="sampler_state"):
        manager.save(state, exact_resume=True)


def test_resume_rejects_identity_mismatch_and_tampering(tmp_path):
    _seed_everything(1)
    model, optimizer = _new_training_state()
    manager = CheckpointManager(tmp_path, _identity())
    metadata = manager.save(_state(model, optimizer, 0))

    wrong = CheckpointManager(tmp_path, _identity("EXP-A-ACT-AAAAAAAAAAAA-A002"))
    with pytest.raises(StateConflict, match="identity mismatch"):
        wrong.load_latest()

    checkpoint = tmp_path / metadata["checkpoint"]
    with checkpoint.open("ab") as handle:
        handle.write(b"tamper")
    with pytest.raises(IntegrityError, match="hash mismatch"):
        verify_checkpoint(checkpoint)

