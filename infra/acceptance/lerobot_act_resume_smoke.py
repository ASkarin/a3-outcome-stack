"""Create a tiny LeRobotDataset and verify official ACT checkpoint/resume."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from lerobot.datasets.lerobot_dataset import LeRobotDataset


def _write_dataset(root: Path) -> int:
    joint_names = [f"L{index}" for index in range(1, 8)]
    state_names = [
        *[f"{name}.pos" for name in joint_names],
        *[f"{name}.vel" for name in joint_names],
    ]
    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (14,),
            "names": state_names,
        },
        "observation.environment_state": {
            "dtype": "float32",
            "shape": (14,),
            "names": state_names,
        },
        "action": {
            "dtype": "float32",
            "shape": (7,),
            "names": [f"{name}.pos" for name in joint_names],
        },
    }
    dataset = LeRobotDataset.create(
        repo_id="local/a3-act-resume-smoke",
        fps=10,
        features=features,
        root=root,
        robot_type="a3",
        use_videos=False,
    )
    for step in range(8):
        position = np.linspace(0.0, 0.06, 7, dtype=np.float32) + step * 0.001
        velocity = np.full(7, 0.01, dtype=np.float32)
        state = np.concatenate((position, velocity))
        dataset.add_frame(
            {
                "observation.state": state,
                "observation.environment_state": state,
                "action": position + 0.002,
                "task": "synthetic ACT checkpoint resume smoke",
            }
        )
    dataset.save_episode()
    dataset.finalize()
    reloaded = LeRobotDataset(repo_id="local/a3-act-resume-smoke", root=root)
    if len(reloaded) != 8:
        raise RuntimeError(f"expected 8 dataset frames, found {len(reloaded)}")
    return len(reloaded)


def _run(command: list[str], log_path: Path) -> None:
    with log_path.open("w", encoding="utf-8") as log:
        completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, text=True)
    if completed.returncode != 0:
        raise RuntimeError(f"command failed with exit {completed.returncode}; see {log_path}")


def _training_step(output_dir: Path) -> int:
    path = output_dir / "checkpoints/last/training_state/training_step.json"
    with path.open(encoding="utf-8") as handle:
        return int(json.load(handle)["step"])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-dir", type=Path, required=True)
    args = parser.parse_args()
    work_dir = args.work_dir.resolve()
    if work_dir.exists():
        raise FileExistsError(f"refusing to overwrite acceptance directory: {work_dir}")
    work_dir.mkdir(parents=True)

    train_cli = shutil.which("lerobot-train")
    if train_cli is None:
        raise RuntimeError("lerobot-train is not installed in the active environment")

    dataset_root = work_dir / "dataset"
    output_dir = work_dir / "training"
    frame_count = _write_dataset(dataset_root)
    initial_command = [
        train_cli,
        "--dataset.repo_id=local/a3-act-resume-smoke",
        f"--dataset.root={dataset_root}",
        "--dataset.use_imagenet_stats=false",
        "--policy.type=act",
        "--policy.device=cpu",
        "--policy.push_to_hub=false",
        "--policy.pretrained_backbone_weights=null",
        "--policy.dim_model=32",
        "--policy.n_heads=4",
        "--policy.dim_feedforward=64",
        "--policy.n_encoder_layers=1",
        "--policy.n_decoder_layers=1",
        "--policy.use_vae=false",
        "--policy.chunk_size=2",
        "--policy.n_action_steps=2",
        f"--output_dir={output_dir}",
        "--job_name=a3-act-resume-smoke",
        "--batch_size=2",
        "--num_workers=0",
        "--steps=1",
        "--save_freq=1",
        "--log_freq=1",
        "--env_eval_freq=0",
        "--wandb.enable=false",
    ]
    _run(initial_command, work_dir / "initial.log")
    initial_step = _training_step(output_dir)
    if initial_step != 1:
        raise RuntimeError(f"initial checkpoint step is {initial_step}, expected 1")

    train_config = output_dir / "checkpoints/last/pretrained_model/train_config.json"
    optimizer_state = output_dir / "checkpoints/last/training_state/optimizer_state.safetensors"
    if not train_config.is_file() or not optimizer_state.is_file():
        raise RuntimeError("official checkpoint is missing its config or optimizer state")
    resume_command = [
        train_cli,
        f"--config_path={train_config}",
        "--resume=true",
        "--steps=2",
        "--log_freq=1",
    ]
    _run(resume_command, work_dir / "resume.log")
    resumed_step = _training_step(output_dir)
    if resumed_step != 2:
        raise RuntimeError(f"resumed checkpoint step is {resumed_step}, expected 2")

    summary = {
        "schema_version": "a3-lerobot-act-resume-smoke-v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "status": "passed",
        "scope": {
            "synthetic_data_only": True,
            "real_robot_hardware_verified": False,
            "policy": "ACT",
            "device": "cpu",
        },
        "dataset": {
            "repo_id": "local/a3-act-resume-smoke",
            "root": str(dataset_root),
            "frames": frame_count,
            "finalized_and_reloaded": True,
        },
        "training": {
            "output_dir": str(output_dir),
            "initial_checkpoint_step": initial_step,
            "resumed_checkpoint_step": resumed_step,
            "optimizer_state_present": True,
        },
        "logs": [str(work_dir / "initial.log"), str(work_dir / "resume.log")],
    }
    summary_path = work_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
