# A3 OutcomeStack

A3 OutcomeStack is a reproducible real-robot learning stack for EduLite A3, spanning data capture, ACT/VLA training, deployment, evaluation, and action-outcome prediction. This repository is the implementation and experiment evidence source. Planning, decisions, and the canonical preregistration live in the local control repository; the frozen preregistration snapshot in this repository is byte-identical and hash-bound to every formal experiment.

The canonical project slug and Python distribution are `a3-outcome-stack`; the Python namespace is `a3_outcome_stack`. The legacy `embodied_ai` import and `embodied-ai` CLI remain compatibility aliases for reproducing Stage 0/1A records and must not be used by new code.

## Evidence model

- `experiments/specs/` contains immutable experiment specifications. `experiments/registry.csv` is generated from specs plus the latest immutable terminal summary for each experiment.
- `metadata/datasets/` and `metadata/assets/` contain immutable SHA-256 manifests for external data and assets.
- `runs/` contains checkpoints and logs and is ignored by Git.
- `results/summaries/` contains immutable per-attempt summaries. `results/index.jsonl` is generated from them.
- `metadata/freezes/` contains sealed final-evaluation manifests.
- Raw data, videos, checkpoints, and model weights must not enter Git.

Experiment IDs have the form `EXP-<TASK>-<POLICY>-<SPEC_SHA12>`. An exact process resume keeps its attempt ID, such as `...-A001`; a from-scratch rerun increments the attempt. Any change to code, configuration, seed, or data identity creates a new experiment. Reusing a checkpoint across identities is a `warm_start`, not a resume.

Dataset versions have the form `<dataset_slug>-vMAJOR.MINOR.PATCH`. Schema or split-rule changes increment major, episode additions/removals or recleaning increment minor, and metadata-only corrections increment patch. The version is always paired with a content SHA-256.

## Commands

Run without installing into the environment:

```bash
PYTHONPATH=src python -m a3_outcome_stack.ops doctor --root .
PYTHONPATH=src python -m a3_outcome_stack.ops experiment register --spec config.json
PYTHONPATH=src python -m a3_outcome_stack.ops dataset manifest --root DATA --version a3-core-v0.1.0 --episodes episodes.json --output metadata/datasets/a3-core-v0.1.0.json
PYTHONPATH=src python -m a3_outcome_stack.ops dataset verify --root DATA --manifest metadata/datasets/a3-core-v0.1.0.json
PYTHONPATH=src python -m a3_outcome_stack.ops asset manifest --root ASSET --asset-id camera-calibration --kind calibration --output metadata/assets/camera-calibration.json
PYTHONPATH=src python -m a3_outcome_stack.ops checkpoint verify --checkpoint runs/EXP/.../step-000000000100.pt
PYTHONPATH=src python -m a3_outcome_stack.ops result reindex
PYTHONPATH=src python -m a3_outcome_stack.ops freeze verify --manifest metadata/freezes/FRZ-....json
```

An editable or packaged installation exposes the canonical `a3-outcome-stack` command.

All JSON is UTF-8 canonicalized before identity hashes are computed. Validation errors exit with code 2, integrity mismatches with 3, and lifecycle conflicts with 4. There is no force-resume path.

## Stage 1A robot contract

The hardware-neutral A3 interface lives in `src/a3_outcome_stack/robot/`. `SafeRobot` is the only public actuator gateway; deterministic mock and strict replay backends exercise the same observation/action, timing, and latched safety contracts. The official SDK is pinned by `configs/upstream/edulite_a3.lock.json` and is imported only when an SDK backend is explicitly constructed.

Stage 1A is hardware-unverified. The checked-in calibration and safety files contain null, unfrozen hardware values and therefore cannot enable motors. Mock limits and timeouts are synthetic test fixtures and must never be reused for hardware.

```bash
PYTHONPATH=src python -m a3_outcome_stack.ops robot contract verify
PYTHONPATH=src python -m a3_outcome_stack.ops robot upstream verify --checkout .cache/upstream/EDULITE_A3
PYTHONPATH=src python -m a3_outcome_stack.ops robot mock record --output runs/stage1a/mock-001 --steps 3
PYTHONPATH=src python -m a3_outcome_stack.ops robot trace verify --trace runs/stage1a/mock-001
PYTHONPATH=src python -m a3_outcome_stack.ops robot replay --trace runs/stage1a/mock-001 --strict
PYTHONPATH=src python -m a3_outcome_stack.ops robot doctor --root . --upstream-checkout .cache/upstream/EDULITE_A3
```

## Verification

Use the existing `pytorch` conda environment; no dependency installation or network access is required:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python -m pytest -p no:cacheprovider
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python -m a3_outcome_stack.ops doctor --root .
```
