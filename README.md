# EduLite A3 Embodied AI

This repository is the implementation and experiment evidence source for the EduLite A3 project. Planning, decisions, and the canonical preregistration live in the local control repository; the frozen preregistration snapshot in this repository is byte-identical and hash-bound to every formal experiment.

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
PYTHONPATH=src python -m embodied_ai.ops doctor --root .
PYTHONPATH=src python -m embodied_ai.ops experiment register --spec config.json
PYTHONPATH=src python -m embodied_ai.ops dataset manifest --root DATA --version a3-core-v0.1.0 --episodes episodes.json --output metadata/datasets/a3-core-v0.1.0.json
PYTHONPATH=src python -m embodied_ai.ops dataset verify --root DATA --manifest metadata/datasets/a3-core-v0.1.0.json
PYTHONPATH=src python -m embodied_ai.ops asset manifest --root ASSET --asset-id camera-calibration --kind calibration --output metadata/assets/camera-calibration.json
PYTHONPATH=src python -m embodied_ai.ops checkpoint verify --checkpoint runs/EXP/.../step-000000000100.pt
PYTHONPATH=src python -m embodied_ai.ops result reindex
PYTHONPATH=src python -m embodied_ai.ops freeze verify --manifest metadata/freezes/FRZ-....json
```

All JSON is UTF-8 canonicalized before identity hashes are computed. Validation errors exit with code 2, integrity mismatches with 3, and lifecycle conflicts with 4. There is no force-resume path.

## Verification

Use the existing `pytorch` conda environment; no dependency installation or network access is required:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python -m pytest -p no:cacheprovider
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python -m embodied_ai.ops doctor --root .
```
