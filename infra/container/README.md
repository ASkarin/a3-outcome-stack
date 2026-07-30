# Remote training container

This directory defines the only supported remote training environment for A3 OutcomeStack.
It is a rebuildable, digest-pinned CUDA container for two SSH users and three dedicated
RTX 3090 GPUs. It does not reuse the former shared Conda environment and does not include
camera, CAN, gamepad, ROS 2, or local-controller dependencies.

## Tracked inputs

- `Dockerfile`: CUDA 12.8, uv-managed Python 3.12, locked project dependencies, and
  root-owned `/opt/a3/.venv`.
- `compose.yaml`: three GPUs, 16 GiB `/dev/shm`, SSH-only ingress, persistent project
  storage, and no privileged, host-IPC, or Docker-socket access.
- `image.lock`: the only deployable GHCR digest. The zero digest is deliberately
  non-deployable until the first approved image build.
- `entrypoint.sh`: creates the two accounts, installs public keys, enforces ACLs, and
  starts key-only SSH.
- `bin/`: environment inspection, GPU reservation, mirror-only artifact acquisition,
  and administrator-only immutable promotion.

## Administrator deployment inputs

1. Copy `env.example` to `.env`.
2. Replace every `CHANGE_ME` value and set mode `0600`.
3. Create three explicit host directories:
   - the project workspace mounted at `/workspace`;
   - persistent SSH host keys;
   - a read-only authorized-key directory containing exactly
     `admin_authorized_keys` and `collaborator_authorized_keys`.
   The two user names and all three UID/GID values must be distinct and must not
   collide with identities already present in the image.
4. Authenticate Docker to GHCR with a token limited to `read:packages`.
5. Run `./a3-compose config` before any pull or startup.

Real usernames, IP addresses, ports, host paths, private keys, and service tokens never
belong in Git.

## Image publication

Pull requests validate the lock, tests, static container policy, and Docker build.
After an environment change merges to `main`, GitHub Actions publishes:

```text
ghcr.io/askarin/a3-outcome-stack-env:git-<40-character-source-commit>
```

The GHCR package must remain private and linked to the repository. Verify its visibility
and grant the collaborator read access before the administrator approves the first
digest-lock PR.
The workflow reports the immutable digest, source commit, base digest, and `uv.lock`
SHA-256. The administrator records those values in `image.lock` through a separate PR.
`a3-compose pull`, `up`, and `restart` reject the zero placeholder or incomplete locks.
They also reject a mutable repository reference, a checked-out `uv.lock` mismatch, or
an image whose provenance labels differ from `image.lock`.

Between merging an environment-source change and merging its separate digest-lock PR,
the tracked `image.lock` intentionally continues to describe the previously approved
image. During that interval the checked-out `uv.lock` mismatch makes the new revision
non-deployable. CI permits this two-phase state; the deployment wrapper does not.

## Decision authority

The project administrator has final authority for merges, image locking, publication,
deployment, rollback, and artifact promotion. Required CI status checks remain mandatory.
Collaborator review is welcome but optional; it is not an approval gate and carries no
veto over an administrator decision.

## User workflow

Each user clones the repository into:

```text
/workspace/users/<user>/src/a3-outcome-stack
```

Project code runs from that clone without altering `/opt/a3/.venv`:

```bash
PYTHONPATH=src python -m pytest
a3-env-doctor --repo "$PWD" --json
```

Formal GPU commands use exact GPU UUIDs:

```bash
a3-gpu-run \
  --gpus GPU-UUID-1,GPU-UUID-2 \
  --run-id EXP-EXAMPLE-A001 \
  --repo "$PWD" \
  --dataset-manifest /workspace/a3/releases/datasets/owner--dataset@COMMIT/a3-artifact-manifest.json \
  -- \
  python train.py
```

The wrapper refuses dirty Git worktrees by default, obtains cooperative `flock` locks,
sets offline Hugging Face and W&B paths, and writes the run environment and terminal
summary under the invoking user's run directory.

## Artifact acquisition and promotion

Downloads use `hf-mirror.com` only in the acquisition command and require an exact
40-character revision:

```bash
a3-artifact-fetch \
  --repo owner/model \
  --revision 0123456789abcdef0123456789abcdef01234567 \
  --type model
```

The command downloads into the invoking user's staging directory and emits a manifest
with every file size and SHA-256. After review, only the administrator promotes it:

```bash
sudo a3-artifact-promote \
  --manifest /workspace/a3/staging/<user>/models/<artifact>/a3-artifact-manifest.json
```

Promotion re-hashes the source and copy, refuses symlinks and overwrites, then makes the
new release root-owned and group-read-only. Formal training sets
`HF_HUB_OFFLINE=1` and consumes the promoted local path.

## Acceptance boundary

Do not create `evidence/environment_verification.json` or call the environment complete
until all CI checks, two-user permission checks, container restart persistence, three
single-GPU allocations, two- and three-GPU NCCL checks, a 10-minute DataLoader check,
artifact offline reload, and offline experiment logging have passed on the actual host.
The exact order and pass/fail boundary are in `acceptance/README.md`.

After the two-user permission checks, the administrator runs the GPU portion from the
canonical checkout:

```bash
infra/container/acceptance/run_gpu_acceptance.sh
```

It writes raw acceptance outputs under the administrator's run directory. Review those
files before creating the immutable environment verification evidence.
