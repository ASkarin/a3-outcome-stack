# Host acceptance procedure

Run this procedure only after `image.lock` contains the approved GHCR digest and the
container was created from a fresh checkout. Keep raw command output under the
administrator's acceptance run; do not create `evidence/environment_verification.json`
until every item passes.

## 1. External SSH boundary

From each member's own workstation, connect with that member's private key and confirm
the reported user. Then confirm that root and password-only authentication are rejected:

```bash
ssh -p <port> <admin>@<host> id
ssh -p <port> <collaborator>@<host> id
ssh -p <port> -o BatchMode=yes -o PreferredAuthentications=publickey root@<host> true
ssh -p <port> -o BatchMode=yes -o PubkeyAuthentication=no \
  -o PreferredAuthentications=password <admin>@<host> true
```

The first two commands must succeed. The final two must fail.

## 2. Account and filesystem boundary

Run the doctor once as each account from a clean personal clone:

```bash
a3-env-doctor --repo "$PWD" --json
```

As the collaborator, all of the following conditions must hold:

```bash
! sudo -n true
! command -v docker
test -w "/workspace/a3/staging/$USER"
test -w "/workspace/a3/runs/$USER"
test -w /workspace/a3/cache
test -w /workspace/a3/locks
test ! -w /workspace/a3/releases/datasets
test ! -w /workspace/a3/releases/models
test ! -w /workspace/projects/a3-outcome-stack
test ! -r "/workspace/users/<admin>"
```

The administrator must be able to run `sudo -n true`, while both accounts must see the
release directories as read-only during ordinary login.

## 3. GPU, shared memory, and run records

From the clean canonical checkout, the administrator runs:

```bash
infra/container/acceptance/run_gpu_acceptance.sh
```

The script reserves all three GPU UUIDs through `a3-gpu-run`, then records:

- one tensor allocation on each GPU;
- 100 all-reduce iterations on the preferred first two GPUs;
- 100 all-reduce iterations across all three GPUs;
- a three-rank, two-workers-per-rank DataLoader run lasting 600 seconds;
- `environment.json`, `summary.json`, metrics placeholders, TensorBoard, and W&B
  offline directories.

Retain the raw output even if a check fails. A failed run is diagnostic evidence, not a
passing environment record.

## 4. Artifact and offline loading

Choose a deliberately small model or dataset and an exact 40-character repository
commit. Fetch it as an ordinary user:

```bash
a3-artifact-fetch --repo <owner/name> --revision <commit> --type <model|dataset>
```

Review its generated manifest, promote it with
`sudo a3-artifact-promote --manifest <path>`, and confirm that a second promotion to the
same destination is rejected.
Disconnect outbound networking or set `HF_HUB_OFFLINE=1`, then load the promoted local
path through the same project loader that training will use. Record the command and
result; merely listing files is not an offline-load acceptance test.

## 5. Restart persistence

Record the SSH host-key fingerprints and hashes of a small file in each persistent area.
Run `./a3-compose restart`, reconnect both users, and verify that the fingerprints,
authorized keys, personal clones, promoted artifact, and acceptance run remain
unchanged. Confirm the running container image ID still corresponds to the digest in
`image.lock`.

Only after these checks and the required GitHub review/CI checks pass may the reviewed
results be summarized in `evidence/environment_verification.json` and merged.
