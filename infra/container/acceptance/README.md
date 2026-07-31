# Host acceptance procedure

Keep raw output under the administrator's persistent run or admin directory. Existing
locked-image verification JSON is historical evidence; it is not a gate for the
current mutable Python environment.

## 1. External SSH boundary

From each member's own workstation, connect with that member's private key and confirm
the reported user. Root and password-only authentication must remain rejected:

```bash
ssh a3-training id
```

The administrator performs the root/password rejection probes without recording
addresses, ports, user-specific key paths, or private material in Git.

## 2. Account, filesystem, and shared Python boundary

Run the doctor once as each account from a personal clone:

```bash
a3-env-doctor --repo "$PWD" --json
```

As the collaborator, all of the following conditions must hold:

```bash
! sudo -n true
! command -v docker
test -r /workspace/a3/python-env
test ! -w /workspace/a3/python-env
test -w "/workspace/a3/staging/$USER"
test -w "/workspace/a3/runs/$USER"
test -w /workspace/a3/cache
test -w /workspace/a3/locks
test ! -w /workspace/a3/releases/datasets
test ! -w /workspace/a3/releases/models
test ! -w /workspace/projects/a3-outcome-stack
test ! -r "/workspace/users/<admin>"
```

The administrator must be able to run `sudo -n true`, modify the shared Python
environment through `a3-python`, and keep both release directories read-only during an
ordinary login.

## 3. Mutable Python cutover

Choose a small package that is absent from the base seed:

```bash
a3-python install <small-package>
a3-python snapshot
```

Confirm that the collaborator can import it but cannot run `a3-python install` or write
the shared environment. Run `./a3-compose restart` on the host, reconnect both users,
and confirm the import still succeeds. Finally uninstall the temporary package if it
is not useful to the project. The install, uninstall, and snapshots must appear in
`/workspace/a3/python-env-history/operations.jsonl`.

## 4. GPU, shared memory, and run records

From the clean canonical checkout, the administrator runs:

```bash
infra/container/acceptance/run_gpu_acceptance.sh
```

The script reserves all three GPU UUIDs through `a3-gpu-run`, then records:

- one tensor allocation on each GPU;
- 100 all-reduce iterations on the preferred first two GPUs;
- 100 all-reduce iterations across all three GPUs;
- a three-rank, two-workers-per-rank DataLoader run lasting 600 seconds;
- `environment.json`, `summary.json`, metrics placeholders, and
  `python-packages.txt`;
- a real TensorBoard event file and a real W&B offline run file.

`environment.json` must record the shared Python executable and the SHA-256 of the
complete package list. Retain raw output even if a check fails.

## 5. Artifact and offline loading

Fetch a deliberately small model or dataset at an exact repository commit:

```bash
a3-artifact-fetch --repo <owner/name> --revision <commit> --type <model|dataset>
```

Review the manifest, promote it with
`sudo a3-artifact-promote --manifest <path>`, and confirm that a second promotion to
the same destination is rejected. Load the promoted local path with
`HF_HUB_OFFLINE=1`; merely listing files is not an offline-load test.

## 6. Restart persistence

Record SSH host-key fingerprints and hashes of small files in each persistent area.
Run `./a3-compose restart`, reconnect both users, and verify that host keys, authorized
keys, personal clones, the shared Python environment and its history, promoted
artifacts, and run records remain unchanged.

Container recreation is a different operation and must be invoked explicitly with
`./a3-compose recreate`.
