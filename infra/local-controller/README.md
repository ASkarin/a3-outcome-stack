# Local controller host

This directory defines the administrator-controlled baseline for the `a3-local`
Ubuntu host. It deliberately does not contain a real identity, network address,
SSH connection parameter, private key, or device-specific udev rule.

The local controller is not a copy of the remote training container:

- the administrator retains system and deployment authority;
- each collaborator uses an independent account, key, clone, and uv environment;
- the active deployment is administrator-built and read-only to human collaborators;
- a controlled operator can use only the local Unix-domain control service;
- raw camera, input, serial, and CAN access belongs only to the hardware service;
- the base service supports the mock backend only and is installed disabled.

## Administrator workflow

Run the scripts from a physical console or a verified Tailscale SSH session. The
security phase creates a timed rollback and must be confirmed from a second working
session.

```bash
sudo bash infra/local-controller/bootstrap-host.sh base
sudo bash infra/local-controller/bootstrap-host.sh gpu
sudo --preserve-env=SSH_CONNECTION A3_CONFIRM_CONSOLE=YES \
  bash infra/local-controller/bootstrap-host.sh security
sudo --preserve-env=SSH_CONNECTION \
  bash infra/local-controller/bootstrap-host.sh confirm-security
sudo bash infra/local-controller/bootstrap-host.sh check
```

The GPU phase installs the selected driver but does not reboot automatically. Reboot
only after the package operation succeeds and physical-console recovery is available.
The uv environment installs PyTorch's wheel-scoped CUDA 12.8 runtime libraries; it
does not install a system CUDA toolkit or `nvcc`. The selected LeRobot extra contains
SmolVLA support but excludes its device-facing `core_scripts` extra.

After the implementation branch has passed review, deploy a clean, committed checkout
as an immutable release:

```bash
sudo /usr/local/sbin/a3-local-deploy-release install <clean-source-checkout>
```

If the locked PyPI CDN is unusably slow, the administrator may pass an HTTPS
index through `A3_PYPI_MIRROR`. The deployer exports only registry dependencies
with their lockfile hashes, preinstalls them from that index, and still runs the
same final frozen sync. Git dependencies and the pinned PyTorch wheels remain on
their locked sources.

```bash
sudo A3_PYPI_MIRROR=https://mirror.example/simple \
  /usr/local/sbin/a3-local-deploy-release install <clean-source-checkout>
```

Rollback switches the `current` link to an existing immutable release; it never
overwrites a prior release. The project is installed non-editably so moving the
completed staging directory into its commit-scoped final path cannot leave a
Python import path pointing back to the staging name.

The security phase:

1. verifies that the current SSH peer routes through `tailscale0`;
2. backs up the existing OpenSSH and UFW configuration;
3. installs public-key-only OpenSSH policy;
4. allows only the detected SSH service through `tailscale0`;
5. schedules an automatic rollback;
6. waits for `confirm-security` from a second verified session.

Tailnet Grants are a separate control-plane step. Do not provision a collaborator
until their individual Tailscale access has been limited to the `a3-local` SSH
service.

## Collaborator lifecycle

No placeholder collaborator account is created by the bootstrap. After receiving an
individual public key and confirming the matching tailnet Grant, provision a named
account with:

```bash
sudo A3_TAILNET_GRANT_CONFIRMED=YES \
  bash infra/local-controller/manage-collaborator.sh \
  provision <account> <public-key-file>
```

Add `--operator` only after the person is approved for supervised on-site operation.
It grants access to the Unix socket, not to raw devices or system administration.

Revocation preserves the home directory and archives the old authorized-key file:

```bash
sudo A3_TAILNET_GRANT_REVOKED=YES \
  bash infra/local-controller/manage-collaborator.sh revoke <account>
```

## Deferred hardware integration

Do not modify the base service to expose real hardware before enumeration and the
physical safety gate:

- SocketCAN interfaces must be moved into a service-only network namespace.
- SLCAN or vendor serial nodes require service-only udev permissions.
- camera and input nodes require explicit device allowlists.
- a real operator permit requires frozen calibration and safety files, a verified
  physical e-stop, the exact deployed Git commit, and `hardware_verified=true`.

The base unit has `PrivateDevices=true`, restricts address families to `AF_UNIX`, does
not restart automatically, and invokes only `robot control serve-mock`.
