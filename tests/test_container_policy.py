from __future__ import annotations

import importlib.util
import json
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CONTAINER = ROOT / "infra" / "container"
LIB = CONTAINER / "lib"
sys.path.insert(0, str(LIB))

from a3_container_common import (  # noqa: E402
    A3ContainerError,
    atomic_write_json,
    file_inventory,
    inventory_identity,
    load_json,
    validate_digest,
    validate_repo_id,
    validate_revision,
    validate_run_id,
)


def test_compose_has_required_isolation() -> None:
    compose = (CONTAINER / "compose.yaml").read_text(encoding="utf-8")
    assert 'shm_size: "16gb"' in compose
    assert "driver: nvidia" in compose
    assert "count: 3" in compose
    assert "/var/run/docker.sock" not in compose
    assert "privileged:" not in compose
    assert "ipc:" not in compose
    assert "init:" not in compose
    assert "${A3_WORKSPACE_HOST" in compose
    assert "${A3_SSH_HOST_KEYS_HOST" in compose
    assert "${A3_AUTH_KEYS_HOST" in compose


def test_dockerfile_pins_the_training_stack() -> None:
    dockerfile = (CONTAINER / "Dockerfile").read_text(encoding="utf-8")
    assert "nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04@sha256:" in dockerfile
    assert "ghcr.io/astral-sh/uv:0.11.32@sha256:" in dockerfile
    assert "ARG PYTHON_VERSION=3.12.13" in dockerfile
    assert "--frozen" in dockerfile
    assert "--no-install-project" in dockerfile
    assert 'ENTRYPOINT ["/usr/bin/tini"' in dockerfile


def test_sshd_disables_root_and_password_login() -> None:
    config = (CONTAINER / "sshd_config").read_text(encoding="utf-8")
    assert "PermitRootLogin no" in config
    assert "PasswordAuthentication no" in config
    assert "KbdInteractiveAuthentication no" in config
    assert "AuthenticationMethods publickey" in config


def test_ssh_shells_load_the_locked_environment() -> None:
    entrypoint = (CONTAINER / "entrypoint.sh").read_text(encoding="utf-8")
    profile = (CONTAINER / "profile.sh").read_text(encoding="utf-8")
    assert "install_shell_startup" in entrypoint
    assert "enable_public_key_account" in entrypoint
    assert "| chpasswd" in entrypoint
    assert '"u:${A3_ADMIN_USER}:rwx,u:${A3_COLLAB_USER}:r-x,m::rwx"' in entrypoint
    assert "source /etc/profile.d/a3.sh" in entrypoint
    assert 'export PATH="/opt/a3/.venv/bin:${PATH}"' in profile


def test_collaborator_supplementary_groups_are_reset() -> None:
    entrypoint = (CONTAINER / "entrypoint.sh").read_text(encoding="utf-8")
    assert 'usermod --groups "${A3_GROUP_NAME}" "${A3_COLLAB_USER}"' in entrypoint
    assert 'usermod --groups "${A3_GROUP_NAME},sudo" "${A3_ADMIN_USER}"' in entrypoint


def test_image_lock_starts_non_deployable() -> None:
    lock = (CONTAINER / "image.lock").read_text(encoding="utf-8")
    assert f"A3_IMAGE_DIGEST=sha256:{'0' * 64}" in lock
    with pytest.raises(A3ContainerError, match="zero placeholder"):
        validate_digest(f"sha256:{'0' * 64}")


def test_deployment_wrapper_keeps_image_lock_authoritative() -> None:
    wrapper = (CONTAINER / "a3-compose").read_text(encoding="utf-8")
    assert wrapper.index('source "${ENV_FILE}"') < wrapper.index('source "${LOCK_FILE}"')
    assert "checked-out uv.lock does not match image.lock" in wrapper
    assert "image source commit label does not match image.lock" in wrapper
    assert "image repository must not contain a mutable tag" in wrapper
    assert "up -d --force-recreate" in wrapper
    assert '--user "${A3_ADMIN_USER}"' in wrapper
    assert "must name an existing host directory" in wrapper
    assert "project GID must differ from both private user IDs" in wrapper
    assert "missing regular non-empty public-key file" in wrapper


def test_release_area_is_root_owned_and_user_read_only() -> None:
    entrypoint = (CONTAINER / "entrypoint.sh").read_text(encoding="utf-8")
    promote = (CONTAINER / "bin" / "a3-artifact-promote").read_text(encoding="utf-8")
    doctor = (CONTAINER / "bin" / "a3-env-doctor").read_text(encoding="utf-8")
    assert 'install -d -m 2750 -o root -g "${A3_GROUP_NAME}"' in entrypoint
    assert "set_release_ownership(temporary, 0, group.gr_gid)" in promote
    assert '(A3_ROOT / "releases" / "datasets", False)' in doctor


def test_pr_container_build_has_no_registry_write_credentials() -> None:
    workflow = (
        ROOT / ".github" / "workflows" / "remote-training-container.yml"
    ).read_text(encoding="utf-8")
    container_job = workflow.split("\n  container:", maxsplit=1)[1].split(
        "\n  publish:", maxsplit=1
    )[0]
    publish_job = workflow.split("\n  publish:", maxsplit=1)[1]
    assert "packages: write" not in container_job
    assert "id-token: write" not in container_job
    assert "push: false" in container_job
    assert "packages: write" in publish_job
    assert "push: true" in publish_job


def test_workflow_actions_are_commit_pinned() -> None:
    workflow = (
        ROOT / ".github" / "workflows" / "remote-training-container.yml"
    ).read_text(encoding="utf-8")
    action_lines = [line.strip() for line in workflow.splitlines() if "uses:" in line]
    assert action_lines
    for line in action_lines:
        assert re.search(r"@[0-9a-f]{40}(?:\s|$)", line), line


@pytest.mark.parametrize(
    ("function", "valid", "invalid"),
    [
        (validate_revision, "a" * 40, "main"),
        (validate_repo_id, "owner/model-name", "../model"),
        (validate_run_id, "EXP-TASK-A001", "bad run/id"),
        (validate_digest, f"sha256:{'a' * 64}", "sha256:abcd"),
    ],
)
def test_identifier_validation(function, valid: str, invalid: str) -> None:
    assert function(valid) == valid
    with pytest.raises(A3ContainerError):
        function(invalid)


def test_artifact_inventory_and_atomic_json(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    (artifact / "weights.bin").write_bytes(b"weights")
    (artifact / "config.json").write_text('{"model":"test"}\n', encoding="utf-8")

    files = file_inventory(artifact)
    assert [record["path"] for record in files] == ["config.json", "weights.bin"]
    assert len(inventory_identity(files)) == 64

    output = tmp_path / "payload.json"
    atomic_write_json(output, {"files": files})
    assert load_json(output) == {"files": files}


def test_artifact_inventory_only_skips_the_root_manifest(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact"
    nested = artifact / "nested"
    nested.mkdir(parents=True)
    (artifact / "a3-artifact-manifest.json").write_text("{}\n", encoding="utf-8")
    (nested / "a3-artifact-manifest.json").write_text("payload\n", encoding="utf-8")

    files = file_inventory(artifact)

    assert [record["path"] for record in files] == [
        "nested/a3-artifact-manifest.json"
    ]


def test_artifact_inventory_rejects_manifest_named_symlinks(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    target = tmp_path / "outside"
    target.write_text("outside\n", encoding="utf-8")
    link = artifact / "a3-artifact-manifest.json"
    try:
        link.symlink_to(target)
    except OSError:
        pytest.skip("symbolic-link creation is unavailable on this platform")

    with pytest.raises(A3ContainerError, match="symbolic links are forbidden"):
        file_inventory(artifact)


def test_image_manifest_writer_imports() -> None:
    path = LIB / "write_image_manifest.py"
    spec = importlib.util.spec_from_file_location("write_image_manifest", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.BASE_IMAGE.endswith(
        "sha256:ad6d59a3bbf3e82c1c849c9ac09cfc2a3e0bbb8655042fd899be6681b3fe2a85"
    )


def test_env_example_contains_no_real_identity() -> None:
    env_example = (CONTAINER / "env.example").read_text(encoding="utf-8")
    assert "CHANGE_ME" in env_example
    assert "csm" not in env_example.lower()
    assert "password" not in env_example.lower()


def test_no_fake_environment_evidence() -> None:
    verification = ROOT / "evidence" / "environment_verification.json"
    assert not verification.exists()


def test_json_module_is_available_for_ci_smoke() -> None:
    assert json.loads('{"ok": true}') == {"ok": True}
