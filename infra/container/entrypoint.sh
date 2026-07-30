#!/usr/bin/env bash
set -euo pipefail

readonly WORKSPACE_ROOT="/workspace"
readonly CONFIG_ROOT="/run/a3-config"
readonly SSH_HOST_KEY_ROOT="/etc/ssh/a3_host_keys"
readonly RUNTIME_CONFIG="/etc/a3/runtime.json"

fail() {
    echo "a3-entrypoint: $*" >&2
    exit 2
}

require_var() {
    local name="$1"
    test -n "${!name:-}" || fail "required environment variable ${name} is missing"
}

validate_user_name() {
    local value="$1"
    [[ "${value}" =~ ^[a-z_][a-z0-9_-]{0,30}$ ]] \
        || fail "invalid Linux user name: ${value}"
}

validate_id() {
    local name="$1"
    local value="$2"
    [[ "${value}" =~ ^[0-9]+$ ]] && (( value >= 1000 && value <= 60000 )) \
        || fail "${name} must be an integer from 1000 through 60000"
}

ensure_project_group() {
    local name="$1"
    local gid="$2"
    local existing

    existing="$(getent group "${name}" || true)"
    if [[ -n "${existing}" ]]; then
        [[ "$(cut -d: -f3 <<<"${existing}")" == "${gid}" ]] \
            || fail "group ${name} exists with a different GID"
        return
    fi

    if getent group "${gid}" >/dev/null; then
        fail "requested project GID ${gid} is already in use"
    fi
    groupadd --gid "${gid}" "${name}"
}

ensure_user() {
    local name="$1"
    local uid="$2"
    local home="$3"
    local project_group="$4"
    local existing
    local private_group

    existing="$(getent passwd "${name}" || true)"
    if [[ -n "${existing}" ]]; then
        [[ "$(cut -d: -f3 <<<"${existing}")" == "${uid}" ]] \
            || fail "user ${name} exists with a different UID"
        [[ "$(cut -d: -f4 <<<"${existing}")" == "${uid}" ]] \
            || fail "user ${name} exists with a different primary GID"
        [[ "$(cut -d: -f6 <<<"${existing}")" == "${home}" ]] \
            || fail "user ${name} exists with a different home"
        [[ "$(cut -d: -f7 <<<"${existing}")" == "/bin/bash" ]] \
            || fail "user ${name} exists with a different shell"
        private_group="$(getent group "${name}" || true)"
        [[ -n "${private_group}" && "$(cut -d: -f3 <<<"${private_group}")" == "${uid}" ]] \
            || fail "private group ${name} is missing or has a different GID"
    else
        if getent passwd "${uid}" >/dev/null; then
            fail "requested UID ${uid} is already in use"
        fi
        if getent group "${name}" >/dev/null; then
            fail "private group ${name} already exists"
        fi
        if getent group "${uid}" >/dev/null; then
            fail "private GID ${uid} is already in use"
        fi
        groupadd --gid "${uid}" "${name}"
        useradd \
            --uid "${uid}" \
            --gid "${name}" \
            --groups "${project_group}" \
            --home-dir "${home}" \
            --shell /bin/bash \
            --no-create-home \
            "${name}"
    fi

    usermod --append --groups "${project_group}" "${name}"
    install -d -m 0700 -o "${name}" -g "${name}" "${home}"
}

enable_public_key_account() {
    local user="$1"
    local random_password

    # OpenSSH rejects a shadow-locked account before it evaluates authorized_keys.
    # Give the account an unknown, short-lived random password so public-key
    # authentication can proceed. Password and keyboard-interactive authentication
    # remain disabled in sshd_config, and the random plaintext is immediately lost.
    random_password="$(head -c 48 /dev/urandom | base64 | tr -d '\n')"
    printf '%s:%s\n' "${user}" "${random_password}" | chpasswd
    unset random_password
}

install_authorized_keys() {
    local user="$1"
    local source="$2"
    local home
    local ssh_directory
    local destination

    [[ -s "${source}" ]] || fail "authorized_keys file is missing or empty: ${source}"
    home="$(getent passwd "${user}" | cut -d: -f6)"
    ssh_directory="${home}/.ssh"
    destination="${ssh_directory}/authorized_keys"
    [[ ! -L "${ssh_directory}" ]] || fail "refusing symbolic-link SSH directory: ${ssh_directory}"
    [[ ! -e "${ssh_directory}" || -d "${ssh_directory}" ]] \
        || fail "SSH path is not a directory: ${ssh_directory}"
    [[ ! -L "${destination}" ]] || fail "refusing symbolic-link authorized_keys: ${destination}"
    [[ ! -e "${destination}" || -f "${destination}" ]] \
        || fail "authorized_keys path is not a regular file: ${destination}"
    install -d -m 0700 -o "${user}" -g "${user}" "${ssh_directory}"
    install -m 0600 -o "${user}" -g "${user}" "${source}" "${destination}"
}

install_shell_startup() {
    local user="$1"
    local home
    local startup
    local source_line='source /etc/profile.d/a3.sh'

    home="$(getent passwd "${user}" | cut -d: -f6)"
    for startup in "${home}/.profile" "${home}/.bashrc"; do
        [[ ! -L "${startup}" ]] || fail "refusing symbolic-link shell startup file: ${startup}"
        [[ ! -e "${startup}" || -f "${startup}" ]] \
            || fail "shell startup path is not a regular file: ${startup}"
        if [[ ! -e "${startup}" ]]; then
            install -m 0644 -o "${user}" -g "${user}" /dev/null "${startup}"
        fi
        if ! grep -Fqx "${source_line}" "${startup}"; then
            printf '\n%s\n' "${source_line}" >>"${startup}"
        fi
        chown "${user}:${user}" "${startup}"
        chmod 0644 "${startup}"
    done
}

set_read_acl() {
    local path="$1"
    setfacl -m "u::rwx,g::r-x,o::---,m::r-x" "${path}"
    setfacl -m "d:u::rwx,d:g::r-x,d:o::---,d:m::r-x" "${path}"
}

set_write_acl() {
    local path="$1"
    setfacl -m "u::rwx,g::rwx,o::---,m::rwx" "${path}"
    setfacl -m "d:u::rwx,d:g::rwx,d:o::---,d:m::rwx" "${path}"
}

for variable in \
    A3_ADMIN_USER \
    A3_ADMIN_UID \
    A3_COLLAB_USER \
    A3_COLLAB_UID \
    A3_GROUP_GID \
    A3_GROUP_NAME \
    A3_IMAGE_DIGEST
do
    require_var "${variable}"
done

validate_user_name "${A3_ADMIN_USER}"
validate_user_name "${A3_COLLAB_USER}"
[[ "${A3_ADMIN_USER}" != "${A3_COLLAB_USER}" ]] || fail "admin and collaborator must differ"
validate_id A3_ADMIN_UID "${A3_ADMIN_UID}"
validate_id A3_COLLAB_UID "${A3_COLLAB_UID}"
validate_id A3_GROUP_GID "${A3_GROUP_GID}"

ensure_project_group "${A3_GROUP_NAME}" "${A3_GROUP_GID}"
ensure_user \
    "${A3_ADMIN_USER}" \
    "${A3_ADMIN_UID}" \
    "${WORKSPACE_ROOT}/users/${A3_ADMIN_USER}" \
    "${A3_GROUP_NAME}"
ensure_user \
    "${A3_COLLAB_USER}" \
    "${A3_COLLAB_UID}" \
    "${WORKSPACE_ROOT}/users/${A3_COLLAB_USER}" \
    "${A3_GROUP_NAME}"

enable_public_key_account "${A3_ADMIN_USER}"
enable_public_key_account "${A3_COLLAB_USER}"

usermod --groups "${A3_GROUP_NAME},sudo" "${A3_ADMIN_USER}"
usermod --groups "${A3_GROUP_NAME}" "${A3_COLLAB_USER}"

[[ -d "${WORKSPACE_ROOT}" && ! -L "${WORKSPACE_ROOT}" ]] \
    || fail "workspace mount must be a real directory: ${WORKSPACE_ROOT}"
setfacl -m \
    "u:${A3_ADMIN_USER}:rwx,u:${A3_COLLAB_USER}:r-x,m::rwx" \
    "${WORKSPACE_ROOT}"

printf '%s ALL=(ALL:ALL) NOPASSWD: ALL\n' "${A3_ADMIN_USER}" \
    >"/etc/sudoers.d/a3-admin"
chmod 0440 /etc/sudoers.d/a3-admin
visudo --check --file /etc/sudoers.d/a3-admin >/dev/null

install_authorized_keys \
    "${A3_ADMIN_USER}" \
    "${CONFIG_ROOT}/admin_authorized_keys"
install_authorized_keys \
    "${A3_COLLAB_USER}" \
    "${CONFIG_ROOT}/collaborator_authorized_keys"
install_shell_startup "${A3_ADMIN_USER}"
install_shell_startup "${A3_COLLAB_USER}"

install -d -m 2750 -o "${A3_ADMIN_USER}" -g "${A3_GROUP_NAME}" \
    "${WORKSPACE_ROOT}/projects" \
    "${WORKSPACE_ROOT}/projects/a3-outcome-stack"

for user in "${A3_ADMIN_USER}" "${A3_COLLAB_USER}"; do
    install -d -m 2750 -o "${user}" -g "${A3_GROUP_NAME}" \
        "${WORKSPACE_ROOT}/a3/staging/${user}" \
        "${WORKSPACE_ROOT}/a3/runs/${user}"
    set_read_acl "${WORKSPACE_ROOT}/a3/staging/${user}"
    set_read_acl "${WORKSPACE_ROOT}/a3/runs/${user}"
    install -d -m 0750 -o "${user}" -g "${A3_GROUP_NAME}" \
        "${WORKSPACE_ROOT}/users/${user}/src"
done

install -d -m 2750 -o root -g "${A3_GROUP_NAME}" \
    "${WORKSPACE_ROOT}/a3/releases" \
    "${WORKSPACE_ROOT}/a3/releases/datasets" \
    "${WORKSPACE_ROOT}/a3/releases/models"
set_read_acl "${WORKSPACE_ROOT}/a3/releases"
set_read_acl "${WORKSPACE_ROOT}/a3/releases/datasets"
set_read_acl "${WORKSPACE_ROOT}/a3/releases/models"

install -d -m 2770 -o "${A3_ADMIN_USER}" -g "${A3_GROUP_NAME}" \
    "${WORKSPACE_ROOT}/a3/cache" \
    "${WORKSPACE_ROOT}/a3/cache/huggingface" \
    "${WORKSPACE_ROOT}/a3/cache/torch" \
    "${WORKSPACE_ROOT}/a3/locks"
set_write_acl "${WORKSPACE_ROOT}/a3/cache"
set_write_acl "${WORKSPACE_ROOT}/a3/cache/huggingface"
set_write_acl "${WORKSPACE_ROOT}/a3/cache/torch"
set_write_acl "${WORKSPACE_ROOT}/a3/locks"

install -d -m 0700 -o "${A3_ADMIN_USER}" -g "${A3_ADMIN_USER}" \
    "${WORKSPACE_ROOT}/a3/admin"

install -d -m 0755 /etc/a3
/opt/a3/.venv/bin/python - "${RUNTIME_CONFIG}" <<'PY'
import json
import os
import pathlib
import sys

output = pathlib.Path(sys.argv[1])
payload = {
    "schema_version": 1,
    "workspace_root": "/workspace",
    "admin_user": os.environ["A3_ADMIN_USER"],
    "collaborator_user": os.environ["A3_COLLAB_USER"],
    "group_name": os.environ["A3_GROUP_NAME"],
    "image_digest": os.environ["A3_IMAGE_DIGEST"],
}
output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
output.chmod(0o644)
PY

install -d -m 0700 "${SSH_HOST_KEY_ROOT}"
if [[ ! -s "${SSH_HOST_KEY_ROOT}/ssh_host_ed25519_key" ]]; then
    ssh-keygen -q -t ed25519 -N '' -f "${SSH_HOST_KEY_ROOT}/ssh_host_ed25519_key"
fi
if [[ ! -s "${SSH_HOST_KEY_ROOT}/ssh_host_rsa_key" ]]; then
    ssh-keygen -q -t rsa -b 3072 -N '' -f "${SSH_HOST_KEY_ROOT}/ssh_host_rsa_key"
fi
chmod 0600 "${SSH_HOST_KEY_ROOT}"/ssh_host_*_key
chmod 0644 "${SSH_HOST_KEY_ROOT}"/ssh_host_*_key.pub

cp /etc/ssh/sshd_config.a3 /run/sshd_config
printf '\nAllowUsers %s %s\n' "${A3_ADMIN_USER}" "${A3_COLLAB_USER}" >>/run/sshd_config
mkdir -p /run/sshd
/usr/sbin/sshd -t -f /run/sshd_config

exec /usr/sbin/sshd -D -e -f /run/sshd_config
