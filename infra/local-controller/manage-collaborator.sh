#!/usr/bin/env bash
set -euo pipefail
export LC_ALL=C

collab_group=a3-collab
operator_group=a3-operator
archive_root=/var/lib/a3-outcome-stack/admin/revocations

fail() {
    echo "error: $*" >&2
    exit 1
}

[[ "${EUID}" -eq 0 ]] || fail "run as root"
action=${1:-}
account=${2:-}
[[ "${account}" =~ ^[a-z_][a-z0-9_-]*$ ]] || fail "invalid account name"

case "${action}" in
    provision)
        key_file=${3:-}
        operator=no
        [[ $# -le 4 ]] || fail "too many provision arguments"
        [[ -z "${4:-}" || "${4:-}" == "--operator" ]] || \
            fail "the only supported provision option is --operator"
        [[ "${4:-}" == "--operator" ]] && operator=yes
        [[ "${A3_TAILNET_GRANT_CONFIRMED:-}" == "YES" ]] || \
            fail "confirm the individual tailnet Grant first"
        [[ -f "${key_file}" && -s "${key_file}" ]] || fail "missing public-key file"
        ssh-keygen -l -f "${key_file}" >/dev/null || fail "invalid public-key file"
        getent group "${collab_group}" >/dev/null || fail "collaborator group is absent"
        getent group "${operator_group}" >/dev/null || fail "operator group is absent"
        ! id "${account}" >/dev/null 2>&1 || fail "account already exists"
        adduser --disabled-password --gecos "" "${account}"
        random_password=$(openssl rand -base64 48)
        password_hash=$(openssl passwd -6 "${random_password}")
        unset random_password
        usermod --password "${password_hash}" "${account}"
        unset password_hash
        groups=${collab_group}
        if [[ "${operator}" == "yes" ]]; then
            groups+="${groups:+,}${operator_group}"
        fi
        usermod --shell /bin/bash --groups "${groups}" "${account}"
        home=$(getent passwd "${account}" | cut -d: -f6)
        [[ -n "${home}" && "${home}" != "/" ]] || fail "unsafe home directory"
        install -d -m 0700 -o "${account}" -g "${account}" "${home}/.ssh"
        install -m 0600 -o "${account}" -g "${account}" \
            "${key_file}" "${home}/.ssh/authorized_keys"
        privileged=$(id -nG "${account}" | tr ' ' '\n' | \
            grep -E '^(sudo|adm|lxd|docker|disk|dialout|input|video|render|plugdev)$' || true)
        [[ -z "${privileged}" ]] || fail "account retained a privileged group"
        ;;
    revoke)
        [[ "${A3_TAILNET_GRANT_REVOKED:-}" == "YES" ]] || \
            fail "remove and confirm the individual tailnet Grant first"
        id "${account}" >/dev/null 2>&1 || fail "account does not exist"
        home=$(getent passwd "${account}" | cut -d: -f6)
        [[ -n "${home}" && "${home}" != "/" ]] || fail "unsafe home directory"
        timestamp=$(date -u +%Y%m%dT%H%M%SZ)
        archive="${archive_root}/${timestamp}"
        install -d -m 0700 -o root -g root "${archive}"
        if [[ -f "${home}/.ssh/authorized_keys" ]]; then
            install -m 0600 -o root -g root \
                "${home}/.ssh/authorized_keys" "${archive}/authorized_keys"
            : >"${home}/.ssh/authorized_keys"
            chown "${account}:${account}" "${home}/.ssh/authorized_keys"
            chmod 0600 "${home}/.ssh/authorized_keys"
        fi
        usermod --groups "" --lock --shell /usr/sbin/nologin "${account}"
        ;;
    *)
        fail "usage: $0 provision <account> <public-key-file> [--operator] | revoke <account>"
        ;;
esac
