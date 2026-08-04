#!/usr/bin/env bash
set -euo pipefail
export LC_ALL=C

deployment_root=/opt/a3-outcome-stack
release_root=${deployment_root}/releases
collab_group=a3-collab

fail() {
    echo "error: $*" >&2
    exit 1
}

[[ "${EUID}" -eq 0 ]] || fail "run as root"
action=${1:-}

activate_release() {
    local commit=${1:-}
    [[ "${commit}" =~ ^[0-9a-f]{40}$ ]] || fail "release must be a full Git commit"
    local release=${release_root}/${commit}
    [[ -d "${release}" && ! -L "${release}" ]] || fail "release does not exist"
    local temporary_link=${deployment_root}/.current.${commit}
    [[ ! -e "${temporary_link}" && ! -L "${temporary_link}" ]] || \
        fail "temporary activation link already exists"
    ln -s "releases/${commit}" "${temporary_link}"
    mv -Tf -- "${temporary_link}" "${deployment_root}/current"
}

case "${action}" in
    install)
        source_root=${2:-}
        [[ -d "${source_root}/.git" ]] || fail "source must be a Git checkout"
        git -C "${source_root}" diff --quiet || fail "source worktree is dirty"
        git -C "${source_root}" diff --cached --quiet || fail "source index is dirty"
        [[ -z "$(git -C "${source_root}" status --porcelain --untracked-files=all)" ]] || \
            fail "source checkout contains untracked files"
        commit=$(git -C "${source_root}" rev-parse HEAD)
        [[ "${commit}" =~ ^[0-9a-f]{40}$ ]] || fail "source commit is invalid"
        destination=${release_root}/${commit}
        [[ ! -e "${destination}" && ! -L "${destination}" ]] || \
            fail "release already exists; releases are immutable"
        temporary=${release_root}/.${commit}.installing
        [[ ! -e "${temporary}" ]] || fail "stale release installation exists"
        install -d -m 0750 -o root -g "${collab_group}" "${temporary}"
        git -C "${source_root}" archive --format=tar "${commit}" | \
            tar -xf - -C "${temporary}"
        UV_PYTHON_INSTALL_DIR=/opt/a3/python \
            uv sync --project "${temporary}" --frozen --extra local-controller --no-dev
        chown -R root:"${collab_group}" "${temporary}"
        chmod -R u=rwX,g=rX,o= "${temporary}"
        mv -- "${temporary}" "${destination}"
        activate_release "${commit}"
        ;;
    activate)
        activate_release "${2:-}"
        ;;
    *)
        fail "usage: $0 install <clean-source-checkout> | activate <full-commit>"
        ;;
esac
