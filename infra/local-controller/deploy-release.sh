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

git_source() {
    git -c "safe.directory=${source_root}" -C "${source_root}" "$@"
}

preinstall_registry_from_mirror() {
    local project=$1
    local mirror=${A3_PYPI_MIRROR:-}
    [[ -n "${mirror}" ]] || return 0
    [[ "${mirror}" =~ ^https://[^[:space:]]+$ ]] || \
        fail "A3_PYPI_MIRROR must be an HTTPS package index"
    local requirements=${project}/.a3-mirror-requirements.txt
    UV_PYTHON_INSTALL_DIR=/opt/a3/python \
        uv export --project "${project}" --frozen --extra local-controller --no-dev \
        --no-emit-project --no-emit-package el-a3-sdk --no-emit-package lerobot \
        --no-emit-package torch --no-emit-package torchvision \
        --format requirements-txt --output-file "${requirements}"
    UV_PYTHON_INSTALL_DIR=/opt/a3/python \
        uv venv --python 3.12.13 "${project}/.venv"
    UV_DEFAULT_INDEX="${mirror}" UV_CONCURRENT_DOWNLOADS=8 \
        UV_HTTP_TIMEOUT=600 UV_HTTP_RETRIES=10 \
        uv pip sync --python "${project}/.venv/bin/python" \
        --require-hashes "${requirements}"
    rm -f -- "${requirements}"
}

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
        git_source diff --quiet || fail "source worktree is dirty"
        git_source diff --cached --quiet || fail "source index is dirty"
        [[ -z "$(git_source status --porcelain --untracked-files=all)" ]] || \
            fail "source checkout contains untracked files"
        commit=$(git_source rev-parse HEAD)
        [[ "${commit}" =~ ^[0-9a-f]{40}$ ]] || fail "source commit is invalid"
        destination=${release_root}/${commit}
        [[ ! -e "${destination}" && ! -L "${destination}" ]] || \
            fail "release already exists; releases are immutable"
        temporary=${release_root}/.${commit}.installing
        [[ ! -e "${temporary}" ]] || fail "stale release installation exists"
        install -d -m 0750 -o root -g "${collab_group}" "${temporary}"
        git_source archive --format=tar "${commit}" | \
            tar -xf - -C "${temporary}"
        preinstall_registry_from_mirror "${temporary}"
        UV_PYTHON_INSTALL_DIR=/opt/a3/python \
            uv sync --project "${temporary}" --frozen --extra local-controller --no-dev \
            --no-editable
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
