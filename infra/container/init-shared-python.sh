#!/usr/bin/env bash
set -euo pipefail

readonly A3_ROOT="/workspace/a3"
readonly SHARED_ENV="${A3_ROOT}/python-env"
readonly HISTORY_ROOT="${A3_ROOT}/python-env-history"
readonly COMMAND_ROOT="${A3_ROOT}/bin"
readonly LIB_ROOT="${A3_ROOT}/lib"
readonly PROFILE_TARGET="${A3_ROOT}/profile.sh"
readonly SEED_ENV="${A3_SEED_PYTHON_ENV:-/opt/a3/.venv}"
readonly SOURCE_BIN="${A3_CONTAINER_BIN_SOURCE:-/usr/local/bin}"
readonly SOURCE_LIB="${A3_CONTAINER_LIB_SOURCE:-/usr/local/lib/a3-container}"
readonly PROFILE_SOURCE="${A3_PROFILE_SOURCE:-/etc/profile.d/a3.sh}"

fail() {
    echo "a3-init-shared-python: $*" >&2
    exit 2
}

[[ "$(id -u)" == "0" ]] || fail "must run as root"
: "${A3_ADMIN_USER:?set A3_ADMIN_USER}"
: "${A3_GROUP_NAME:?set A3_GROUP_NAME}"
id "${A3_ADMIN_USER}" >/dev/null 2>&1 || fail "administrator account does not exist"
getent group "${A3_GROUP_NAME}" >/dev/null || fail "project group does not exist"
[[ -x "${SEED_ENV}/bin/python" ]] || fail "seed Python environment is unavailable"
[[ -f "${SOURCE_LIB}/a3_container_common.py" ]] || fail "container helper library is missing"
[[ -f "${PROFILE_SOURCE}" ]] || fail "shared profile source is missing"

for command in \
    a3-artifact-fetch \
    a3-artifact-promote \
    a3-env-doctor \
    a3-gpu-run \
    a3-python
do
    [[ -f "${SOURCE_BIN}/${command}" ]] || fail "command source is missing: ${command}"
done

install -d -m 0750 -o "${A3_ADMIN_USER}" -g "${A3_GROUP_NAME}" "${A3_ROOT}"

if [[ ! -e "${SHARED_ENV}" ]]; then
    temporary="${A3_ROOT}/.python-env.init.$$"
    [[ ! -e "${temporary}" ]] || fail "temporary initialization path already exists"
    cleanup() {
        if [[ -d "${temporary}" && "${temporary}" == "${A3_ROOT}/.python-env.init.$$" ]]; then
            rm -rf --one-file-system "${temporary}"
        fi
    }
    trap cleanup EXIT

    install -d -m 0750 "${temporary}"
    rsync -a "${SEED_ENV}/" "${temporary}/"
    "${temporary}/bin/python" -m ensurepip --upgrade
    "${temporary}/bin/python" - "${temporary}" "${SHARED_ENV}" "${SEED_ENV}" <<'PY'
from pathlib import Path
import sys

temporary = sys.argv[1]
destination = sys.argv[2]
seed = sys.argv[3]
bin_root = Path(temporary) / "bin"
for path in bin_root.iterdir():
    if path.is_symlink() or not path.is_file():
        continue
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        continue
    updated = text.replace(temporary, destination).replace(seed, destination)
    if updated != text:
        path.write_text(updated, encoding="utf-8")
PY
    printf 'a3-shared-python-v1\n' >"${temporary}/.a3-shared-environment"
    mv "${temporary}" "${SHARED_ENV}"
    trap - EXIT
fi

[[ -x "${SHARED_ENV}/bin/python" ]] || fail "shared Python interpreter is missing"
if ! "${SHARED_ENV}/bin/python" -m pip --version >/dev/null 2>&1; then
    "${SHARED_ENV}/bin/python" -m ensurepip --upgrade
fi

install -d -m 0750 -o "${A3_ADMIN_USER}" -g "${A3_GROUP_NAME}" \
    "${HISTORY_ROOT}" \
    "${HISTORY_ROOT}/snapshots" \
    "${COMMAND_ROOT}" \
    "${LIB_ROOT}"
install -m 0750 -o "${A3_ADMIN_USER}" -g "${A3_GROUP_NAME}" \
    "${SOURCE_BIN}/a3-artifact-fetch" \
    "${SOURCE_BIN}/a3-artifact-promote" \
    "${SOURCE_BIN}/a3-env-doctor" \
    "${SOURCE_BIN}/a3-gpu-run" \
    "${SOURCE_BIN}/a3-python" \
    "${COMMAND_ROOT}/"
install -m 0640 -o "${A3_ADMIN_USER}" -g "${A3_GROUP_NAME}" \
    "${SOURCE_LIB}/a3_container_common.py" \
    "${LIB_ROOT}/a3_container_common.py"
install -m 0640 -o "${A3_ADMIN_USER}" -g "${A3_GROUP_NAME}" \
    "${PROFILE_SOURCE}" \
    "${PROFILE_TARGET}"

chown -R "${A3_ADMIN_USER}:${A3_GROUP_NAME}" \
    "${SHARED_ENV}" \
    "${HISTORY_ROOT}" \
    "${COMMAND_ROOT}" \
    "${LIB_ROOT}" \
    "${PROFILE_TARGET}"
chmod -R u+rwX,g+rX,g-w,o-rwx "${SHARED_ENV}"
chmod 0750 "${HISTORY_ROOT}" "${HISTORY_ROOT}/snapshots" "${COMMAND_ROOT}" "${LIB_ROOT}"
chmod 0640 "${PROFILE_TARGET}" "${LIB_ROOT}/a3_container_common.py"

"${SHARED_ENV}/bin/python" -m pip --version
"${SHARED_ENV}/bin/python" -c "import torch; print(torch.__version__)"
