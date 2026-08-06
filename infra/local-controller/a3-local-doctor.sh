#!/usr/bin/env bash
set -euo pipefail
export LC_ALL=C

status=pass
hostname_ok=false
[[ "$(hostname)" == "a3-local" ]] && hostname_ok=true || status=incomplete

driver_version=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -n1 || true)
gpu_name=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -n1 || true)
uv_version=$(uv --version 2>/dev/null | awk '{print $2}' || true)
python_executable=$(find /opt/a3/python -type f -path '*/bin/python3.12' -print -quit 2>/dev/null || true)
python_version=
if [[ -n "${python_executable}" ]]; then
    python_version=$("${python_executable}" -c 'import platform; print(platform.python_version())')
fi
gpu_ok=false
[[ -n "${gpu_name}" && "${driver_version}" == 595.* ]] && gpu_ok=true || status=incomplete
uv_ok=false
[[ "${uv_version}" == "0.11.32" ]] && uv_ok=true || status=incomplete
python_ok=false
[[ "${python_version}" == "3.12.13" ]] && python_ok=true || status=incomplete

groups_json='{}'
for role in collaborator legacy_operator legacy_hardware; do
    case "${role}" in
        collaborator) group=a3-collab ;;
        legacy_operator) group=a3-operator ;;
        legacy_hardware) group=a3-hardware ;;
    esac
    present=false
    getent group "${group}" >/dev/null && present=true || true
    groups_json=$(jq -c --arg role "${role}" --argjson present "${present}" \
        '. + {($role): $present}' <<<"${groups_json}")
done

deployment_exists=false
[[ -d /opt/a3-outcome-stack ]] && deployment_exists=true || status=incomplete
legacy_service_enabled=false
systemctl is-enabled a3-local-control.service >/dev/null 2>&1 && \
    legacy_service_enabled=true || true
legacy_service_active=false
systemctl is-active a3-local-control.service >/dev/null 2>&1 && \
    legacy_service_active=true || true
legacy_service_installed=false
[[ -f /etc/systemd/system/a3-local-control.service ]] && legacy_service_installed=true || true
[[ "${legacy_service_enabled}" == "false" && "${legacy_service_active}" == "false" ]] || \
    status=incomplete

sshd_policy_ok=false
if sshd_effective=$(/usr/sbin/sshd -T 2>/dev/null); then
    sshd_policy_ok=true
    for expected in \
        "permitrootlogin no" \
        "passwordauthentication no" \
        "kbdinteractiveauthentication no" \
        "authenticationmethods publickey" \
        "allowagentforwarding yes" \
        "allowtcpforwarding no" \
        "allowstreamlocalforwarding no" \
        "permittunnel no"; do
        grep -Fqx "${expected}" <<<"${sshd_effective}" || sshd_policy_ok=false
    done
fi
[[ "${sshd_policy_ok}" == "true" ]] || status=incomplete

ufw_policy_ok=false
ufw_status=$(ufw status verbose 2>/dev/null || true)
if grep -q '^Status: active' <<<"${ufw_status}" && \
    grep -q '^Default: deny (incoming), allow (outgoing)' <<<"${ufw_status}"; then
    ufw_added=$(ufw show added 2>/dev/null || true)
    unexpected_ufw=$(grep -E '^ufw allow' <<<"${ufw_added}" | \
        grep -Ev '^ufw allow in on tailscale0 to any port [0-9]+ proto tcp( comment .*)?$' || true)
    if grep -Eq '^ufw allow in on tailscale0 to any port [0-9]+ proto tcp' <<<"${ufw_added}" && \
        [[ -z "${unexpected_ufw}" ]]; then
        ufw_policy_ok=true
    fi
fi
[[ "${ufw_policy_ok}" == "true" ]] || status=incomplete

torch_json='{"installed":false,"version":"","build_cuda":"","cuda_available":false,"device_name":"","target_satisfied":false}'
deployment_python=/opt/a3-outcome-stack/current/.venv/bin/python
if [[ -x "${deployment_python}" ]]; then
    if torch_json=$("${deployment_python}" - <<'A3_LOCAL_TORCH_PROBE'
import json
import torch

cuda_available = torch.cuda.is_available()
device_name = torch.cuda.get_device_name(0) if cuda_available else ""
version = torch.__version__
build_cuda = torch.version.cuda or ""
print(json.dumps({
    "installed": True,
    "version": version,
    "build_cuda": build_cuda,
    "cuda_available": cuda_available,
    "device_name": device_name,
    "target_satisfied": (
        version == "2.11.0+cu128"
        and build_cuda == "12.8"
        and cuda_available
        and "RTX 3060" in device_name
    ),
}, sort_keys=True))
A3_LOCAL_TORCH_PROBE
    ); then
        [[ "$(jq -r '.target_satisfied' <<<"${torch_json}")" == "true" ]] || \
            status=incomplete
    else
        status=incomplete
        torch_json='{"installed":true,"probe_failed":true,"target_satisfied":false}'
    fi
else
    status=incomplete
fi

not_checked_access='{"can":{"status":"not_checked","devices":[]},"d435":{"status":"not_checked","devices":[]},"ar0234":{"status":"not_checked","devices":[]},"xbox":{"status":"not_checked","devices":[]}}'

probe_access_as() {
    local account=$1
    [[ -n "${account}" && "${account}" != "root" ]] || {
        printf '%s\n' "${not_checked_access}"
        return
    }
    id "${account}" >/dev/null 2>&1 || {
        printf '%s\n' "${not_checked_access}"
        return
    }
    [[ -x "${deployment_python}" ]] || {
        printf '%s\n' "${not_checked_access}"
        return
    }
    sudo -u "${account}" env A3_AR0234_DEVICE="${A3_AR0234_DEVICE:-}" \
        "${deployment_python}" - <<'A3_DEVICE_ACCESS_PROBE'
import json
import os
import socket
from pathlib import Path

def ordinary(paths):
    if not paths:
        return {"status": "not_checked", "devices": []}
    devices = [
        {"path": str(path), "readable": os.access(path, os.R_OK), "writable": os.access(path, os.W_OK)}
        for path in paths
    ]
    return {
        "status": "pass" if all(item["readable"] and item["writable"] for item in devices) else "fail",
        "devices": devices,
    }

interfaces = sorted(path.name for path in Path("/sys/class/net").glob("can*"))
can_devices = []
for interface in interfaces:
    opened = False
    error = None
    try:
        handle = socket.socket(socket.AF_CAN, socket.SOCK_RAW, socket.CAN_RAW)
        try:
            handle.bind((interface,))
            opened = True
        finally:
            handle.close()
    except OSError as exc:
        error = str(exc)
    can_devices.append({"interface": interface, "raw_socket_opened": opened, "error": error})
can_access = {
    "status": "not_checked" if not can_devices else ("pass" if all(item["raw_socket_opened"] for item in can_devices) else "fail"),
    "devices": can_devices,
}

video_root = Path("/dev/v4l/by-id")
videos = sorted(video_root.iterdir()) if video_root.is_dir() else []
d435 = [path for path in videos if "realsense" in path.name.lower() or "d435" in path.name.lower()]
ar0234 = [path for path in videos if "ar0234" in path.name.lower()]
if os.environ.get("A3_AR0234_DEVICE"):
    ar0234.append(Path(os.environ["A3_AR0234_DEVICE"]))
input_root = Path("/dev/input/by-id")
xbox = sorted(input_root.glob("*-event-joystick")) if input_root.is_dir() else []
print(json.dumps({"can": can_access, "d435": ordinary(d435), "ar0234": ordinary(ar0234), "xbox": ordinary(xbox)}, sort_keys=True))
A3_DEVICE_ACCESS_PROBE
}

administrator=${SUDO_USER:-}
administrator_identified=false
[[ -n "${administrator}" && "${administrator}" != "root" ]] && \
    administrator_identified=true || true
administrator_access=$(probe_access_as "${administrator}")
collaborator=${A3_COLLABORATOR_ACCOUNT:-}
collaborator_identified=false
[[ -n "${collaborator}" && "${collaborator}" != "root" ]] && \
    collaborator_identified=true || true
collaborator_access=$(probe_access_as "${collaborator}")

jq -n \
    --arg schema_version "a3-local-environment-v2" \
    --arg generated_at_utc "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    --arg status "${status}" \
    --argjson hostname_ok "${hostname_ok}" \
    --arg gpu_name "${gpu_name}" \
    --arg driver_version "${driver_version}" \
    --arg uv_version "${uv_version}" \
    --arg python_version "${python_version}" \
    --argjson gpu_ok "${gpu_ok}" \
    --argjson uv_ok "${uv_ok}" \
    --argjson python_ok "${python_ok}" \
    --argjson groups "${groups_json}" \
    --argjson deployment_exists "${deployment_exists}" \
    --argjson legacy_service_enabled "${legacy_service_enabled}" \
    --argjson legacy_service_active "${legacy_service_active}" \
    --argjson legacy_service_installed "${legacy_service_installed}" \
    --argjson sshd_policy_ok "${sshd_policy_ok}" \
    --argjson ufw_policy_ok "${ufw_policy_ok}" \
    --argjson torch "${torch_json}" \
    --argjson administrator_identified "${administrator_identified}" \
    --argjson administrator_access "${administrator_access}" \
    --argjson collaborator_identified "${collaborator_identified}" \
    --argjson collaborator_access "${collaborator_access}" \
    '{
        schema_version: $schema_version,
        generated_at_utc: $generated_at_utc,
        status: $status,
        host_identity: {hostname_matches_alias: $hostname_ok},
        gpu: {name: $gpu_name, driver_version: $driver_version, target_satisfied: $gpu_ok},
        runtime: {
            uv_version: $uv_version,
            uv_target_satisfied: $uv_ok,
            python_version: $python_version,
            python_target_satisfied: $python_ok,
            pytorch: $torch
        },
        network_security: {
            openssh_target_satisfied: $sshd_policy_ok,
            ufw_target_satisfied: $ufw_policy_ok
        },
        permissions: {
            unique_administrator: {
                account_identified: $administrator_identified,
                highest_privilege: true,
                raw_hardware_authorized: true,
                enumerated_device_access: $administrator_access
            },
            collaborator: {
                account_identified: $collaborator_identified,
                sudo_authorized: false,
                raw_hardware_authorized: false,
                enumerated_device_access: $collaborator_access
            },
            groups_configured: $groups,
            deployment_root_exists: $deployment_exists
        },
        legacy_control_service: {
            installed: $legacy_service_installed,
            enabled: $legacy_service_enabled,
            active: $legacy_service_active,
            expected_state: "disabled_inactive_pending_authorized_removal"
        },
        hardware_available: false,
        hardware_tests_executed: false,
        motor_enable_executed: false,
        real_can_traffic_executed: false,
        hardware_verified: false
    }'
