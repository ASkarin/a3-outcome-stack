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
for role in collaborator operator hardware_service; do
    case "${role}" in
        collaborator) group=a3-collab ;;
        operator) group=a3-operator ;;
        hardware_service) group=a3-hardware ;;
    esac
    present=false
    getent group "${group}" >/dev/null && present=true || status=incomplete
    groups_json=$(jq -c --arg role "${role}" --argjson present "${present}" \
        '. + {($role): $present}' <<<"${groups_json}")
done

deployment_exists=false
[[ -d /opt/a3-outcome-stack ]] && deployment_exists=true || status=incomplete
service_enabled=false
systemctl is-enabled a3-local-control.service >/dev/null 2>&1 && service_enabled=true || true
service_active=false
systemctl is-active a3-local-control.service >/dev/null 2>&1 && service_active=true || true
service_installed=false
[[ -f /etc/systemd/system/a3-local-control.service ]] && service_installed=true || status=incomplete
[[ "${service_enabled}" == "false" && "${service_active}" == "false" ]] || status=incomplete

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
if ufw status 2>/dev/null | grep -q '^Status: active'; then
    ufw_added=$(ufw show added 2>/dev/null || true)
    unexpected_ufw=$(grep -E '^ufw allow' <<<"${ufw_added}" | \
        grep -Ev '^ufw allow in on tailscale0 proto tcp to any port [0-9]+( comment .*)?$' || true)
    if grep -Fqx 'ufw default deny incoming' <<<"${ufw_added}" && \
        grep -Fqx 'ufw default allow outgoing' <<<"${ufw_added}" && \
        grep -Eq '^ufw allow in on tailscale0 proto tcp to any port [0-9]+' <<<"${ufw_added}" && \
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
print(
    json.dumps(
        {
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
        },
        sort_keys=True,
    )
)
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

jq -n \
    --arg schema_version "a3-local-environment-v1" \
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
    --argjson service_enabled "${service_enabled}" \
    --argjson service_active "${service_active}" \
    --argjson service_installed "${service_installed}" \
    --argjson sshd_policy_ok "${sshd_policy_ok}" \
    --argjson ufw_policy_ok "${ufw_policy_ok}" \
    --argjson torch "${torch_json}" \
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
            role_groups_configured: $groups,
            deployment_root_exists: $deployment_exists,
            raw_hardware_access_granted_to_humans: false
        },
        control_service: {
            installed: $service_installed,
            enabled: $service_enabled,
            active: $service_active,
            network_listener_configured: false
        },
        hardware_available: false,
        hardware_tests_executed: false,
        motor_enable_executed: false,
        real_can_traffic_executed: false,
        hardware_verified: false
    }'
