#!/usr/bin/env bash
set -euo pipefail

# The pinned CUDA development base alone is about 10 GiB. Standard GitHub-hosted
# runners contain optional SDKs that are unrelated to this Linux container build.
# Remove only this explicit allowlist from the disposable runner.
readonly TARGETS=(
    /usr/local/lib/android
    /usr/share/dotnet
    /opt/ghc
    /usr/local/.ghcup
    /opt/hostedtoolcache/CodeQL
)

for target in "${TARGETS[@]}"; do
    case "${target}" in
        /usr/local/lib/android | \
        /usr/share/dotnet | \
        /opt/ghc | \
        /usr/local/.ghcup | \
        /opt/hostedtoolcache/CodeQL)
            ;;
        *)
            echo "refusing unexpected cleanup target: ${target}" >&2
            exit 2
            ;;
    esac
    if [[ -e "${target}" ]]; then
        sudo rm -rf -- "${target}"
    fi
done

df -h /
