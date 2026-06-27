#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

export NSYS_PROFILE_SCOPE="${NSYS_PROFILE_SCOPE:-request_only}"
export NSYS_DELAY="${NSYS_DELAY:-120}"
export NSYS_DURATION="${NSYS_DURATION:-45}"

exec bash backend_compare/scripts/start_vllm_qwen35_nsys.sh "$@"
