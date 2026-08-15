#!/bin/bash

set -euo pipefail

ROOT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
BUILD_DIR="$ROOT_DIR/build/xpu-tests"
CXX=${CXX:-icpx}
UR_INCLUDE_DIR=${UR_INCLUDE_DIR:-}

if [ -z "$UR_INCLUDE_DIR" ]; then
    CXX_PATH=$(command -v "$CXX")
    UR_INCLUDE_DIR=$(CDPATH= cd -- "$(dirname -- "$CXX_PATH")/../include" && pwd)
fi
if [ ! -f "$UR_INCLUDE_DIR/ur_api.h" ]; then
    echo "Unified Runtime headers were not found in $UR_INCLUDE_DIR" >&2
    exit 1
fi

mkdir -p "$BUILD_DIR"

"$CXX" -std=c++17 -O2 -g -Wall -Wextra -Werror -pthread \
    "$ROOT_DIR/tests/ur_usm_hook_unit.cpp" -I"$UR_INCLUDE_DIR" \
    -ldl -o "$BUILD_DIR/ur_usm_hook_unit"
"$BUILD_DIR/ur_usm_hook_unit"

echo "passed UR hook unit tests"
