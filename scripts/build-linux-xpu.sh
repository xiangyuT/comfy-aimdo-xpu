#!/bin/bash

set -euo pipefail

ROOT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
BUILD_DIR="${AIMDO_XPU_BUILD_DIR:-$ROOT_DIR/build/xpu}"
OUTPUT_PATH="${AIMDO_XPU_OUTPUT_PATH:-$ROOT_DIR/comfy_aimdo/aimdo_xpu.so}"
CC=${CC:-gcc}
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

# Source identity is independent of owner-controlled distribution versions.
# Hash tracked and new source inputs; the build tree is ignored by Git.
python3 "$ROOT_DIR/scripts/write-source-identity.py" "$ROOT_DIR" "$BUILD_DIR/xpu-source-identity.h"

COMMON_SOURCES=(
    control.c
    debug.c
    hostbuf-decommit.c
    hostbuf-file-reader.c
    hostbuf-prewarm.c
    hostbuf.c
    malloc-graph.c
    malloc-rogue.c
    vmm-ref.c
    model-vbar.c
    pyt-cu-plug-alloc.c
    pyt-cu-plug-alloc-async.c
    vrambuf.c
    xfer-file.c
)
POSIX_SOURCES=(
    hostbuf-plat.c
    model-mmap.c
    thread-plat.c
    xfer-file-plat.c
)
OBJECTS=()

for source in "${COMMON_SOURCES[@]}"; do
    object="$BUILD_DIR/${source%.c}.o"
    "$CC" -c -o "$object" -fPIC -O2 -g -pthread -DAIMDO_XPU \
        ${AIMDO_EXTRA_CFLAGS:-} \
        -include "$BUILD_DIR/xpu-source-identity.h" \
        "$ROOT_DIR/src/$source" -I"$ROOT_DIR/src"
    OBJECTS+=("$object")
done

for source in "${POSIX_SOURCES[@]}"; do
    object="$BUILD_DIR/posix-${source%.c}.o"
    "$CC" -c -o "$object" -fPIC -O2 -g -pthread -DAIMDO_XPU \
        ${AIMDO_EXTRA_CFLAGS:-} \
        "$ROOT_DIR/src-posix/$source" -I"$ROOT_DIR/src"
    OBJECTS+=("$object")
done

"$CC" -c -o "$BUILD_DIR/xpu-stubs.o" -fPIC -O2 -g -pthread -DAIMDO_XPU \
    ${AIMDO_EXTRA_CFLAGS:-} \
    "$ROOT_DIR/src-xpu/stubs.c" -I"$ROOT_DIR/src"
OBJECTS+=("$BUILD_DIR/xpu-stubs.o")

"$CXX" -c -o "$BUILD_DIR/xpu-dispatch.o" -fPIC -O2 -g -std=c++17 -fsycl \
    ${AIMDO_EXTRA_CXXFLAGS:-} \
    "$ROOT_DIR/src-xpu/dispatch.cpp" -I"$ROOT_DIR/src"
OBJECTS+=("$BUILD_DIR/xpu-dispatch.o")

"$CXX" -c -o "$BUILD_DIR/xpu-ur-usm-hook.o" -fPIC -O2 -g -std=c++17 \
    ${AIMDO_EXTRA_CXXFLAGS:-} \
    "$ROOT_DIR/src-xpu/ur-usm-hook.cpp" -I"$UR_INCLUDE_DIR"
OBJECTS+=("$BUILD_DIR/xpu-ur-usm-hook.o")

# ComfyUI may have loaded the official CUDA AIMDO DSO before OmniXPU
# prestartup. Keep same-named lifecycle functions inside this XPU DSO from
# being interposed by that earlier RTLD_GLOBAL object.
"$CXX" -shared -o "$OUTPUT_PATH" -fsycl -pthread \
    "${OBJECTS[@]}" -lze_loader -ldl \
    -Wl,-Bsymbolic-functions \
    -Wl,--version-script="$ROOT_DIR/src-xpu/ur-usm-hook.map"

echo "built $OUTPUT_PATH"

# The Torch-facing owner is a separate, opt-in diagnostic DSO. The normal
# 0.5.5 XPU build and its public compiler capability remain unchanged.
NATIVE_OWNER_DIAGNOSTIC=${AIMDO_XPU_BUILD_NATIVE_OWNER_DIAGNOSTIC:-0}
case "$NATIVE_OWNER_DIAGNOSTIC" in
    0) ;;
    1)
        TORCH_PYTHON=${AIMDO_TORCH_PYTHON:-python3}
        TORCH_ROOT=$("$TORCH_PYTHON" -c '
import pathlib
import torch
print(pathlib.Path(torch.__file__).resolve().parent)
')
        TORCH_CXX11_ABI=$("$TORCH_PYTHON" -c 'import torch; print(int(torch._C._GLIBCXX_USE_CXX11_ABI))')
        TORCH_VERSION_DEFINE=$("$TORCH_PYTHON" -c 'import json, torch; print("-DAIMDO_TORCH_VERSION=" + json.dumps(str(torch.__version__)))')
        TORCH_INCLUDE="$TORCH_ROOT/include"
        TORCH_LIB="$TORCH_ROOT/lib"
        if [ ! -f "$TORCH_INCLUDE/c10/xpu/XPUCachingAllocator.h" ] || \
           [ ! -f "$TORCH_LIB/libc10.so" ] || \
           [ ! -f "$TORCH_LIB/libc10_xpu.so" ] || \
           [ ! -f "$TORCH_LIB/libtorch_xpu.so" ]; then
            echo "Torch XPU headers or required libraries are missing" >&2
            exit 1
        fi
        NATIVE_OWNER_OUTPUT_PATH="${AIMDO_XPU_NATIVE_OWNER_OUTPUT_PATH:-$ROOT_DIR/comfy_aimdo/aimdo_xpu_native_owner.so}"
        mkdir -p "$(dirname -- "$NATIVE_OWNER_OUTPUT_PATH")"
        "$CXX" -std=c++20 -fsycl -shared -fPIC -O2 \
            -D_GLIBCXX_USE_CXX11_ABI="$TORCH_CXX11_ABI" \
            "$TORCH_VERSION_DEFINE" \
            ${AIMDO_EXTRA_CXXFLAGS:-} \
            -I"$TORCH_INCLUDE" \
            "$ROOT_DIR/src-xpu/native-owner-proxy.cpp" \
            -L"$TORCH_LIB" -lc10_xpu -lc10 -ldl \
            -o "$NATIVE_OWNER_OUTPUT_PATH"
        echo "built diagnostic $NATIVE_OWNER_OUTPUT_PATH"
        ;;
    *)
        echo "AIMDO_XPU_BUILD_NATIVE_OWNER_DIAGNOSTIC must be 0 or 1" >&2
        exit 1
        ;;
esac
