#!/bin/bash

set -euo pipefail

ROOT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
BUILD_DIR="$ROOT_DIR/build/xpu"
OUTPUT_PATH="$ROOT_DIR/comfy_aimdo/aimdo_xpu.so"
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

COMMON_SOURCES=(
    control.c
    debug.c
    hostbuf-decommit.c
    hostbuf-file-reader.c
    hostbuf-prewarm.c
    hostbuf.c
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
