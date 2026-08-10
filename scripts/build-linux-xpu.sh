#!/bin/bash

set -euo pipefail

ROOT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
BUILD_DIR="$ROOT_DIR/build/xpu"
OUTPUT_PATH="$ROOT_DIR/comfy_aimdo/aimdo_xpu.so"
CC=${CC:-gcc}
CXX=${CXX:-icpx}

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

"$CXX" -shared -o "$OUTPUT_PATH" -fsycl -pthread \
    "${OBJECTS[@]}" -lze_loader -ldl

echo "built $OUTPUT_PATH"
