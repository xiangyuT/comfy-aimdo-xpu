"""Exercise the production H2D function with CPU-only fault injection.

The shim replaces the device/runtime, not the copy control flow or log sink.
Native SYCL compilation and device execution remain separate checks.
"""

import ctypes
import logging
from pathlib import Path
import shutil
import subprocess

import pytest

from comfy_aimdo import control


SHIM = r'''
#include "copy-diagnostics.h"
#include <atomic>
#include <memory>
#include <stdexcept>
#include <system_error>
#include <vector>

static int mode, attempts, vbar;
namespace sycl {
class category_type : public std::error_category {
 public:
    const char *name() const noexcept override { return "test_sycl"; }
    std::string message(int) const override { return "test error"; }
};
static category_type category;
class exception : public std::runtime_error {
    std::error_code code_;
 public:
    exception(const char *message, int code)
        : std::runtime_error(message), code_(code, category) {}
    std::error_code code() const { return code_; }
};
struct event {
    int attempt;
    void wait_and_throw() {
        if (mode == 0) return;
        if (attempt == 1) {
            if (mode == 6) throw std::runtime_error("initial standard failure");
            if (mode == 7) throw 7;
            throw exception("original copy failure", 1);
        }
        if (mode == 2 || (mode == 9 && attempt == 2)) return;
        if (mode == 3) throw std::runtime_error("retry standard failure");
        if (mode == 4) throw 4;
        throw exception("different retry failure", 77);
    }
};
enum class aspect { ext_intel_free_memory };
namespace ext { namespace intel { namespace info { namespace device {
struct free_memory {};
}}}}
namespace info { namespace device { struct global_mem_size {}; }}
struct device {
    bool has(aspect) const { return true; }
    template <typename T> size_t get_info() const { return 7051460608ULL; }
};
namespace usm { enum class alloc { device, host, shared, unknown }; }
inline usm::alloc get_pointer_type(void *, int) { return usm::alloc::device; }
struct queue {
    event memcpy(void *, const void *, size_t) { return {++attempts}; }
    sycl::device get_device() {
        if (mode == 5) throw exception("diagnostic query failure", 88);
        return {};
    }
    int get_context() { return 0; }
};
}
using CUresult = int;
using CUdeviceptr = uintptr_t;
using CUstream = void *;
constexpr CUresult CUDA_SUCCESS = 0, kCudaErrorUnknown = 999;
enum {
    kSynchronousHostToDeviceCalls, kHostToDeviceBytes,
    kSynchronousHostToDeviceCompletions, kHostToDeviceSplitRetries,
};
static std::atomic<uint64_t> g_stats[4];
static std::shared_ptr<sycl::queue> queue = std::make_shared<sycl::queue>();
static auto resolve_queue(CUstream) { return queue; }
static void trace_sync(const char *, const char *, uint64_t, void *, size_t) {}
static int aimdo_vbar_describe_range(uint64_t, uint64_t, int *mapped,
        unsigned *pin, uint64_t *page, uint64_t *unmapped, uint64_t *span) {
    *mapped = 0; *pin = 0; *page = 558; *unmapped = 558; *span = 2;
    return vbar;
}
'''

EXPORTS = r'''
extern "C" int test_run_h2d(int selected_mode, size_t size, int vbar_hit) {
    mode = selected_mode; attempts = 0; vbar = vbar_hit;
    for (auto &stat : g_stats) stat.store(0);
    std::vector<unsigned char> source(size), destination(size);
    return xpu_memcpy_host_to_device(
        reinterpret_cast<CUdeviceptr>(destination.data()),
        source.data(), size, nullptr);
}
extern "C" uint64_t test_stat(int index) { return g_stats[index].load(); }
extern "C" int test_attempts() { return attempts; }
extern "C" void test_message(const char *message) {
    aimdo_xpu::log_h2d_exception(AIMDO_LOG_ERROR, 42, "initial", message);
}
'''


@pytest.fixture(scope="module")
def native(tmp_path_factory):
    cc, cxx = shutil.which("gcc"), shutil.which("g++")
    if not cc or not cxx:
        pytest.skip("requires gcc and g++ for the native logging harness")
    root = Path(__file__).resolve().parents[1]
    build = tmp_path_factory.mktemp("h2d-diagnostics")
    dispatch = (root / "src-xpu/dispatch.cpp").read_text()
    function = dispatch.split("CUresult xpu_memcpy_host_to_device(", 1)[1]
    function = function.split("CUresult xpu_event_create(", 1)[0]
    harness = build / "harness.cpp"
    harness.write_text(SHIM + "\nCUresult xpu_memcpy_host_to_device(" + function + EXPORTS)
    obj = build / "debug.o"
    # plat.h has pre-existing no-op platform functions with unused arguments.
    subprocess.run([cc, "-std=gnu11", "-fPIC", "-Wall", "-Wextra", "-Werror",
                    "-Wno-unused-parameter",
                    "-I" + str(root / "src"), "-c", str(root / "src/debug.c"),
                    "-o", str(obj)], check=True)
    library = build / "harness.so"
    subprocess.run([cxx, "-std=c++17", "-fPIC", "-shared", "-Wall", "-Wextra",
                    "-Werror", "-I" + str(root / "src"),
                    "-I" + str(root / "src-xpu"), str(harness), str(obj),
                    "-o", str(library)], check=True)
    lib = ctypes.CDLL(str(library))
    lib.set_log_callback.argtypes = [control._LOG_CALLBACK]
    lib.set_log_callback.restype = None
    lib.test_run_h2d.argtypes = [ctypes.c_int, ctypes.c_size_t, ctypes.c_int]
    lib.test_run_h2d.restype = ctypes.c_int
    lib.test_stat.argtypes = [ctypes.c_int]
    lib.test_stat.restype = ctypes.c_uint64
    lib.test_attempts.restype = ctypes.c_int
    lib.test_message.argtypes = [ctypes.c_char_p]
    lib.test_message.restype = None
    return lib


@pytest.fixture
def collected(native, caplog):
    caplog.set_level(logging.DEBUG)
    callback = control._LOG_CALLBACK(control._native_log)
    native.set_log_callback(callback)
    try:
        yield native, caplog
    finally:
        native.set_log_callback(control._LOG_CALLBACK())


def test_success_remains_quiet_and_preserves_counters(collected):
    lib, logs = collected
    assert lib.test_run_h2d(0, 32, 0) == 0
    assert not logs.records
    assert [lib.test_stat(i) for i in range(4)] == [1, 32, 1, 0]


def test_initial_and_retry_errors_reach_python_collector(collected, capfd):
    lib, logs = collected
    assert lib.test_run_h2d(1, 10 << 20, 0) == 999
    messages = [r.getMessage() for r in logs.records]
    assert "message=original copy failure" in messages[0]
    assert "stage=initial" in messages[0]
    assert any("stage=split_retry" in m and "message=different retry failure" in m
               for m in messages)
    assert any("sycl_code=77 category=test_sycl" in m for m in messages)
    assert all("call=1 " in m for m in messages)
    assert any("dest_kind=usm_device vbar=0" in m for m in messages)
    assert not any("all_mapped=" in m for m in messages)
    assert not capfd.readouterr().err  # Collected once, through Python.


def test_original_exception_is_collected_before_retry(native):
    observed = []
    callback = control._LOG_CALLBACK(
        lambda level, message: observed.append((native.test_attempts(), message)))
    native.set_log_callback(callback)
    try:
        assert native.test_run_h2d(1, 10 << 20, 0) == 999
    finally:
        native.set_log_callback(control._LOG_CALLBACK())
    assert observed[0][0] == 1
    assert b"message=original copy failure" in observed[0][1]


def test_error_only_collector_keeps_both_failures_and_copy_identity(collected):
    lib, logs = collected
    logs.set_level(logging.ERROR)
    assert lib.test_run_h2d(1, 10 << 20, 0) == 999
    assert "stage=failed_initial" in logs.text
    assert "message=original copy failure" in logs.text
    assert "message=different retry failure" in logs.text
    assert "queue=" in logs.text and "destination=" in logs.text
    assert all(r.levelno == logging.ERROR for r in logs.records)


def test_recovered_retry_is_not_reported_as_fatal(collected):
    lib, logs = collected
    size = 10 << 20
    assert lib.test_run_h2d(2, size, 0) == 0
    assert all(r.levelno == logging.WARNING for r in logs.records)
    assert "result=recovered" in logs.records[-1].getMessage()
    assert [lib.test_stat(i) for i in range(4)] == [1, size, 1, 1]


@pytest.mark.parametrize(("mode", "expected"), [
    (3, "retry standard failure"), (4, "<non-standard exception>"),
    (6, "initial standard failure"), (7, "<non-standard exception>"),
])
def test_standard_and_unknown_exceptions_are_preserved(collected, mode, expected):
    lib, logs = collected
    assert lib.test_run_h2d(mode, 10 << 20, 0) == 999
    assert any("message=" + expected in r.getMessage() for r in logs.records)


def test_retry_records_the_failing_chunk_offset(collected):
    lib, logs = collected
    assert lib.test_run_h2d(9, 10 << 20, 0) == 999
    assert any("offset=8388608 piece_size=2097152" in r.getMessage()
               for r in logs.records)


def test_diagnostic_failure_does_not_hide_copy_errors(collected):
    lib, logs = collected
    assert lib.test_run_h2d(5, 10 << 20, 0) == 999
    text = logs.text
    assert "original copy failure" in text
    assert "different retry failure" in text
    assert "stage=diagnostics" in text and "diagnostic query failure" in text


def test_small_failure_keeps_vbar_diagnostics_without_retry(collected):
    lib, logs = collected
    assert lib.test_run_h2d(1, 32, 1) == 999
    assert lib.test_attempts() == 1
    assert all(r.levelno == logging.ERROR for r in logs.records)
    assert "all_mapped=0 min_pin=0 first_page=558 span=2 first_unmapped=558" in logs.text


@pytest.mark.parametrize("message", ["", "x" * 8192 + "TAIL",
    "x" * 191 + "显存错误🙂" * 100 + "\r\npath\\file\nTAIL"],
    ids=["empty", "long", "multiline-utf8"])
def test_long_multiline_utf8_messages_are_complete(collected, message):
    lib, logs = collected
    lib.test_message(message.encode())
    records = [r.getMessage() for r in logs.records]
    parts = [m.split(" message=", 1)[1] for m in records if " message=" in m]
    expected = message.replace("\\", "\\\\").replace("\r", "\\r").replace("\n", "\\n")
    assert "".join(parts) == expected
    assert f"message_end parts={len(parts)}" in records[-1]
    assert all("\n" not in m and "\r" not in m for m in records)
    assert all(len(m.encode()) < 640 for m in records)


def test_native_stderr_fallback_is_complete(native, capfd):
    native.set_log_callback(control._LOG_CALLBACK())
    native.test_message(b"x" * 8192 + b"TAIL")
    output = capfd.readouterr().err
    parts = [line.split(" message=", 1)[1] for line in output.splitlines()
             if " message=" in line]
    assert "".join(parts) == "x" * 8192 + "TAIL"
    assert "message_end parts=" in output


def test_broken_python_collector_falls_back_to_stderr(collected, monkeypatch, capfd):
    lib, _ = collected
    def broken_collector(*args, **kwargs):
        raise RuntimeError("collector failed")
    monkeypatch.setattr(control.logging, "log", broken_collector)
    # Exercise interrupted and partial writes in the fallback, too.
    original_write = control.os.write
    calls = []
    def short_write(fd, data):
        calls.append(len(data))
        if len(calls) == 1:
            raise InterruptedError()
        return original_write(fd, data[:7])
    monkeypatch.setattr(control.os, "write", short_write)
    lib.test_message(b"primary error survives a broken collector")
    output = capfd.readouterr().err
    assert "message=primary error survives a broken collector" in output
    assert "message_end parts=1" in output
    assert len(calls) > 2
