from pathlib import Path


def test_wddm_pressure_uses_current_usage_as_sampled_baseline():
    source = (
        Path(__file__).resolve().parents[1] / "src-win" / "shmem-detect.c"
    ).read_text(encoding="utf-8")
    normalized = " ".join(source.split())

    assert "effective_usage = info.CurrentUsage;" in source
    # The DXGI signal must enforce the configured reserve, not just the
    # 512 MiB floor: applying only WDDM_BUDGET_HEADROOM here while
    # deficit_simple applied the reserve to AIMDO's own accounting let usage
    # settle a fixed 0.77-0.78 GiB above target, which is the untracked
    # SYCL/oneDNN/driver share.
    assert (
        "deficit_sync = (ssize_t)effective_usage + wddm_headroom - "
        "(ssize_t)effective_budget;"
    ) in normalized
    assert "simple_vram_headroom > wddm_headroom" in normalized
    assert "total_vram_usage + WDDM_BUDGET_HEADROOM" not in source


def test_xpu_pressure_accounts_for_wddm_nonlocal_fallback():
    source = (
        Path(__file__).resolve().parents[1] / "src-win" / "shmem-detect.c"
    ).read_text(encoding="utf-8")

    assert "DXGI_MEMORY_SEGMENT_GROUP_NON_LOCAL" in source
    assert "nonlocal_info.CurrentUsage -" in source
    assert "nonlocal_excess > WDDM_BUDGET_HEADROOM" in source
    assert '"WDDM non-local usage"' in source


def test_xpu_reclaims_without_blocking_inside_the_allocation_path():
    source = (
        Path(__file__).resolve().parents[1] / "src-xpu" / "stubs.c"
    ).read_text(encoding="utf-8")

    assert "if (size >= (size_t)1 << 30)" in source
    assert "aimdo_wddm_force_poll();" in source
    prepare = source.split("bool aimdo_xpu_prepare_allocation", 1)[1]
    windows_path = prepare.split(
        "#if defined(_WIN32) || defined(_WIN64)", 1
    )[1].split("#else", 1)[0]
    # The allocation hook runs inside the driver's allocation call, so it may
    # only use the reclaim that never waits on the compute queue.
    assert "vbars_free_retired(deficit);" in windows_path
    assert "vbars_free(" not in windows_path
    assert "cuCtxSynchronize" not in windows_path


def test_windows_model_boundary_applies_deferred_native_pressure():
    source = (
        Path(__file__).resolve().parents[1]
        / "comfy_aimdo"
        / "model_vbar.py"
    ).read_text(encoding="utf-8")

    assert "anticipated_growth = max(0, peak_reserved - reserved)" in source
    assert "lib.vbars_prepare_allocation(" in source
    normalized = " ".join(source.split())
    assert (
        "lib.vbars_prepare_allocation.argtypes = [ "
        "ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint64 ]"
    ) in normalized
    assert (
        "lib.vbars_prepare_allocation( self._devctx, self._ptr, "
        "anticipated_growth )"
    ) in normalized
    assert "previous_watermark = lib.vbar_get_watermark(" in source
    assert "self._prioritized_once = False" in source
    assert 'sys.platform == "win32" and self._prioritized_once' in source
    assert "self._prioritized_once = True" in source
    assert "previous_watermark * _VBAR_PAGE_SIZE" not in source
    prioritize_call = source.index("lib.vbar_prioritize(")
    get_watermark_call = source.index(
        "previous_watermark = lib.vbar_get_watermark("
    )
    prepare_call = source.index(
        "lib.vbars_prepare_allocation(", prioritize_call
    )
    assert source.count("lib.vbar_prioritize(") == 1
    assert (
        get_watermark_call
        < prioritize_call
        < prepare_call
    )


def test_windows_speculative_reclaim_preserves_active_vbar():
    source = (
        Path(__file__).resolve().parents[1] / "src" / "model-vbar.c"
    ).read_text(encoding="utf-8")
    normalized = " ".join(source.split())

    assert "static size_t vbars_free_except(" in source
    assert "if (i == preserved)" in source
    assert "return vbars_free_except(size, NULL);" in source
    assert (
        "void vbars_prepare_allocation(void *devctx, void *vbar, "
        "uint64_t size)"
    ) in normalized
    assert (
        "vbars_free_except(budget_deficit((size_t)size), "
        "(ModelVBAR *)vbar);"
    ) in normalized


def test_vbar_fault_reclaims_the_actual_deficit_before_retry():
    source = (
        Path(__file__).resolve().parents[1] / "src" / "model-vbar.c"
    ).read_text(encoding="utf-8")

    assert "allocation_deficit = budget_deficit(VBAR_PAGE_SIZE);" in source
    assert "allocation_deficit > (ssize_t)VBAR_PAGE_SIZE" in source
    assert "vbars_free((ssize_t)retry_reclaim);" in source


def test_vbar_fault_wddm_retry_margin_is_windows_xpu_only():
    source = (
        Path(__file__).resolve().parents[1] / "src" / "model-vbar.c"
    ).read_text(encoding="utf-8")

    guard = "#if defined(AIMDO_XPU) && (defined(_WIN32) || defined(_WIN64))"
    # The retry margin, the page retirement epoch and the non-blocking reclaim
    # are all Windows XPU only; Linux and CUDA keep the original behaviour.
    assert source.count(guard) >= 2
    assert "#define VBAR_WDDM_RETRY_RECLAIM (512 << 20)" in source
    assert "(void)vbars_free_retired(VBAR_WDDM_RETRY_RECLAIM);" in source
    # A fault is already on the execution path that needs the missing page.
    # Waiting for the whole device here can deadlock; a failed non-blocking
    # reclaim must become a recoverable host-offload OOM instead.
    retry = source.split(
        "VBAR Windows XPU retry reclaiming an additional", 1
    )[1].split("#endif", 1)[0]
    assert "vbars_free(VBAR_WDDM_RETRY_RECLAIM)" not in retry
    for windows_only in ("VBAR_WDDM_RETRY_RECLAIM", "retire_epoch",
                         "vbars_free_retired"):
        assert windows_only not in source.split(guard, 1)[0]


def test_non_blocking_reclaim_never_waits_or_moves_the_watermark():
    source = (
        Path(__file__).resolve().parents[1] / "src" / "model-vbar.c"
    ).read_text(encoding="utf-8")

    body = source.split("size_t vbars_free_retired(ssize_t size)", 1)[1]
    body = body.split("\n}\n", 1)[0]

    # Waiting here would block the driver's allocation call behind work that
    # may itself be waiting for the residency being requested.
    assert "cuCtxSynchronize" not in body
    assert "aimdo_xpu_retired_epoch()" in body
    assert "rp->retire_epoch > retired" in body
    # Lowering the watermark denies future faults until the next activation,
    # which turns a momentary spike into a permanently smaller working set.
    assert "watermark--" not in body
    assert "i->watermark =" not in body


def test_windows_vbar_reclaim_serializes_page_state_without_waiting():
    root = Path(__file__).resolve().parents[1]
    source = (root / "src" / "model-vbar.c").read_text(encoding="utf-8")
    control = (root / "src" / "control.c").read_text(encoding="utf-8")
    header = (root / "src" / "control.h").read_text(encoding="utf-8")

    assert "void *_vbar_lock" in header
    assert "vbar_lock = mutex_create();" in control
    assert "mutex_destroy((Mutex)vbar_lock);" in control

    reclaim = source.split("size_t vbars_free_retired(ssize_t size)", 1)[1]
    reclaim = reclaim.split("\n}\n", 1)[0]
    assert "vbar_state_try_lock()" in reclaim
    assert "vbar_state_lock()" not in reclaim
    assert "cuCtxSynchronize" not in reclaim
    assert "if (!vbar_async_reclaim_enabled())" in reclaim

    fault = source.split("int vbar_fault(void *devctx", 1)[1]
    fault = fault.split("\n}\n", 1)[0]
    assert "vbar_state_lock();" in fault
    assert "vbar_fault_locked(" in fault
    assert "vbar_state_unlock();" in fault


def test_windows_final_unpin_publishes_epoch_before_idle_state():
    source = (
        Path(__file__).resolve().parents[1] / "src" / "model-vbar.c"
    ).read_text(encoding="utf-8")
    unpin = source.split("void vbar_unpin_stream", 1)[1]
    unpin = unpin.split("\n}\n", 1)[0]

    epoch_snapshot = unpin.index(
        "retirement_epoch = aimdo_xpu_retire_epoch_current();"
    )
    state_lock = unpin.index("vbar_state_lock();")
    epoch_publish = unpin.index("rp->retire_epoch = retirement_epoch;")
    idle_publish = unpin.index("rp->pin_count = 0;")
    state_unlock = unpin.index("vbar_state_unlock();")

    assert epoch_snapshot < state_lock < epoch_publish < idle_publish < state_unlock
