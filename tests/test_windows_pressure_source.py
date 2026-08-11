from pathlib import Path


def test_wddm_pressure_uses_current_usage_as_sampled_baseline():
    source = (
        Path(__file__).resolve().parents[1] / "src-win" / "shmem-detect.c"
    ).read_text(encoding="utf-8")
    normalized = " ".join(source.split())

    assert "effective_usage = info.CurrentUsage;" in source
    assert (
        "deficit_sync = (ssize_t)effective_usage + "
        "(ssize_t)WDDM_BUDGET_HEADROOM - (ssize_t)effective_budget;"
    ) in normalized
    assert "total_vram_usage + WDDM_BUDGET_HEADROOM" not in source


def test_xpu_pressure_accounts_for_wddm_nonlocal_fallback():
    source = (
        Path(__file__).resolve().parents[1] / "src-win" / "shmem-detect.c"
    ).read_text(encoding="utf-8")

    assert "DXGI_MEMORY_SEGMENT_GROUP_NON_LOCAL" in source
    assert "nonlocal_info.CurrentUsage -" in source
    assert "nonlocal_excess > WDDM_BUDGET_HEADROOM" in source
    assert '"WDDM non-local usage"' in source


def test_xpu_forces_wddm_refresh_only_for_large_allocations():
    source = (
        Path(__file__).resolve().parents[1] / "src-xpu" / "stubs.c"
    ).read_text(encoding="utf-8")

    assert "if (size >= (size_t)1 << 30)" in source
    assert "aimdo_wddm_force_poll();" in source
    assert "poll_budget_deficit(&deficit_method);" in source
    prepare = source.split("bool aimdo_xpu_prepare_allocation", 1)[1]
    windows_path = prepare.split(
        "#if defined(_WIN32) || defined(_WIN64)", 1
    )[1].split("#else", 1)[0]
    assert "vbars_free" not in windows_path


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
    assert source.count(guard) == 2
    assert "#define VBAR_WDDM_RETRY_RECLAIM (512 << 20)" in source
    assert "vbars_free(VBAR_WDDM_RETRY_RECLAIM);" in source
