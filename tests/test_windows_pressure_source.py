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


def test_xpu_defers_vbar_mutation_out_of_the_allocation_path():
    source = (
        Path(__file__).resolve().parents[1] / "src-xpu" / "stubs.c"
    ).read_text(encoding="utf-8")

    assert "if (size >= (size_t)1 << 30)" in source
    assert "aimdo_wddm_force_poll();" in source
    prepare = source.split("bool aimdo_xpu_prepare_allocation", 1)[1]
    windows_path = prepare.split(
        "#if defined(_WIN32) || defined(_WIN64)", 1
    )[1].split("#else", 1)[0]
    # Being above the driver entry point is not sufficient: the callback still
    # runs under the allocator/UMF stack. It can publish pressure, but actual
    # Level Zero unmap must happen at a model-owner boundary.
    assert "vbars_request_reclaim(deficit);" in windows_path
    assert "vbars_free_retired" not in windows_path
    assert "vbars_free(" not in windows_path
    assert "cuCtxSynchronize" not in windows_path

    retry = source.split("bool aimdo_xpu_retry_allocation", 1)[1]
    retry = retry.split("bool aimdo_xpu_allocation_deficit", 1)[0]
    evict = source.split("bool aimdo_xpu_evict_for_allocation", 1)[1]
    evict = evict.split("bool aimdo_xpu_account_allocation", 1)[0]
    for allocation_path in (retry, evict):
        assert "vbars_request_reclaim" in allocation_path
        assert "vbars_free_retired" not in allocation_path


def test_windows_callback_streaming_path_only_publishes_reclaim_pressure():
    root = Path(__file__).resolve().parents[1]
    source = (root / "src" / "hostbuf.c").read_text(encoding="utf-8")

    assert "vbars_request_reclaim(deficit);" in source
    assert "vbars_free_retired(deficit);" not in source


def test_windows_direct_file_reader_reclaims_before_submitting_h2d():
    source = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "hostbuf-file-reader.c"
    ).read_text(encoding="utf-8")
    windows_path = source.split(
        "#if defined(AIMDO_XPU) && (defined(_WIN32) || defined(_WIN64))", 2
    )[2].split("#endif", 1)[0]

    assert "remaining_pages = vbars_free_retired(deficit);" in windows_path
    assert "if (remaining_pages)" in windows_path
    assert "vbars_request_reclaim(deficit);" in windows_path
    assert source.index("remaining_pages = vbars_free_retired(deficit);") < source.index(
        "CUresult copy_result = cuMemcpyHtoDAsync"
    )
    assert '"[AIMDO XPU RECLAIM] op=pre_h2d destination=%p "' in windows_path
    assert "requested_pages - remaining_pages" in windows_path
    assert "cuCtxSynchronize" not in windows_path


def test_windows_owner_boundary_consumes_deferred_reclaim():
    root = Path(__file__).resolve().parents[1]
    source = (root / "src" / "model-vbar.c").read_text(encoding="utf-8")
    header = (root / "src" / "control.h").read_text(encoding="utf-8")

    assert "int64_t _vbar_reclaim_requested;" in header
    assert "void vbars_request_reclaim(ssize_t size)" in source
    request = source.split("void vbars_request_reclaim(ssize_t size)", 1)[1]
    request = request.split("\n}\n", 1)[0]
    assert "InterlockedCompareExchange64" in request
    assert "vbars_free_retired" not in request
    assert "cuMemUnmap" not in request

    fault = source.split("int vbar_fault(void *devctx", 1)[1]
    fault = fault.split("\n}\n", 1)[0]
    assert "vbars_reclaim_at_owner_boundary(budget_deficit(0));" in fault
    assert fault.index("vbars_reclaim_at_owner_boundary") < fault.index(
        "vbar_state_lock();"
    )

    prepare = source.split("void vbars_prepare_allocation", 1)[1]
    prepare = prepare.split("void vbar_prioritize", 1)[0]
    assert "vbars_take_reclaim_request();" in prepare
    assert "vbars_free_synchronized_except(reclaim, (ModelVBAR *)vbar);" in prepare
    assert "vbars_free_retired_except(reclaim, (ModelVBAR *)vbar);" in prepare
    assert "if (vbar_async_reclaim_enabled())" in prepare


def test_windows_model_boundary_applies_deferred_native_pressure():
    source = (
        Path(__file__).resolve().parents[1]
        / "comfy_aimdo"
        / "model_vbar.py"
    ).read_text(encoding="utf-8")

    assert "historical_growth = max(0, peak_reserved - reserved)" in source
    assert (
        "inference_budget = current_inference_memory_budget(self.device)"
        in source
    )
    assert (
        "anticipated_growth = max(historical_growth, inference_budget)"
        in source
    )
    assert "lib.vbars_prepare_allocation(" in source
    assert "if anticipated_growth:" not in source
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
    assert "reclaim = budget_deficit((size_t)size);" in normalized
    assert (
        "(void)vbars_free_synchronized_except(reclaim, (ModelVBAR *)vbar);"
        in normalized
    )
    assert (
        "(void)vbars_free_retired_except(reclaim, (ModelVBAR *)vbar);"
        in normalized
    )

    prepare = source.split("void vbars_prepare_allocation", 1)[1]
    prepare = prepare.split("\n}\n", 1)[0]
    assert "one_time_setup();" not in prepare

    free_except = source.split("static size_t vbars_free_except", 1)[1]
    free_except = free_except.split("\n}\n", 1)[0]
    assert free_except.index("vbar_state_lock();") < free_except.index(
        "one_time_setup();"
    )


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
    # The retry margin, page retirement tokens and non-blocking reclaim are all
    # Windows XPU only; Linux and CUDA keep the original behaviour.
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
    for windows_only in ("VBAR_WDDM_RETRY_RECLAIM", "retire_tokens",
                         "vbars_free_retired"):
        assert windows_only not in source.split(guard, 1)[0]


def test_non_blocking_reclaim_never_waits_or_moves_the_watermark():
    source = (
        Path(__file__).resolve().parents[1] / "src" / "model-vbar.c"
    ).read_text(encoding="utf-8")

    body = source.split(
        "static size_t vbars_free_retired_except", 1
    )[1]
    body = body.split("\n}\n", 1)[0]

    # Waiting here would block the driver's allocation call behind work that
    # may itself be waiting for the residency being requested.
    assert "cuCtxSynchronize" not in body
    assert "aimdo_xpu_retire_snapshot(" in body
    assert "vbar_freeze_retired_candidates(" in body
    freeze = source.split(
        "static size_t vbar_freeze_retired_candidates", 1
    )[1].split("\n}\n", 1)[0]
    assert "vbar_retire_complete(rp, completed, completed_count)" in freeze
    assert "cuCtxSynchronize" not in freeze
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

    reclaim = source.split(
        "static size_t vbars_free_retired_except", 1
    )[1]
    reclaim = reclaim.split("\n}\n", 1)[0]
    freeze = source.split(
        "static size_t vbar_freeze_retired_candidates", 1
    )[1].split("\n}\n", 1)[0]
    commit = source.split(
        "static size_t vbar_commit_candidates", 1
    )[1].split("\n}\n", 1)[0]
    assert "vbar_state_try_lock()" in freeze
    assert "cuCtxSynchronize" not in reclaim
    assert "if (!vbar_async_reclaim_enabled())" in reclaim
    assert "rp->evicting = 1;" in freeze
    for identity in (
        "rp->handle != candidate->handle",
        "rp->serial != candidate->serial",
        "rp->eviction_generation != candidate->eviction_generation",
        "rp->pin_count",
    ):
        assert identity in commit

    fault = source.split("int vbar_fault(void *devctx", 1)[1]
    fault = fault.split("\n}\n", 1)[0]
    assert "vbar_state_lock();" in fault
    assert "vbar_fault_locked(" in fault
    assert "vbar_state_unlock();" in fault


def test_windows_unpin_publishes_queue_token_before_idle_state():
    source = (
        Path(__file__).resolve().parents[1] / "src" / "model-vbar.c"
    ).read_text(encoding="utf-8")
    unpin = source.split("void vbar_unpin_stream", 1)[1]
    unpin = unpin.split("\n}\n", 1)[0]

    token_snapshot = unpin.index("retirement_token = vbar_consumer_dependency(")
    state_lock = unpin.index("vbar_state_lock();")
    token_publish = unpin.index(
        "vbar_retire_record(rp, retirement_token);"
    )
    idle_publish = unpin.index("rp->pin_count--;")
    state_unlock = unpin.index("vbar_state_unlock();")

    assert token_snapshot < state_lock < token_publish < idle_publish < state_unlock


def test_windows_async_mode_is_default_and_reference_has_no_token_path():
    root = Path(__file__).resolve().parents[1]
    source = (root / "src" / "model-vbar.c").read_text(encoding="utf-8")
    dispatch = (root / "src-xpu" / "dispatch.cpp").read_text(
        encoding="utf-8"
    )

    dependency = source.split("static uint64_t vbar_consumer_dependency", 1)[1]
    dependency = dependency.split("\n}\n", 1)[0]
    disabled = dependency.split("if (!vbar_async_reclaim_enabled())", 1)[1]
    disabled = disabled.split("return 0;", 1)[0]
    assert "aimdo_xpu_register_consumer_queue" in disabled
    assert "aimdo_xpu_retire_token_current" not in disabled
    assert "submit_retire_fence_locked" not in disabled
    default_mode = source.split(
        "static bool vbar_async_reclaim_enabled", 1
    )[1].split("static inline void vbar_retire_reset", 1)[0]
    assert "LONG detected = 1;" in default_mode
    assert "length == 1 && value[0] == '0'" in default_mode

    assert "std::atomic<bool> g_retire_accepting{false};" in dispatch
    assert "g_retire_accepting.store(false" in dispatch
    assert "queue.wait_and_throw();" in dispatch
    reset = dispatch.split("bool aimdo_xpu_retire_reset(void)", 1)[1]
    reset = reset.split("AIMDO_XPU_EXPORT void *xpu_alloc_fn", 1)[0]
    assert "if (!drained)" in reset
    assert reset.index("if (!drained)") < reset.index("queue.reset();")


def test_windows_queue_registry_owns_and_validates_queue_identity():
    dispatch = (
        Path(__file__).resolve().parents[1] / "src-xpu" / "dispatch.cpp"
    ).read_text(encoding="utf-8")

    assert "std::shared_ptr<sycl::queue> queue;" in dispatch
    assert "void *source_pointer = nullptr;" in dispatch
    assert "ze_context_handle_t context = nullptr;" in dispatch
    assert "ze_device_handle_t device = nullptr;" in dispatch
    assert "uint64_t incarnation = 0;" in dispatch
    assert "retire_queue.context == context" in dispatch
    assert "retire_queue.device == device" in dispatch
    assert "*retire_queue.queue == *queue" in dispatch
    context_sync = dispatch.split("CUresult xpu_context_synchronize()", 1)[1]
    context_sync = context_sync.split("\n}\n", 1)[0]
    lock_end = context_sync.index("for (sycl::queue &queue : queues)")
    assert context_sync.index("std::lock_guard<std::mutex>") < lock_end
    assert context_sync.index("queue.wait_and_throw();") > lock_end


def test_windows_explicit_consumer_and_capture_fail_closed_are_exposed():
    source = (
        Path(__file__).resolve().parents[1]
        / "comfy_aimdo"
        / "model_vbar.py"
    ).read_text(encoding="utf-8")

    assert "def vbar_register_consumer(alloc, stream=None):" in source
    assert "lib.vbar_register_consumer_stream(" in source
    assert 'getattr(torch.xpu, "is_current_stream_capturing", None)' in source
    capture = source.split("def _consumer_queue_ptr", 1)[1]
    capture = capture.split("def _trace_vbar", 1)[0]
    assert "if callable(is_capturing) and is_capturing():" in capture
    assert "return 0" in capture


def test_windows_unmap_release_failure_keeps_metadata_truthful():
    source = (
        Path(__file__).resolve().parents[1] / "src" / "model-vbar.c"
    ).read_text(encoding="utf-8")
    mod1 = source.split("static inline bool mod1", 1)[1]
    mod1 = mod1.split("static size_t vbars_free_except", 1)[0]

    assert "if (!CHECK_CU_ERROR(cuMemUnmap" in mod1
    assert "if (!CHECK_CU_ERROR(cuMemRelease" in mod1
    assert "vbar_restore_page_mapping" in mod1
    assert "rp->mapped = VBAR_MAPPING_UNMAPPED;" in mod1
    assert "rp->retire_unknown = 1;" in mod1
    assert mod1.index("cuMemRelease") < mod1.index("rp->handle = 0;")

    map_new = source.split("static CUresult vbar_map_new_page", 1)[1]
    map_new = map_new.split("static inline bool mod1", 1)[0]
    assert "if (rp->handle)" in map_new
    assert "VBAR_MAPPING_UNKNOWN" in map_new
    assert "total_vram_usage += VBAR_PAGE_SIZE;" in map_new


def test_windows_retirement_is_per_queue_and_fence_submissions_are_batched():
    root = Path(__file__).resolve().parents[1]
    dispatch = (root / "src-xpu" / "dispatch.cpp").read_text(
        encoding="utf-8"
    )
    model_vbar = (root / "src" / "model-vbar.c").read_text(
        encoding="utf-8"
    )

    assert "constexpr size_t kRetireBatchUses = 64;" in dispatch
    assert "struct RetireQueue" in dispatch
    assert "uint64_t aimdo_xpu_retire_token_current(" in dispatch
    assert "retire_queue.pending_uses < kRetireBatchUses" in dispatch
    assert "retire_queue.pending_uses >= kRetireBatchUses" in dispatch
    assert "size_t aimdo_xpu_retire_snapshot(" in dispatch
    assert "force_submit && retire_queue.pending_uses" in dispatch
    assert "retire_queue.incarnation <<" in dispatch
    assert "completed_incarnation != incarnation" in model_vbar
    assert "g_retired_epoch" not in dispatch
    assert "g_retire_pending" not in dispatch

    assert "retire_tokens[AIMDO_XPU_RETIRE_MAX_QUEUES]" in model_vbar
    assert "vbar_retire_record(rp, retirement_token);" in model_vbar
    assert "vbar_retire_complete(" in model_vbar
    assert "retire_unknown" in model_vbar
