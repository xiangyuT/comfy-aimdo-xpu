#include "plat.h"
#include "xfer-file.h"

#define HOSTBUF_FILE_READER_WINDOW (64ULL * 1024ULL * 1024ULL)
#define LEAD_IN_THRESHOLD (HOSTBUF_FILE_READER_WINDOW - 16ULL * 1024ULL * 1024ULL)

#if defined(AIMDO_XPU) && (defined(_WIN32) || defined(_WIN64))
#include <windows.h>

/* Diagnostic switch for the streaming-path reclaim, so its effect can be
 * measured on the real workload instead of assumed. */
static bool aimdo_stream_reclaim_disabled(void) {
    static int disabled = -1;
    char value[8];

    if (disabled < 0) {
        DWORD length = GetEnvironmentVariableA(
            "AIMDO_XPU_NO_STREAM_RECLAIM", value, sizeof(value));
        disabled = (length > 0 && length < sizeof(value) && value[0] == '1');
    }
    return disabled != 0;
}
#endif

static bool hostbuf_file_reader_retire_active(void) {
    HostbufFileReaderSlot *slot;

    if (g_devctx->_hostbuf_file_reader_active < 0) {
        return true;
    }

    slot = &g_devctx->_hostbuf_file_reader_slots[g_devctx->_hostbuf_file_reader_active];
    if (!slot->offset) {
        return true;
    }
    if (slot->event) {
        log(AIMDO_LOG_ERROR, "%s: active slot %d already has a completion event\n", __func__,
            g_devctx->_hostbuf_file_reader_active);
        return false;
    }
    return CHECK_CU(cuEventCreate(&slot->event, CU_EVENT_DISABLE_TIMING)) &&
           CHECK_CU(cuEventRecord(slot->event, (CUstream)slot->stream));
}

static HostbufFileReaderSlot *hostbuf_file_reader_next(cudaStream_t stream) {
    HostbufFileReaderSlot *slot;

    g_devctx->_hostbuf_file_reader_active =
        (g_devctx->_hostbuf_file_reader_active + 1) % HOSTBUF_FILE_READER_SLOTS;
    slot = &g_devctx->_hostbuf_file_reader_slots[g_devctx->_hostbuf_file_reader_active];

    if (slot->buffer && slot->event) {
        if (!CHECK_CU(cuEventSynchronize(slot->event)) ||
            !CHECK_CU(cuEventDestroy(slot->event))) {
            return NULL;
        }
        slot->event = NULL;
    }

    if (!slot->buffer &&
        !CHECK_CU_OOM_ERROR(cuMemAllocHost((void **)&slot->buffer, HOSTBUF_FILE_READER_WINDOW))) {
        return NULL;
    }

    slot->offset = 0;
    slot->stream = (CUstream)stream;
    return slot;
}

SHARED_EXPORT
bool hostbuf_file_reader_read(int device, uint64_t file_handle, uint64_t file_offset,
                              uint64_t size, cudaStream_t stream,
                              uint64_t device_ptr, bool mark_cold) {
    if (size == 0) {
        return true;
    }
    if (!device_ptr || device < 0 || !set_devctx_for_device(device)) {
        log(AIMDO_LOG_ERROR, "%s: input validation failed device_ptr=%p device=%d\n",
            __func__, (void *)(uintptr_t)device_ptr, device);
        return false;
    }

    while (size) {
        HostbufFileReaderSlot *slot = g_devctx->_hostbuf_file_reader_active < 0 ? NULL :
            &g_devctx->_hostbuf_file_reader_slots[g_devctx->_hostbuf_file_reader_active];
        size_t chunk;

        if (!slot || slot->stream != (CUstream)stream ||
            (slot->offset + size >= HOSTBUF_FILE_READER_WINDOW &&
             slot->offset >= LEAD_IN_THRESHOLD)) {
            if (!hostbuf_file_reader_retire_active() ||
                !(slot = hostbuf_file_reader_next(stream))) {
                return false;
            }
        }

        chunk = (size_t)MIN(size, HOSTBUF_FILE_READER_WINDOW - slot->offset);
        if (!xfer_file_read(file_handle, file_offset, slot->buffer + slot->offset,
                            chunk, mark_cold)) {
            log(AIMDO_LOG_ERROR, "%s: file read failed handle=0x%llx offset=%llu size=%zu\n",
                __func__, (ull)file_handle, (ull)file_offset, chunk);
            return false;
        }
#if defined(AIMDO_XPU) && (defined(_WIN32) || defined(_WIN64))
        /* This direct reader runs on the model-owner call stack, before the
         * Level Zero copy is submitted.  The preceding VBAR fault has already
         * mapped and pinned the destination, so a non-blocking scan cannot
         * select it.  Reclaim other pages whose consumer fences are already
         * complete now: deferring all of this pressure to the next fault can
         * deadlock forward progress when the synchronous H2D wait itself is
         * the operation that cannot complete under residency pressure.
         *
         * Never wait here. If the completed set cannot satisfy the shortage,
         * preserve the request for the next model-owner boundary. */
        if (!aimdo_stream_reclaim_disabled()) {
            ssize_t deficit = budget_deficit(chunk);

            if (deficit > 0) {
                size_t remaining = vbars_free_retired(deficit);

                if (remaining) {
                    vbars_request_reclaim(deficit);
                }
            }
        }
#endif
        CUresult copy_result = cuMemcpyHtoDAsync((CUdeviceptr)device_ptr,
                                                 slot->buffer + slot->offset,
                                                 chunk, (CUstream)stream);
        if (!CHECK_CU(copy_result)) {
            log(AIMDO_LOG_ERROR, "%s: device copy failed result=%d device_ptr=%p device=%d stream=%p size=%zu\n",
                __func__, (int)copy_result, (void *)(uintptr_t)device_ptr, device,
                (void *)stream, chunk);
            return false;
        }

        slot->offset += chunk;
        file_offset += chunk;
        device_ptr += chunk;
        size -= chunk;
    }

    return true;
}

SHARED_EXPORT
void hostbuf_file_reader_cleanup(void) {
    if (!g_devctx) {
        return;
    }

    hostbuf_file_reader_retire_active();
    for (unsigned i = 0; i < HOSTBUF_FILE_READER_SLOTS; i++) {
        HostbufFileReaderSlot *slot = &g_devctx->_hostbuf_file_reader_slots[i];

        if (slot->buffer && slot->event) {
            CHECK_CU(cuEventSynchronize(slot->event));
            CHECK_CU(cuEventDestroy(slot->event));
        }
        if (slot->buffer) {
            CHECK_CU(cuMemFreeHost(slot->buffer));
        }
    }
    memset(g_devctx->_hostbuf_file_reader_slots, 0,
           sizeof(g_devctx->_hostbuf_file_reader_slots));
    g_devctx->_hostbuf_file_reader_active = -1;
}
