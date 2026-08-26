from pathlib import Path


SOURCE = (Path(__file__).parents[1] / "src" / "model-vbar.c").read_text(
    encoding="utf-8"
)


def _function_body(signature, next_signature):
    return SOURCE.split(signature, 1)[1].split(next_signature, 1)[0]


def test_vbar_page_rounding_and_metadata_size_fail_closed_on_overflow():
    page_count = _function_body(
        "static bool vbar_page_count", "static bool vbar_metadata_allocation_size"
    )
    metadata_size = _function_body(
        "static bool vbar_metadata_allocation_size",
        "static bool vbar_fault_page_range",
    )
    allocation = _function_body(
        "void *vbar_allocate", "void vbar_set_watermark_limit"
    )

    assert "size > UINT64_MAX - (VBAR_PAGE_SIZE - 1)" in page_count
    assert "pages > SIZE_MAX" in page_count
    assert (
        "nr_pages > (SIZE_MAX - sizeof(ModelVBAR)) / sizeof(ResidentPage)"
        in metadata_size
    )
    assert allocation.index("vbar_page_count(size, &nr_pages)") < allocation.index(
        "size = (uint64_t)nr_pages * VBAR_PAGE_SIZE"
    )
    assert allocation.index(
        "vbar_metadata_allocation_size(nr_pages, &allocation_size)"
    ) < allocation.index("calloc(1, allocation_size)")


def test_vbar_fault_validates_bytes_before_end_addition_or_page_access():
    range_helper = _function_body(
        "static bool vbar_fault_page_range", "/* These public entry points"
    )
    fault = _function_body(
        "static int vbar_fault_locked", "int vbar_fault(void *devctx"
    )

    byte_guard = "offset > reserved_size || size > reserved_size - offset"
    assert byte_guard in range_helper
    assert range_helper.index(byte_guard) < range_helper.index(
        "vbar_page_count(offset + size, page_end)"
    )
    assert "VBAR_GET_PAGE_NR_UP(offset + size)" not in fault
    assert fault.index("vbar_fault_page_range(") < fault.index(
        "if (page_end > mv->watermark)"
    )
    assert fault.index("vbar_fault_page_range(") < fault.index(
        "mv->residency_map[page_nr]"
    )
    assert "return VBAR_FAULT_OOM;" in fault.split(
        "if (!vbar_fault_page_range(", 1
    )[1].split("log(VVERBOSE", 1)[0]
