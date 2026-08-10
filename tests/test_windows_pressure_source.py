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
