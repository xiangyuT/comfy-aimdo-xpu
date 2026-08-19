import pytest

from comfy_aimdo import model_vbar


def test_inference_memory_budget_is_scoped_per_device():
    assert model_vbar.current_inference_memory_budget(0) == 0

    with model_vbar.inference_memory_budget(4096, [0, 1]):
        assert model_vbar.current_inference_memory_budget(0) == 4096
        assert model_vbar.current_inference_memory_budget(1) == 4096
        assert model_vbar.current_inference_memory_budget(2) == 0

    assert model_vbar.current_inference_memory_budget(0) == 0


def test_nested_inference_memory_budget_keeps_larger_outer_budget():
    with model_vbar.inference_memory_budget(8192, [0]):
        with model_vbar.inference_memory_budget(4096, [0, 1]):
            assert model_vbar.current_inference_memory_budget(0) == 8192
            assert model_vbar.current_inference_memory_budget(1) == 4096
        assert model_vbar.current_inference_memory_budget(0) == 8192
        assert model_vbar.current_inference_memory_budget(1) == 0


def test_inference_memory_budget_is_cleared_after_exception():
    with pytest.raises(RuntimeError, match="load failed"):
        with model_vbar.inference_memory_budget(4096, [0]):
            raise RuntimeError("load failed")

    assert model_vbar.current_inference_memory_budget(0) == 0
