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


class _FakeVBAR:
    def __init__(self):
        self.calls = []
        self.acquire_result = True
        self.release_result = True

    def acquire_consumer(self, offset, size, kind):
        self.calls.append(("acquire", offset, size, kind))
        return self.acquire_result

    def release_consumer(self, offset, size, kind, stream=None):
        self.calls.append(("release", offset, size, kind, stream))
        return self.release_result


def test_external_consumer_contract_acquires_before_and_releases_after_body():
    vbar = _FakeVBAR()
    allocation = (vbar, 4096, 1024)

    with model_vbar.vbar_external_consumer(allocation, stream=17):
        assert vbar.calls == [
            ("acquire", 4096, 1024, model_vbar._CONSUMER_HOLD_EXTERNAL)
        ]
        vbar.calls.append(("submitted",))

    assert vbar.calls[-1] == (
        "release", 4096, 1024,
        model_vbar._CONSUMER_HOLD_EXTERNAL, 17,
    )


def test_external_consumer_exception_is_released_unknown():
    vbar = _FakeVBAR()
    allocation = (vbar, 8192, 2048)

    with pytest.raises(RuntimeError, match="submission failed"):
        with model_vbar.vbar_external_consumer(allocation, stream=23):
            raise RuntimeError("submission failed")

    assert vbar.calls[-1] == (
        "release", 8192, 2048,
        model_vbar._CONSUMER_HOLD_EXTERNAL, 0,
    )


def test_capture_lease_stays_active_until_explicit_release():
    vbar = _FakeVBAR()
    allocation = (vbar, 12288, 4096)

    lease = model_vbar.vbar_capture_begin(allocation)
    assert lease.active
    assert len(vbar.calls) == 1

    lease.release(stream=31)
    assert not lease.active
    assert vbar.calls[-1] == (
        "release", 12288, 4096,
        model_vbar._CONSUMER_HOLD_CAPTURE, 31,
    )
    with pytest.raises(RuntimeError, match="already released"):
        lease.release(stream=31)


def test_consumer_lease_can_retry_when_native_release_did_not_run():
    vbar = _FakeVBAR()
    vbar.release_result = -1
    lease = model_vbar.VBARConsumerLease(
        (vbar, 16384, 4096), model_vbar._CONSUMER_HOLD_EXTERNAL
    )

    with pytest.raises(RuntimeError, match="kept fail-closed"):
        lease.release(stream=37)
    assert lease.active

    vbar.release_result = 1
    lease.release(stream=37)
    assert not lease.active
