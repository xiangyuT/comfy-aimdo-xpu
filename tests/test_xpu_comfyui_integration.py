import gc

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("comfy.quant_ops")

from comfy_aimdo import control


pytestmark = pytest.mark.skipif(
    not hasattr(torch, "xpu") or not torch.xpu.is_available(),
    reason="requires an available Torch XPU device",
)


@pytest.fixture(scope="module", autouse=True)
def initialized_xpu_backend():
    assert control.init(implementation="xpu", simple_vram_headroom=64 << 20)
    assert control.init_devices([torch.xpu.current_device()])
    yield
    torch.xpu.synchronize()
    control.deinit()


def test_tensorwise_int8_quantized_tensor_rebuilds_inside_vbar():
    from comfy.memory_management import interpret_gathered_like, vram_aligned_size
    from comfy.quant_ops import QuantizedTensor, TensorWiseINT8Layout
    from comfy_aimdo.model_vbar import ModelVBAR
    from comfy_aimdo.torch import aimdo_to_tensor, copy_to_vbar

    device = torch.device("xpu", torch.xpu.current_device())
    shape = (512, 256)
    qdata = (
        torch.arange(shape[0] * shape[1], dtype=torch.int32)
        .remainder(255)
        .sub(127)
        .to(torch.int8)
        .reshape(shape)
    )
    scale = torch.tensor(0.03125, dtype=torch.float32)
    params = TensorWiseINT8Layout.Params(
        scale=scale,
        orig_dtype=torch.bfloat16,
        orig_shape=shape,
        is_weight=True,
        convrot=False,
        convrot_groupsize=256,
    )
    source = QuantizedTensor(qdata, "TensorWiseINT8Layout", params)

    size = vram_aligned_size(source)
    vbar = ModelVBAR(32 << 20, device.index)
    allocation = vbar.alloc(size)
    assert vbar.fault(allocation[1], allocation[2]) is not None
    gathered = aimdo_to_tensor(allocation, device)
    destination = interpret_gathered_like([source], gathered)[0]

    copy_to_vbar(destination._qdata, source._qdata)
    copy_to_vbar(destination._params.scale, source._params.scale)
    torch.xpu.synchronize()

    assert torch.equal(destination._qdata.cpu(), qdata)
    assert torch.equal(destination._params.scale.cpu(), scale)
    expected = (qdata.float() * scale).to(torch.bfloat16)
    torch.testing.assert_close(destination.dequantize().cpu(), expected)

    vbar.unpin(allocation[1], allocation[2])
    del destination, gathered, source, allocation
    vbar.__del__()
    del vbar
    gc.collect()
