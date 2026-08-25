# AIMDO XPU runtime provider wheel

This builder converts an already-built XPU `comfy-aimdo` wheel into the
co-installable `comfy-aimdo-xpu-runtime` provider distribution. The provider
owns only `comfy_aimdo_xpu_runtime`; Python modules and the Level Zero native
library are stored below its private `_vendor` directory. It never installs a
top-level `comfy_aimdo` file, so the official AIMDO distribution remains
independently upgradeable.

After building `comfy_aimdo/aimdo_xpu.so` and the canonical wheel, run:

```bash
python packaging/xpu_runtime_provider/build_wheel.py \
  --source-wheel dist/comfy_aimdo-0.4.13-cp39-abi3-linux_x86_64.whl \
  --output-dir dist/provider \
  --source-revision "$(git rev-parse HEAD)" \
  --torch-version 2.13.0+xpu \
  --xpu-target bmg
```

The output contains a lightweight
`comfyui_omnixpu.runtime_providers` entry point and a manifest covering the
canonical version, exact source revision, source-wheel hash, native-library
hash, supported runtime, and allocator modes. Importing its metadata does not
import PyTorch or AIMDO.

ComfyUI-OmniXPU activates this provider only when DynamicVRAM is explicitly
enabled and the official AIMDO attempt has left no live native or allocator
state. Linux selects the global XPU pluggable allocator; Windows selects the
native Unified Runtime hook. A failure after either becomes live is fatal
because allocator ownership cannot be rolled back safely.

Run the portable provider and Linux source-contract tests inside the target
development container:

```bash
python -m pytest -q \
  tests/test_xpu_runtime_provider_wheel.py \
  tests/test_linux_xpu_source_contract.py
```
