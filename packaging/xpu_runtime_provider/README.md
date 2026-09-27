# AIMDO XPU runtime provider wheel

This builder converts an already-built XPU `comfy-aimdo` wheel into the
co-installable `comfy-aimdo-xpu-runtime` provider distribution. The provider
owns only `comfy_aimdo_xpu_runtime`; Python modules and the Level Zero native
library are stored below its private `_vendor` directory. It never installs a
top-level `comfy_aimdo` file, so the official AIMDO distribution remains
independently upgradeable.

After building `comfy_aimdo/aimdo_xpu.so` and the canonical 0.5.5 wheel, run:

```bash
python packaging/xpu_runtime_provider/build_wheel.py \
  --source-wheel dist/comfy_aimdo-0.5.5-cp39-abi3-linux_x86_64.whl \
  --output-dir dist/provider \
  --source-revision "$(git rev-parse HEAD)" \
  --torch-version 2.13.0+xpu \
  --xpu-target bmg
```

The output contains a lightweight
`comfyui_omnixpu.runtime_providers` entry point and a manifest covering the
canonical version, exact source revision, source-wheel hash, native-library
hash, supported runtime, and allocator modes. Importing its metadata does not
import PyTorch or AIMDO. The source and provider wheels both use version 0.5.5.

The provider retains its Linux and Windows runtime declarations. Local 0.5.5
build and install validation covers Linux only. The Windows XPU build script
now includes upstream `disk-id.c` and its required libraries, providing the
`aimdo_storage_fast_disk` source for `storage.fast_disk()`. A Windows build
and runtime test are still required before this 0.5.5 API and the existing
Windows allocator route can be claimed as validated.

ComfyUI-OmniXPU activates this provider only when DynamicVRAM is explicitly
enabled and the official AIMDO attempt has left no live native or allocator
state. The provider defaults to `native_hook` on Linux and Windows, keeping
PyTorch's native XPU allocator. Linux also supports an explicit
`AIMDO_XPU_ALLOCATOR_MODE=global` override.
The manifest records supported modes separately from platform defaults.
Linux native mode needs its verified provider DSO in `LD_PRELOAD` before Python
starts. The standard OmniXPU entrypoint resolves the provider default and
prepares this automatically. Direct Python launchers must prepare the preload
before startup as well. The standalone `control.init()` API retains its Linux
`global` default because it cannot add a preload to an already running process.
A failure after allocator or native state becomes live is fatal
because allocator ownership cannot be rolled back safely.

Linux native VBAR recovery retries only after Torch actually returns reserved
cache bytes. It restores the watermark from immediately before the failed fault,
then repeats the normal pressure checks once; a caller's earlier watermark limit
is preserved. This does not select Windows budget or retirement policy.

Run the portable provider and Linux source-contract tests inside the target
development container:

```bash
python -m pytest -q \
  tests/test_xpu_runtime_provider_wheel.py \
  tests/test_linux_xpu_source_contract.py
```
