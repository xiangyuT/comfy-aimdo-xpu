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

For the opt-in Linux B70 Torch 2.14 native-owner diagnostic, build from an
environment with the exact official `torch==2.14.0+xpu` wheel and oneAPI C++20
compiler:

```bash
UR_INCLUDE_DIR=/opt/intel/oneapi/compiler/2026.1/include/unified-runtime \
AIMDO_XPU_BUILD_NATIVE_OWNER_DIAGNOSTIC=1 \
AIMDO_TORCH_PYTHON=/opt/venv/bin/python \
bash scripts/build-linux-xpu.sh

SETUPTOOLS_SCM_PRETEND_VERSION_FOR_COMFY_AIMDO=0.5.5 \
/opt/venv/bin/python -m build --wheel --no-isolation --outdir dist

python packaging/xpu_runtime_provider/build_wheel.py \
  --source-wheel dist/comfy_aimdo-0.5.5-cp39-abi3-linux_x86_64.whl \
  --output-dir dist/provider \
  --source-revision "$(git rev-parse HEAD)" \
  --torch-version 2.14.0+xpu \
  --xpu-target bmg
```

This adds `aimdo_xpu_native_owner.so` as a second, privately vendored native
artifact. The manifest binds it to Torch 2.14.0 XPU and marks it disabled by
default. Only an explicitly set `AIMDO_XPU_NATIVE_OWNER_DIAGNOSTIC=1` with Linux
`native_hook` can install it before XPU initialization. The process-lifetime
proxy cannot be unloaded or switched off in the same process. This diagnostic
path does not enable public XPU `record()` or compiler capability; it has not
passed provider package or ComfyUI acceptance solely by being built.
`native_owner.inject_next_compiler_owner_insert_failure()` and
`native_owner.inject_duplicate_compiler_pointer()` exist only for bounded
error-path diagnostics. The latter is fatal to the process after it has
preserved the previously live owner; callers must not resume normal work.
The sidecar build also requires the XPU-only `malloc_graph_free_owned` export
in the matching D1 native library. It is used only after owner and registered
consumer queues complete; the standard shared-core free API is unchanged.
Its additional `malloc_graph_test_fail_next_page_creates` export supports a
bounded, active-graph-only synthetic OOM diagnostic. One rejected page attempt
exercises retry; two rejected attempts exercise failure propagation. Neither
case establishes behavior under real device memory pressure.

For a Linux source wheel containing `malloc_graph.py`, the builder requires the
complete provider module set and all twelve compiler ABI/provenance exports in
`aimdo_xpu.so`. This D1 source declares compatibility only with the reviewed
official `0.5.5` API, imported at
`3b8e8c162efeb9470d912609a7a6e7a2b1c693ec`. The provider distribution
keeps the source wheel's existing version. Source revision and native content
hashes identify development changes independently of that version. Windows
provider packaging retains its previous exact-version behavior until its native
compiler build receives separate validation.

`control.get_memory_compiler_capability()` reports the built core separately
from runtime availability. XPU recording remains unavailable until a logical
allocation router and its lifetime contract are implemented and validated;
explicit `record(xpu_stream)` raises an unsupported error. `malloc_graph` and
`control` must resolve to the same provider directory. A missing local module or
native ABI prevents Linux XPU initialization before allocator installation.

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
