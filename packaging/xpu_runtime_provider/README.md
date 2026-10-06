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
artifact and `aimdo_xpu_native_owner_abi.json` with SHA-256 for the C10,
C10/XPU and Torch/XPU libraries used by the build. The provider manifest binds
both files. Installation checks the runtime Torch CXX11 ABI flag and these
three library bytes before loading the sidecar; a same-version different build
fails closed. The diagnostic remains disabled by default. Only an explicitly
set `AIMDO_XPU_NATIVE_OWNER_DIAGNOSTIC=1` with Linux
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
The opt-in Python `native_owner.consumer_scope(tensor, stream)` verifies a live
compiler-owned storage base on the same indexed XPU device, including views
with nonzero storage offsets, and registers the stream
before yielding to queued work. It is a caller contract, not automatic
tracking: previously queued unregistered work remains outside the guarantee.
The private `native_owner.suspend_compiler_scope()` context temporarily sends
allocations through Torch's ordinary XPU allocator while a caller pauses its
diagnostic graph. The caller resumes the graph before leaving the context;
nested suspensions are supported, nested compiler scopes are rejected, and a
failed native scope transition requires process exit. ComfyUI caller wiring
remains a separate gate.
`native_owner.paused_graph_scope(graph)` pairs that route suspension with
graph pause/resume and restores both on exceptional exit. Nested calls require
the same graph and sync mode; a failed graph transition requires process exit.
The Torch 2.14 sidecar publishes its D1 function pointers once as an immutable
table. Later compiler scopes verify the same native addresses and source
revision; concurrent graph threads do not overwrite pointers read by tensor
allocation or final release.
Its additional `malloc_graph_test_fail_next_page_creates` export supports a
bounded, active-graph-only synthetic OOM diagnostic. One rejected page attempt
exercises retry; two rejected attempts exercise failure propagation. Neither
case establishes behavior under real device memory pressure.
The sidecar also exposes diagnostic scoped-raw counters and a live
compiler-owner query. Backend `raw_alloc` workspaces used inside an opt-in
compiler scope remain owned by Torch's native allocator; the compiler owns
only positive-size tensor requests served through its `DataPtr` route.
The wheel builder requires the complete deferred-free and lifecycle transition
exports as well as the matching core allocation, ownership and retirement
exports. An older or incomplete sidecar cannot pass this check solely by
matching the distribution and Torch versions.
The diagnostic graph also requires `malloc_graph_destroy_checked` from its
matching D1 library. A foreign-thread close is queued to the graph's creator;
deinitialization fails while any graph handle remains live or deferred.
The creator's later `close()` or repeated `abort()` on that graph drains the
queued close before returning; cleanup failures still propagate and retain
native ownership.
An escaped compiler tensor or scoped raw workspace must also release its owner
before deinitialization can clean native state.
The matching D1 library additionally exposes a completed-graph-only synthetic
destroy-release failure hook for mapped, physical and virtual owner retry
diagnostics. It is not a real driver failure or a public allocator API.
The Linux diagnostic can separately inject one error return through the XPU
VMM adapter's Level Zero release function pointers. OOM and device-lost codes
exercise the normal VMM/core error propagation while the actual driver stays
healthy.
The checked XPU destroy reports a terminal state for generic/device-loss
release errors: the graph handle remains owned and no second native release is
attempted in that process. OOM-style failures retain the retry route.
If a diagnostic graph's creator thread has exited, foreign-thread close still
retains the handle, but reports a terminal process-exit requirement immediately.
No other thread adopts that native graph or continues allocator scopes.
If a compiler tensor is finally released on a foreign thread before the graph
has completed, its DataPtr waits registered consumers and defers the native
free to the original graph thread. Graph operations drain those frees before
pop/abort/destroy; deinitialization counts outstanding deferred owners as live.
A failed deferred free is terminal, while unregistered consumers remain outside
the diagnostic contract.
The sidecar's unique native thread identity marks a pending owner terminal if
its original thread exits. The Python guard checks this state before later
graph work or deinitialization, including when no Python graph handle remains.
No other thread drains that queue; process exit is required.

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

The additive `native_owner_diagnostic` field reports whether the private proxy
is installed and active, and names its explicit `record_stream` caller contract.
It never makes public `available` true, including after an opt-in install.
Active requires completed `init_devices()` and a successfully enabled native
hook; the private graph/scope entry points reject pre-device and teardown
contexts. Hook teardown clears active before native calls and stays inactive
after a failed teardown attempt.
The query remains read-only before initialization and after `deinit()`.

Linux queue dispatch validates the Level Zero context and device before binding
or submitting work. It retains owned SYCL queue handles, including after the
caller's queue wrapper is released, and snapshots the binding under a mutex
before context synchronization. Queue waits run after the registry lock is
released. These dispatch rules do not enable public memory compilation.

The diagnostic compiler retains the last allocation driver error per thread.
Its matching sidecar translates device-memory allocation failures to Torch's
`OutOfMemoryError`, so callers can recognize OOM independently of unsupported
scope or other driver errors. Abort/owner cleanup remains required after a
failed allocation; this does not change public compiler availability.

The Linux Torch 2.14 diagnostic closes contexts under a lifecycle gate. New
logical allocations, raw allocations, scopes and graph creation are excluded
during this transition; live owners and native cache storage still prevent it.
Native graph handles are counted independently of Python graph references.

After those checks, remaining caller-owned UR buffers can move to a non-owning
retired ledger. AIMDO's direct USM allocations are explicitly marked and cannot
be retired. Retirement does not free an external buffer or increment physical
free counters. A late free continues through the original UR interface without
accessing the destroyed context. On re-initialization, surviving records are
adopted into the new budget before the hook becomes active. Separate retirement
statistics report this lifetime, including external oneDNN cache storage.
The default hook path without the diagnostic retains its strict disable rule.

An unrestricted diagnostic compiler scope excludes another queue on the same
device/context through the ordinary native allocator, retaining its native
owner and cache behavior. Exact-size scopes and foreign device/context queues
remain strict. A replay root with only successfully excluded requests preserves
its previous event tree and increments `MallocGraph.skipped_replays`; this is
not counted as compiler replay. Partial compiler sequences and ordinary missing
allocation frames keep the existing break rules. Cross-queue input consumers
still require explicit `record_stream` registration before work is submitted.

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
