# XPU platform memory policy

Intel XPU uses one VBAR priority/fault model on Linux and Windows, but regular
PyTorch memory has different ownership on the two platforms. Pressure and
reclaim must preserve that ownership boundary.

## Allocator modes

| Platform | Provider default mode | Regular allocation owner | Pressure point |
| --- | --- | --- | --- |
| Linux | `native_hook` | PyTorch native XPU caching allocator | Unified Runtime physical USM growth at `urUSMDeviceAlloc` |
| Windows | `native_hook` | PyTorch native XPU caching allocator | Unified Runtime physical USM growth at `urUSMDeviceAlloc` |

Linux requires AIMDO to be interposed before Python starts. The standard OmniXPU
entrypoint prepares the verified provider preload automatically. An explicit
`AIMDO_XPU_ALLOCATOR_MODE=global` selects the AIMDO pluggable allocator instead.
The standalone `control.init()` API retains its Linux `global` default; a direct
native caller must preload the library and select `native_hook` explicitly.
Windows does not replace PyTorch's allocator.

An optional Linux Torch 2.14 diagnostic build may install the AIMDO native
owner proxy before XPU initialization when
`AIMDO_XPU_NATIVE_OWNER_DIAGNOSTIC=1` is explicitly set. Ordinary tensor
requests still delegate to PyTorch's native cache; selected compiler scopes
have a separate owner. This process-lifetime experiment does not change the
default `native_hook` route or public memory-compiler capability. The proxy's
private C10/XPU ABI, stream and failure contracts need separate validation
before it can become a supported allocator path.
The read-only `get_memory_compiler_capability()` query reports this path under
`native_owner_diagnostic`: process installation, active `native_hook` context,
its opt-in graph entry point and explicit `record_stream` consumer contract are
separate from public `available=false` and `xpu_consumer_tracking=false`.
The diagnostic is active only after `init_devices()` has established at least
one device context and enabled the native hook. It stays inactive during
initialization and from the start of hook teardown, including if teardown
fails; the graph and compiler-scope entry points use the same readiness check.
After `deinit()`, the process-lifetime proxy can remain installed while the
diagnostic context is inactive. This status query loads no new DSO and performs
no XPU allocation.

For the opt-in diagnostic, `native_owner.paused_graph_scope(graph)` pairs graph
pause/resume with temporary native Torch allocation routing, including on an
exceptional exit. Nested pauses retain the outer pause only for the same graph
and sync mode. `native_owner.suspend_compiler_scope()` remains the lower-level
route control for callers managing the graph themselves. Nested compiler scopes
are rejected; a failed native scope or graph transition is process-terminal.
These are private caller building blocks, not automatic ComfyUI integration.

The diagnostic sidecar treats a failed owner-map insertion after a fresh
compiler allocation as a rollback: the unregistered owner releases that new
allocation. A duplicate live compiler VA is different. Releasing the rejected
claim could free the existing tensor, so the sidecar preserves the old owner,
reports a fatal collision, and requires the process to exit after cleanup.
Neither rule enables public memory compilation or weakens the ordinary native
allocator's ownership.

For an explicit XPU stream switch inside a diagnostic memory graph, an owner
may be released after the graph has switched back to another stream. The
sidecar waits its allocation queue and each registered consumer, then uses
the XPU-only `malloc_graph_free_owned` entry point to apply the free event on
the owner's stream and restore the graph's previous stream. The original
`malloc_graph_free` keeps its strict current-stream check. Unregistered
external consumers remain outside this contract.
The opt-in `native_owner.consumer_scope(tensor, stream)` checks an indexed XPU
device and live compiler ownership at the tensor's storage base, so a view with
a nonzero storage offset can register its underlying compiler owner. It calls
Torch `record_stream` **before**
yielding to the secondary stream, then restores the previous stream on exit.
Callers must enter it before queuing that stream's first use of the tensor;
entering later cannot repair work that was already unregistered. Torch's
explicit stream registration or an equivalent caller-managed completion wait
is required before compiler backing may be reused. This diagnostic helper does
not discover arbitrary hidden consumers or enable public graph recording.
Concurrent diagnostic graph threads bind the sidecar's D1 native entry points
through one immutable function table. The first valid scope publishes that
table atomically; later scopes require byte-identical function addresses and
the expected source revision before routing allocations. Owner destructors
load only the published table, so a concurrent scope cannot rewrite the
functions used by an in-flight free.

The opt-in Torch 2.14 diagnostic also exposes a bounded per-graph physical-page
OOM injection. One failed attempt exercises the existing reclaim/retry path;
two failed attempts exercise terminal allocation failure and graph abort. This
hook does not simulate device pressure or establish a production OOM result.

Some Torch XPU operators request temporary buffers through `raw_alloc` while a
diagnostic compiler scope is active. The optional proxy delegates those buffers
to Torch's native allocator and tracks their release separately from compiler
`DataPtr` owners. The diagnostic exposes the scoped raw count and an exact live
compiler-owner query so operator probes can detect output routing gaps. This
path does not make unregistered consumers or arbitrary operators supported.

For the optional Torch 2.14 diagnostic graph, native destruction now reports
whether the owner thread actually released its graph. A close requested from
another Python thread is retained for the creator thread to drain; AIMDO
deinitialization refuses outstanding graph handles. This does not make active
graph cancellation or arbitrary cross-thread consumers public functionality.
If that creator later calls `close()` or repeats `abort()` on the same Python
graph object, it first drains its queued native close. A different graph method
also drains the queue before reporting that the object is closed. An owner-side
cleanup failure remains visible and retains the queued handle for retry or
process exit.
It also refuses live native-owner or scoped raw allocations after a graph has
handed an escaped compiler tensor to rogue ownership; final tensor free must
finish before native cleanup.

The optional XPU diagnostic can inject one failed graph-destroy release at a
completed graph's small mapped page, small physical page or virtual-range
stage. The checked destroy retains unfinished owners for an owner-thread retry.
This exercises its recovery contract without claiming a real driver unmap or
physical-release failure was observed.

For a separate completed-graph Linux diagnostic, the XPU VMM adapter can
return one OOM or device-lost code at the Level Zero unmap, physical destroy
or virtual free function-pointer boundary. The core and VMM manager then take
their normal error path. This tests metadata retention after earlier cleanup
stages have succeeded; it still does not mean the hardware driver itself
returned an error or mutated the targeted primitive before failing.
An OOM-style release error may be retried with retained graph ownership. A
generic or device-loss release error marks that diagnostic graph terminal:
subsequent close calls do not re-enter native release, and deinitialization
continues to refuse the live handle until process exit. The driver state after
a real device loss is not assumed recoverable.
The diagnostic queue also treats an exited graph creator as terminal. It keeps
the handle owned and prevents later scopes, graph creation and deinitialization
from silently proceeding; no surviving thread adopts a dead thread's native
graph. Process exit is required for that unresolved owner.
For a compiler tensor whose final Python reference is released on another
thread while its graph is still recording, the optional proxy waits its
allocation and registered consumer queues, then retains the native owner for
the original graph thread. The owner drains pending tensor frees before graph
pop, abort or destroy. Pending owners count as live at deinitialization; a
failed owner-thread free blocks further graph work and requires process exit.
This does not detect an unregistered asynchronous consumer.
The proxy also tracks whether a native compiler owner thread has exited while
its deferred-free queue is nonempty. This is independent of Python graph-object
tracking: an unclaimed owner cannot be adopted by a later thread with a reused
thread ID. New graph work and deinitialization fail closed until process exit,
even when Python no longer has a graph handle to inspect.

Linux native mode keeps Torch's allocator, statistics and cache-management APIs.
At model prioritization, Python publishes a cached-byte estimate to the hook.
A budget deficit uses Torch's cache-release retry only when that estimate is
positive; the hook consumes each estimate once. Without a positive estimate,
it attempts VBAR reclaim directly, avoiding a cache flush with no known cache.
The estimate is advisory: split or pending blocks may remain unreleasable.
After a real allocator OOM, retry accounts for tracked bytes actually returned
on the same device/context before choosing a residual reclaim amount.

The reverse direction runs at a failed VBAR fault, outside the UR callback.
It retries the fault once only if `torch.xpu.empty_cache()` reduced reserved
bytes. An unchanged unsuccessful cache state is suppressed until allocator
state changes. This Linux path does not inherit Windows WDDM time or size
thresholds. These mechanisms require target-local runtime and workflow
validation; a configured default alone makes no performance claim.

Before Linux native unpin, the actual Torch consumer queue is registered with
the existing synchronized reclaim path. Reclaim waits every registered queue.
Unknown queues and graph-capture consumers retain the pin and raise an error;
Linux native mode does not claim Windows external/capture lease support.

## Shared invariants

1. VBAR pages are reclaimed only when unpinned and safe to retire.
2. The active model has higher priority than older VBARs.
3. Actual allocation pressure may reduce active-model residency; speculative
   model-boundary reclaim must not.
4. Pressure uses pending physical growth, not logical tensor size or virtual
   reservation size.
5. A component probe cannot establish workload liveness or performance unless
   the changed pressure path executes in the same run.

## Linux allocation-time pressure

In `global` mode, an AIMDO allocator cache miss supplies the exact request to
the budget policy before allocating:

```text
PyTorch cache miss
  -> budget_deficit(requested bytes)
  -> reclaim eligible VBAR pages only when required
  -> sycl::malloc_device()
```

A cache hit creates no physical growth and performs no speculative reclaim.
Freed blocks remain cached per device and queue, and their completion state
protects reuse.

## Windows allocation-time pressure

Windows keeps PyTorch's splitting, coalescing, stream ordering, retry,
statistics, and `empty_cache()` behavior. AIMDO observes physical USM segments
at the Unified Runtime boundary.

The allocation hook follows two rules:

- it never waits on a SYCL queue or performs re-entrant driver memory
  management;
- it injects a synthetic allocation failure only for a PyTorch request when
  PyTorch has cached bytes that its normal retry can release.

After PyTorch releases its cache and retries, AIMDO may reclaim the residual
deficit from retirement-proven VBAR pages. Direct non-PyTorch SYCL callers are
never given the PyTorch-specific synthetic failure.

The hook handles the direction where PyTorch requests memory. A VBAR fault that
needs memory while PyTorch holds freed blocks does not call `urUSMDeviceAlloc`.
The rate-limited fault-boundary cache trim covers that reverse direction and
must run outside the hook.

See [Windows Unified Runtime allocator hook](WINDOWS_XPU_UR_ALLOCATOR_HOOK.md)
for lifecycle, configuration, and counters.

## Windows pressure accounting

DXGI/WDDM local-memory `CurrentUsage` and `Budget` are the sampled process-wide
baseline. This includes SYCL, oneDNN, driver, and other allocations not owned
by AIMDO. Between rate-limited DXGI samples, AIMDO applies the delta from
native allocations it observes.

When DXGI/LUID mapping is unavailable, the backend falls back to Level Zero
free-memory information. WDDM non-local usage is considered fallback pressure
only after the platform safety margin, so ordinary bookkeeping does not cause
continuous VBAR eviction.

## Retirement and model boundaries

Windows reclaim cannot create a queue fence at the moment memory is needed and
then wait for it. Pages publish per-queue retirement generations when they
become unpinned. Pressure closes a partial fence batch when necessary, but the
allocation path only compares completed generations and returns immediately.

Non-blocking two-phase retirement is the Windows default. Unpin publishes the
actual consumer queue before the page becomes idle. A pressure boundary closes
any partial fence batch, snapshots completed generations, freezes eligible
pages, and revalidates their handle, serial, eviction generation, and pin state
before unmapping. A marker submitted by the current pressure boundary is not
treated as complete in that same boundary. Setting
`AIMDO_XPU_ASYNC_VBAR_RECLAIM=0` selects a synchronized model-boundary oracle;
because it cannot bound pages touched within one activation, it is not a
product-performance mode.

At `prioritize()`, Windows may use PyTorch's observed peak-minus-current
reserved memory as a hint for expected native growth. This is speculative:
older VBARs may be reclaimed, but the newly active VBAR is preserved. Its full
fault range is reopened so a transient pressure window does not become a
permanent streaming ceiling.

Individual faults revalidate current pressure and can lower residency when the
requested pages do not fit. If pressure later clears, a request above the old
watermark may reopen the range and fault pages normally.

## Explicit consumer and capture ownership

The standard ComfyUI path submits the operator on the current Torch XPU stream
and unpins immediately after submission. AIMDO registers that current stream
automatically. A custom or external runtime may submit work to another queue or
retain the foreign VBAR pointer after the model pin is released; such work must
use an explicit ownership lease:

```python
from comfy_aimdo.model_vbar import vbar_external_consumer

with vbar_external_consumer(allocation, stream=consumer_stream):
    submit_external_kernel(weight_pointer)
```

The lease is acquired before submission, so pressure cannot unmap the page in
the interval before queue registration. Normal exit publishes the selected
queue dependency. Exceptional exit is fail-closed because AIMDO cannot know
whether an external runtime accepted part of a submission. The older
`vbar_register_consumer()` one-shot API remains valid only when the model pin
is still active through post-submission registration.

Graph capture has a longer lifetime than capture construction:

```python
from comfy_aimdo.model_vbar import vbar_capture_begin

capture_lease = vbar_capture_begin(allocation)
capture_graph()
for _ in range(replays):
    graph.replay()
# No future replay is permitted; the final replay is already submitted.
capture_lease.release(final_replay_stream)
```

While the capture lease is active, a capture-time unpin with no valid event
does not permanently poison the page; the capture hold itself prevents every
reclaim path. Release records the final replay queue before removing that hold.
An invalid release queue converts the page to permanent unknown/fail-closed
state. Forgetting to release a lease also remains fail-closed at teardown.

`control.get_xpu_memory_snapshot()` keeps ownership domains explicit:
PyTorch native allocator statistics and optional segment snapshots are under
`native_allocator`, while VMM counters, UR-hook counters and VBAR page state
are under `aimdo`. `ModelVBAR.snapshot()` reports pins, queue tokens, external
holds, capture holds, unknown state and eviction state without changing them.
Owner-boundary VBAR OOM decisions retain a compact, rate-limited snapshot in
`control.get_last_xpu_oom_snapshot()`; no Python snapshot is taken from the
allocator/UR callback.

## Reserve policy

ComfyUI passes `--reserve-vram` to AIMDO as `simple_vram_headroom`. On Windows,
AIMDO keeps at least:

```text
max(simple_vram_headroom, 512 MiB)
```

between complete-process DXGI `CurrentUsage` and `Budget`. The usage basis and
reserve target must come from the same accounting domain; AIMDO-only usage is
not sufficient on Windows.

On Linux, the reserve is evaluated at the exact allocator request against the
allocator/VBAR accounting available to the backend.

## Physical-page allocation retry

A VBAR physical-page request first reclaims the full value returned by
`budget_deficit()`, which may be larger than one page. If Level Zero still
reports physical allocation OOM, Windows performs one final 512 MiB reclaim
and retries once. This is an error-recovery margin, not the primary pressure
control loop.

## Small VBAR copy fallback

Some Windows multi-adapter configurations reject small host-to-VBAR copies on a
non-primary Level Zero node. `comfy_aimdo.torch.copy_to_vbar()` is the
supported way to write host data into AIMDO VBAR storage; integrations must use
it instead of calling `Tensor.copy_` directly.

The helper takes the fallback route only when the request is Windows XPU,
CPU source to XPU destination, contiguous with identical shape and dtype,
1 byte through 2 MiB, inside a fully mapped and pinned VBAR range, and on an
adapter whose Level Zero node mask is not `0x1`. It then stages through private
device memory padded past the affected size and writes exactly the requested
range with a kernel, so padding never touches VBAR. Every other request uses
the normal copy path.

`AIMDO_XPU_SMALL_VBAR_COPY_FALLBACK` forces the route on with `1` or off with
`0`. `control.get_xpu_vmm_stats()` reports `small_vbar_copy_fallback_calls`,
`small_vbar_copy_fallback_bytes`, and `small_vbar_copy_fallback_failures`.

## Minimal regression check

Run from an environment with the XPU backend built and Torch XPU available:

```powershell
python tests\repro_xpu_platform_memory_policy.py
```

The script creates a lower-priority and an active VBAR, applies controlled
pressure, and reports residency around the platform policy.

Windows retirement changes must additionally run
`tests/run_xpu_vbar_resident_growth.py`: the default 16-page/512 MiB test must
remain bounded at one resident page under deterministic live pressure. This
capacity gate precedes any ComfyUI workflow benchmark.
