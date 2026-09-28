# Linux native-owner memory compiler prototype

`proxy.cpp` is a **diagnostic-only** Torch XPU allocator proxy for AIMDO 0.5.5 on Linux B70. It is not linked into `aimdo_xpu.so` or included in the provider wheel. Public XPU `record()` and compiler capability remain unavailable.

The proxy is installed before XPU device initialization in a disposable process. It forwards ordinary tensor allocations to Torch's native caching allocator, wraps the native `DataPtr` in a stable owner/deleter map, and dispatches a selected 4,097-byte scope to the existing AIMDO D1 `malloc_graph_alloc/free` ABI. Native and compiler pointers retain separate owners. Registered consumer streams are fenced before compiler free; an escaped output can be freed from another thread through the core's rogue handoff.

The [target-local B70 experiment](https://github.com/xiangyuT/omni-xpu-kernel-tuning/tree/main/experiments/targets/bmg-g31/aimdo-native-compiler-va-proxy-055) owns the executable container admission, four full-payload record/replay checks, cross-thread escape case, exact source/driver/image bindings and host validator. This source file is intended to match the experiment's reviewed prototype bytes exactly. A task-owned development container is restricted to host XPU 0; XPU 1 is user-allowed but has no result here.

The proxy uses Torch 2.13 C10 allocator registry and the XPU allocator pointer exposed in that exact header. These are not established as a stable product ABI. The selected size, graph and registered consumer case do not establish unregistered stream, abort, pressure, arbitrary tensor shapes, ComfyUI caller, public packaging or performance support. No Torch source file or package version is changed by this experiment.
