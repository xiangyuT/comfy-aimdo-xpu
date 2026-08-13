# Documentation

This directory contains the current implementation and validation contract for
the AIMDO XPU backend. It intentionally does not retain dated benchmark runs,
failed hypotheses, machine-specific timing, or session history.

## Maintained documents

| Document | Scope |
| --- | --- |
| [XPU platform memory policy](XPU_PLATFORM_MEMORY_POLICY.md) | Linux and Windows allocator ownership, pressure, reclaim, and reserve invariants |
| [Windows Unified Runtime allocator hook](WINDOWS_XPU_UR_ALLOCATOR_HOOK.md) | Current Windows hook architecture, lifecycle, configuration, counters, and focused tests |
| [Windows build and validation](WINDOWS_XPU_BUILD_TEST_ACCEPTANCE.md) | Reproducible native build, runtime identity checks, and validation hierarchy |

## Scope

Each document states the behavior the code guarantees today and the checks that
prove it. When implementation behavior changes, update the affected contract in
place. Do not add dated benchmark runs, environment snapshots, or superseded
hypotheses to this directory.
