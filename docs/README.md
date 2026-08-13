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
| [Windows second-GPU small-copy workaround](WINDOWS_XPU_SECOND_GPU_COPY_FAILURE.md) | Current driver workaround, selection boundary, diagnostics, and removal gate |

The automated MiniMax H3 prompt/media contract is maintained next to its
runner in [`tests/WINDOWS_H3_ACCEPTANCE.md`](../tests/WINDOWS_H3_ACCEPTANCE.md).

## Historical evidence

Development history belongs in
[`omni-xpu-kernel-tuning`](https://github.com/xiangyuT/omni-xpu-kernel-tuning),
the cross-machine source of truth for experiments, rejected hypotheses,
machine-specific observations, and dated validation results. Windows AIMDO
records are indexed under:

- `docs/results/bmg/2026-08-12/` for allocator interception, pressure, and
  liveness work;
- `docs/results/bmg/2026-08-13/` for the second-adapter small-copy workaround
  and physical-GPU performance variance.

When implementation behavior changes, update the maintained contract here and
append new revision-specific evidence to the tuning repository. Do not copy a
dated result back into this directory as a current guarantee.
