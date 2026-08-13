# Windows XPU physical GPU 0 performance variance

Status: **variance isolated to an external WDDM contention window; exact
indirect-display owner still needs a local-console controlled A/B test.**

## Finding

The latency spikes on physical GPU 0 are present in a pure PyTorch workload
with AIMDO never initialized. Physical GPU 1 runs the identical workload with
almost no spread. This excludes the AIMDO allocator hook, VBAR, and reclaim
policy as necessary causes.

The active WDDM counters simultaneously show a protected `WUDFHost.exe`
process using roughly 54-65% of the engine on GPU 0's LUID
`00000000:00016072`. The system has both a ToDesk virtual display adapter
(`tdIdd`) and a Microsoft Remote Display Adapter (`RdpIdd`). Because Windows
does not expose the protected UMDF process's command line or modules here, the
measurement proves process/LUID contention but does not identify which one of
those two indirect-display drivers owns that process.

Do not disable either display adapter during an active remote session: doing so
can disconnect the operator. The remaining attribution test must be performed
from a local console.

## Measurement

Run the benchmark without AIMDO:

```powershell
<portable>\python_embeded\python.exe -s tests\benchmark_windows_xpu_step.py `
    --device 0 --no-aimdo --iterations 300
<portable>\python_embeded\python.exe -s tests\benchmark_windows_xpu_step.py `
    --device 1 --no-aimdo --iterations 300
```

Observed during the contended window on driver `32.0.101.8515`:

| Metric | Physical GPU 0 | Physical GPU 1 |
| --- | ---: | ---: |
| mean | 8.990 ms | 4.062 ms |
| p50 | 3.657 ms | 4.089 ms |
| p90 | 30.418 ms | 4.126 ms |
| p95 | 35.125 ms | 4.135 ms |
| p99 | 62.195 ms | 4.180 ms |
| maximum | 65.770 ms | 4.291 ms |
| samples over 10 ms | 49 / 300 | 0 / 300 |

GPU 0's slightly faster median shows the compute kernel itself is healthy.
The high mean comes from frequent 30-65 ms preemption/queueing outliers. GPU 1
provides the clean control.

Later, without changing AIMDO or the benchmark, the `WUDFHost.exe` engine load
fell below the counter's 1% reporting threshold. A fresh 300-iteration run then
gave:

| Metric | Physical GPU 0, WUDF idle | Physical GPU 1 |
| --- | ---: | ---: |
| mean | 3.993 ms | 4.046 ms |
| p50 | 4.008 ms | 4.095 ms |
| p99 | 4.171 ms | 4.133 ms |
| maximum | 4.208 ms | 4.170 ms |
| samples over 10 ms | 0 / 300 | 0 / 300 |

This natural before/after observation is strong evidence that the transient
WDDM engine owner causes the variance. It is not a controlled driver-toggle
experiment, so it cannot distinguish RDP from ToDesk.

## Adapter mapping

The physical adapters are:

| Torch/Level Zero adapter | PCI | LUID |
| --- | --- | --- |
| physical GPU 0 | `04:00.0` | `00000000:00016072` |
| physical GPU 1 | `31:00.0` | `00000000:000163ad` |

DXGI also enumerates indirect/logical display adapters, so ordinal-only WDDM
attribution is unsafe. Match counters by LUID.

## Recommended resolution test

From a local console, with no ComfyUI or other GPU workload:

1. record the two 300-iteration baselines;
2. disconnect Remote Desktop and exit ToDesk;
3. disable one indirect display adapter at a time;
4. confirm that the `WUDFHost.exe` engine usage on LUID `00016072` disappears;
5. rerun GPU 0 after each change;
6. re-enable any required display adapter.

The issue is resolved operationally when GPU 0 has no persistent indirect
display engine load and its p95/p99 collapse near its median. Until that A/B is
performed, physical GPU 1 is the reliable compute adapter on this machine.
