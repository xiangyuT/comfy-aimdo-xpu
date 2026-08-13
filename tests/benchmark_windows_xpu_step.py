"""Measure per-step latency and outliers on a selected Windows XPU adapter.

The workload uses only PyTorch when ``--no-aimdo`` is set, which separates
system/driver scheduling variance from AIMDO allocation behavior.

    <portable>\\python_embeded\\python.exe -s tests\\benchmark_windows_xpu_step.py \\
        --device 0 --no-aimdo --iterations 300
"""

import argparse
import json
import pathlib
import statistics
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

MIB = 1024 * 1024


def build_workload(torch, device, hidden, batch):
    weights = [
        torch.randn(hidden, hidden, dtype=torch.float16, device=device)
        for _ in range(4)
    ]
    activation = torch.randn(batch, hidden, dtype=torch.float16, device=device)
    return weights, activation


def step(torch, weights, activation):
    result = activation
    for weight in weights:
        result = torch.nn.functional.gelu(result @ weight)
    return result


def percentile(values, percentage):
    ordered = sorted(values)
    index = round((len(ordered) - 1) * percentage / 100)
    return ordered[index]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--hidden", type=int, default=4096)
    parser.add_argument("--batch", type=int, default=4096)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--no-aimdo", action="store_true")
    parser.add_argument("--reserve-mib", type=int, default=512)
    parser.add_argument("--json", default=None)
    arguments = parser.parse_args()

    control = None
    if not arguments.no_aimdo:
        from comfy_aimdo import control as aimdo_control

        if not aimdo_control.init(
            implementation="xpu",
            simple_vram_headroom=arguments.reserve_mib * MIB,
        ):
            print("control.init() failed", file=sys.stderr)
            return 1
        control = aimdo_control

    import torch

    if control is not None and not control.init_devices([arguments.device]):
        print("control.init_devices() failed", file=sys.stderr)
        return 1

    device = torch.device("xpu", arguments.device)
    mode = "no-aimdo" if arguments.no_aimdo else (
        control.get_xpu_allocator_mode() or "aimdo"
    )
    print(f"mode={mode} device={arguments.device} torch={torch.__version__} "
          f"hidden={arguments.hidden} batch={arguments.batch}")

    weights, activation = build_workload(
        torch, device, arguments.hidden, arguments.batch
    )
    for _ in range(arguments.warmup):
        step(torch, weights, activation)
    torch.xpu.synchronize(arguments.device)

    samples = []
    for _ in range(arguments.iterations):
        started = time.perf_counter()
        step(torch, weights, activation)
        torch.xpu.synchronize(arguments.device)
        samples.append((time.perf_counter() - started) * 1000)

    summary = {
        "mode": mode,
        "device": arguments.device,
        "iterations": len(samples),
        "mean_ms": statistics.mean(samples),
        "p50_ms": percentile(samples, 50),
        "p90_ms": percentile(samples, 90),
        "p95_ms": percentile(samples, 95),
        "p99_ms": percentile(samples, 99),
        "minimum_ms": min(samples),
        "maximum_ms": max(samples),
        "spread_ms": max(samples) - min(samples),
        "over_10ms": sum(value > 10 for value in samples),
        "per_iteration_ms": samples,
    }
    print(json.dumps({key: value for key, value in summary.items()
                      if key != "per_iteration_ms"}, indent=2))

    if arguments.json:
        pathlib.Path(arguments.json).write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )

    if control is not None:
        control.deinit()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
