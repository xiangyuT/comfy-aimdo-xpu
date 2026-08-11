"""Replay a prompt from a ComfyUI history export in one server process."""

from __future__ import annotations

import argparse
import asyncio
import copy
import json
import time
import uuid
from pathlib import Path

import aiohttp


def load_prompt(path: Path) -> dict:
    history = json.loads(path.read_text(encoding="utf-8"))
    if not history:
        raise RuntimeError(f"history export is empty: {path}")
    record = next(iter(history.values()))
    return record["prompt"][2]


def make_uncached(
    prompt: dict,
    run_index: int,
    megapixels: float | None = None,
    duration_seconds: float | None = None,
    steps: int | None = None,
) -> tuple[int, str]:
    seed = 917_000_000_000_000 + run_index
    prefix = f"video/aimdo_saved_prompt_run{run_index}_{uuid.uuid4().hex[:8]}"
    found_seed = False
    found_output = False
    for node in prompt.values():
        class_type = node.get("class_type")
        if class_type == "RandomNoise":
            node["inputs"]["noise_seed"] = seed
            found_seed = True
        elif class_type == "SaveVideo":
            node["inputs"]["filename_prefix"] = prefix
            found_output = True
        elif class_type == "ResolutionSelector" and megapixels is not None:
            node["inputs"]["megapixels"] = megapixels
        elif class_type == "PrimitiveFloat" and duration_seconds is not None:
            node["inputs"]["value"] = duration_seconds
        elif class_type == "BasicScheduler" and steps is not None:
            node["inputs"]["steps"] = steps
    if not found_seed or not found_output:
        raise RuntimeError("saved prompt lacks RandomNoise or SaveVideo")
    return seed, prefix


async def wait_for_result(session, server: str, prompt_id: str, timeout: float):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        async with session.get(f"{server}/history/{prompt_id}") as response:
            response.raise_for_status()
            record = (await response.json()).get(prompt_id)
        if record is not None:
            status = record.get("status", {})
            if status.get("completed") or status.get("status_str") == "error":
                return record
        await asyncio.sleep(2)
    raise TimeoutError(f"prompt {prompt_id} exceeded {timeout:.0f} seconds")


async def run(args):
    server = args.server.rstrip("/")
    base_prompt = load_prompt(args.history_json)
    timeout = aiohttp.ClientTimeout(total=None, sock_connect=30, sock_read=None)
    async with aiohttp.ClientSession(timeout=timeout, trust_env=False) as session:
        for run_index in range(1, args.runs + 1):
            prompt = copy.deepcopy(base_prompt)
            seed, prefix = make_uncached(
                prompt,
                run_index,
                megapixels=args.megapixels,
                duration_seconds=args.duration_seconds,
                steps=args.steps,
            )
            started = time.monotonic()
            async with session.post(
                f"{server}/prompt",
                json={"prompt": prompt, "client_id": "aimdo-saved-replay"},
            ) as response:
                body = await response.json()
                if response.status >= 400:
                    raise RuntimeError(
                        f"prompt submission failed ({response.status}): {body}"
                    )
            prompt_id = body["prompt_id"]
            print(
                f"RUN_START index={run_index} prompt_id={prompt_id} "
                f"seed={seed} prefix={prefix}",
                flush=True,
            )
            record = await wait_for_result(
                session, server, prompt_id, args.timeout_seconds
            )
            elapsed = time.monotonic() - started
            status = record.get("status", {})
            result = {
                "run": run_index,
                "prompt_id": prompt_id,
                "elapsed_seconds": elapsed,
                "status": status.get("status_str"),
                "completed": status.get("completed"),
                "outputs": record.get("outputs", {}),
                "messages": status.get("messages", []),
            }
            print("RUN_RESULT", json.dumps(result, ensure_ascii=False), flush=True)
            if result["status"] != "success" or not result["completed"]:
                raise RuntimeError(f"run {run_index} failed: {result}")
    print("REPLAY_PASS", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--history-json", type=Path, required=True)
    parser.add_argument("--server", default="http://127.0.0.1:8188")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--timeout-seconds", type=float, default=4 * 60 * 60)
    parser.add_argument("--megapixels", type=float)
    parser.add_argument("--duration-seconds", type=float)
    parser.add_argument("--steps", type=int)
    asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    main()
