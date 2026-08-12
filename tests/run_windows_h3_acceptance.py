"""Run the Windows AIMDO MiniMax H3 acceptance workload against ComfyUI.

Acceptance defaults to three sequential, uncached T2V prompts in one server
process: 1280x736, 362 frames at 24 fps (about 15 seconds), and two sampler
steps.  A different noise seed and output prefix are used for every run.
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import datetime as dt
import json
import pathlib
import time
import uuid

import aiohttp
import av


WIDTH = 1280
HEIGHT = 736
FRAMES = 362
FPS = 24.0
STEPS = 2  # default; override with --steps


def build_prompt(seed: int, filename_prefix: str, steps: int = STEPS,
                 width: int = WIDTH, height: int = HEIGHT,
                 frames: int = FRAMES) -> dict:
    return {
        "save": {
            "inputs": {
                "filename_prefix": filename_prefix,
                "format": "auto",
                "codec": "auto",
                "video": ["create_video", 0],
            },
            "class_type": "SaveVideo",
        },
        "video_vae": {
            "inputs": {"vae_name": "minimax_h3_video_vae_fp16.safetensors"},
            "class_type": "VAELoader",
        },
        "audio_vae": {
            "inputs": {"vae_name": "minimax_h3_audio_vae_fp32.safetensors"},
            "class_type": "VAELoader",
        },
        "decode_audio": {
            "inputs": {"samples": ["sampler", 0], "vae": ["audio_vae", 0]},
            "class_type": "VAEDecodeAudio",
        },
        "decode_video": {
            "inputs": {"samples": ["sampler", 0], "vae": ["video_vae", 0]},
            "class_type": "VAEDecode",
        },
        "sampler_select": {
            "inputs": {"sampler_name": "res_multistep"},
            "class_type": "KSamplerSelect",
        },
        "scheduler": {
            "inputs": {
                "scheduler": "simple",
                "steps": steps,
                "denoise": 1.0,
                "model": ["model", 0],
            },
            "class_type": "BasicScheduler",
        },
        "sampler": {
            "inputs": {
                "noise": ["noise", 0],
                "guider": ["guider", 0],
                "sampler": ["sampler_select", 0],
                "sigmas": ["scheduler", 0],
                "latent_image": ["conditioning", 1],
            },
            "class_type": "SamplerCustomAdvanced",
        },
        "guider": {
            "inputs": {
                "model": ["model", 0],
                "conditioning": ["conditioning", 0],
            },
            "class_type": "BasicGuider",
        },
        "model": {
            "inputs": {
                "unet_name": "minimax_h3_fl2va_pruned_int8_convrot.safetensors",
                "weight_dtype": "default",
            },
            "class_type": "UNETLoader",
        },
        "clip": {
            "inputs": {
                "clip_name": "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
                "type": "minimax",
                "device": "default",
            },
            "class_type": "CLIPLoader",
        },
        "noise": {
            "inputs": {"noise_seed": seed},
            "class_type": "RandomNoise",
        },
        "create_video": {
            "inputs": {
                "fps": FPS,
                "bit_depth": 8,
                "images": ["decode_video", 0],
                "audio": ["decode_audio", 0],
            },
            "class_type": "CreateVideo",
        },
        "conditioning": {
            "inputs": {
                "prompt": (
                    "Cinematic live-action footage of a futuristic city after rain. "
                    "A runner crosses rooftops at dusk while flying vehicles pass "
                    "between towers. Natural motion, realistic lighting, coherent "
                    "camera movement, stereo city ambience and music; no text or logos."
                ),
                "width": width,
                "height": height,
                "length": frames,
                "clip": ["clip", 0],
                "vae": ["video_vae", 0],
            },
            "class_type": "MiniMaxH3ImageToVideo",
        },
    }


async def get_json(session: aiohttp.ClientSession, url: str) -> dict:
    async with session.get(url) as response:
        response.raise_for_status()
        return await response.json()


async def wait_for_history(
    session: aiohttp.ClientSession,
    base_url: str,
    prompt_id: str,
    timeout: float,
) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        history = await get_json(session, f"{base_url}/history/{prompt_id}")
        record = history.get(prompt_id)
        if record is not None:
            status = record.get("status", {})
            if status.get("completed") or status.get("status_str") in {
                "success",
                "error",
            }:
                return record
        await asyncio.sleep(2)
    raise TimeoutError(f"prompt {prompt_id} exceeded {timeout:.0f} seconds")


def find_artifacts(output_root: pathlib.Path, basename: str) -> list[dict]:
    files = []
    for path in output_root.rglob(f"{basename}*"):
        if path.is_file():
            files.append(
                {
                    "path": str(path),
                    "size": path.stat().st_size,
                }
            )
    return sorted(files, key=lambda item: item["path"])


def inspect_video(path: pathlib.Path, width: int = WIDTH, height: int = HEIGHT,
                  frames: int = FRAMES) -> dict:
    with av.open(str(path)) as container:
        stream = next((item for item in container.streams if item.type == "video"), None)
        if stream is None:
            raise RuntimeError(f"{path} has no video stream")
        fps = float(stream.average_rate) if stream.average_rate is not None else 0.0
        duration = (
            float(stream.duration * stream.time_base)
            if stream.duration is not None and stream.time_base is not None
            else 0.0
        )
        details = {
            "width": stream.codec_context.width,
            "height": stream.codec_context.height,
            "fps": fps,
            "frames": stream.frames,
            "duration_seconds": duration,
        }

    expected_duration = frames / FPS
    if details["width"] != width or details["height"] != height:
        raise RuntimeError(f"unexpected video dimensions for {path}: {details}")
    if abs(details["fps"] - FPS) > 1e-6:
        raise RuntimeError(f"unexpected frame rate for {path}: {details}")
    if details["frames"] != frames:
        raise RuntimeError(f"unexpected frame count for {path}: {details}")
    if abs(details["duration_seconds"] - expected_duration) > (1.0 / FPS):
        raise RuntimeError(f"unexpected video duration for {path}: {details}")
    return details


async def run(args: argparse.Namespace) -> list[dict]:
    base_url = args.server.rstrip("/")
    client_id = f"aimdo-acceptance-{uuid.uuid4().hex}"
    output_root = pathlib.Path(args.output_root).resolve()
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    results = []

    timeout = aiohttp.ClientTimeout(total=None, sock_connect=30, sock_read=None)
    async with aiohttp.ClientSession(timeout=timeout, trust_env=False) as session:
        stats = await get_json(session, f"{base_url}/system_stats")
        print("SYSTEM", json.dumps(stats, ensure_ascii=False, separators=(",", ":")))

        for run_index in range(1, args.runs + 1):
            seed = args.seed + run_index - 1
            basename = f"aimdo_h3_720p_15s_2step_{stamp}_run{run_index}"
            filename_prefix = f"video/{basename}"
            prompt = copy.deepcopy(build_prompt(seed, filename_prefix, args.steps,
                                        args.width, args.height, args.frames))
            started = time.monotonic()
            async with session.post(
                f"{base_url}/prompt",
                json={"prompt": prompt, "client_id": client_id},
            ) as response:
                body = await response.json()
                if response.status >= 400:
                    raise RuntimeError(
                        f"prompt submission failed ({response.status}): {body}"
                    )
            prompt_id = body["prompt_id"]
            print(
                f"RUN_START index={run_index} prompt_id={prompt_id} seed={seed} "
                f"width={args.width} height={args.height} frames={args.frames} fps={FPS:g} "
                f"steps={args.steps}",
                flush=True,
            )

            record = await wait_for_history(
                session, base_url, prompt_id, args.timeout_seconds
            )
            elapsed = time.monotonic() - started
            status = record.get("status", {})
            status_str = status.get("status_str")
            artifacts = find_artifacts(output_root, basename)
            videos = []
            for artifact in artifacts:
                path = pathlib.Path(artifact["path"])
                if path.suffix.lower() == ".mp4" and artifact["size"] > 0:
                    videos.append({**artifact, "media": inspect_video(
                    path, args.width, args.height, args.frames)})
            result = {
                "run": run_index,
                "prompt_id": prompt_id,
                "seed": seed,
                "elapsed_seconds": elapsed,
                "status": status_str,
                "completed": status.get("completed"),
                "artifacts": artifacts,
                "videos": videos,
                "messages": status.get("messages", []),
            }
            print("RUN_RESULT", json.dumps(result, ensure_ascii=False), flush=True)
            if status_str != "success" or not status.get("completed"):
                raise RuntimeError(f"run {run_index} did not succeed: {result}")
            if not videos:
                raise RuntimeError(
                    f"run {run_index} produced no conforming non-empty MP4: {result}"
                )
            results.append(result)

    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", default="http://127.0.0.1:8188")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--steps", type=int, default=STEPS,
                        help="sampler steps; the 2-step profile does not "
                             "reach memory pressure")
    parser.add_argument("--width", type=int, default=WIDTH)
    parser.add_argument("--height", type=int, default=HEIGHT)
    parser.add_argument("--frames", type=int, default=FRAMES,
                        help="MiniMax H3 requires 17k+5")
    parser.add_argument("--seed", type=int, default=874633819053040)
    parser.add_argument("--timeout-seconds", type=float, default=4 * 60 * 60)
    args = parser.parse_args()
    results = asyncio.run(run(args))
    print("ACCEPTANCE_PASS", json.dumps(results, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
