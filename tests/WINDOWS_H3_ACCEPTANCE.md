# Windows XPU MiniMax H3 acceptance

## Workload

The Windows AIMDO milestone is accepted with a MiniMax H3 text-to-video
workload at the model's 720p-class aligned size:

* Resolution: 1280x736
* Length: 362 frames at 24 fps (15.083 seconds)
* Sampler steps: 2
* Repetitions: 3 sequential prompts with different seeds
* Process lifetime: one ComfyUI process for all three prompts

Do not restart ComfyUI or manually clear its caches between runs.

## Models

The test requires these files in the normal ComfyUI model directories:

* `minimax_h3_fl2va_pruned_int8_convrot.safetensors`
* `qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors`
* `minimax_h3_video_vae_fp16.safetensors`
* `minimax_h3_audio_vae_fp32.safetensors`

## Execution

Start ComfyUI with DynamicVRAM enabled. From the AIMDO repository, run the
acceptance client with the Portable Python environment:

```powershell
<portable>\python_embeded\python.exe tests\run_windows_h3_acceptance.py `
    --server http://127.0.0.1:8188 `
    --output-root <portable>\ComfyUI\output
```

## Pass criteria

The test passes only when:

1. All three prompts report `execution_success`.
2. Every output is a non-empty MP4 with 1280x736 resolution, 24 fps, 362
   frames, and 15.083 seconds duration.
3. The sampler is executed for every prompt; changing the seed prevents the
   generated result from being served from the ComfyUI cache. Static model and
   VAE loader nodes may remain cached between prompts.
4. The ComfyUI server log for the complete three-run window contains no AIMDO
   copy failure (`result=999`), null-copy error, traceback, or device-reset
   error.

The acceptance client prints `ACCEPTANCE_PASS` after its automated prompt and
media checks succeed. Server-log and Windows system-event checks are performed
separately for the same test window.
