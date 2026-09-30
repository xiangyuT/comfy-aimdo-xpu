"""Real Torch XPU queue identity, owned-handle and concurrent snapshot checks."""

import argparse
import ctypes
import gc
import hashlib
import importlib.metadata as metadata
import importlib.util
import json
from pathlib import Path
import sys
import threading
from types import SimpleNamespace


def run(args):
    sys.argv = ['queue_binding_probe', '--enable-dynamic-vram']
    import comfy.cli_args
    from comfy_aimdo import control
    entry = Path('/llm/ComfyUI/custom_nodes/ComfyUI-OmniXPU/runtime_bootstrap.py')
    spec = importlib.util.spec_from_file_location('queue_binding_bootstrap', entry)
    runtime = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = runtime
    spec.loader.exec_module(runtime)
    dist = metadata.distribution('comfy-aimdo-xpu-runtime')
    manifest = json.loads(Path(dist.locate_file('comfy_aimdo_xpu_runtime/provider.json')).read_text())
    provider = runtime._validate_manifest(SimpleNamespace(name='comfy_aimdo.xpu', dist=dist), manifest)
    runtime.bootstrap(providers_override={provider.provider_id: provider}, dynamic_vram_override=True)
    import torch
    from comfy_aimdo import control
    assert torch.__version__ == '2.14.0+xpu' and control.init_devices([0])
    props = torch.xpu.get_device_properties(0)
    assert torch.xpu.device_count() == 1 and props.device_id == 0xE223
    assert str(props.uuid) == '868023e2-0000-0000-1800-000000000000'
    helper = ctypes.CDLL(str(args.helper))
    helper.queue_test_init.argtypes = [ctypes.c_char_p]
    helper.queue_test_init.restype = ctypes.c_bool
    for name in ('queue_test_clone', 'queue_test_foreign_context'):
        getattr(helper, name).argtypes = [ctypes.c_void_p]
        getattr(helper, name).restype = ctypes.c_void_p
    helper.queue_test_same_context.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    helper.queue_test_same_context.restype = ctypes.c_bool
    helper.queue_test_destroy.argtypes = [ctypes.c_void_p]
    helper.queue_test_destroy.restype = None
    helper.queue_test_event_bind.argtypes = [ctypes.c_void_p]
    helper.queue_test_event_bind.restype = ctypes.c_int
    helper.queue_test_sync.argtypes = []
    helper.queue_test_sync.restype = ctypes.c_int
    helper.queue_test_copy.argtypes = [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_size_t]
    helper.queue_test_copy.restype = ctypes.c_int
    assert helper.queue_test_init(str(control.lib._name).encode())
    primary = torch.xpu.current_stream(0)
    pointer = int(primary.sycl_queue)
    assert helper.queue_test_event_bind(pointer) == 0
    before = control.get_xpu_vmm_stats()
    foreign = helper.queue_test_foreign_context(pointer)
    assert foreign and not helper.queue_test_same_context(pointer, foreign)
    rejected = helper.queue_test_event_bind(foreign)
    after = control.get_xpu_vmm_stats()
    expected = 0 if args.mode == 'before' else 999
    assert rejected == expected
    rebind_delta = after['queue_rebind_calls'] - before['queue_rebind_calls']
    assert rebind_delta == (1 if args.mode == 'before' else 0)
    assert after['retire_queue_identity_mismatches'] - before['retire_queue_identity_mismatches'] == 1
    assert helper.queue_test_event_bind(pointer) == 0
    helper.queue_test_destroy(foreign)
    assert helper.queue_test_sync() == 0
    checks = []
    if args.mode == 'candidate':
        clone = helper.queue_test_clone(pointer)
        assert clone and helper.queue_test_event_bind(clone) == 0
        helper.queue_test_destroy(clone)
        # The device binding must retain its queue after the caller wrapper dies.
        assert helper.queue_test_sync() == 0
        secondary = torch.xpu.Stream(device=0)
        targets = [torch.empty(4097, dtype=torch.uint8, device='xpu:0') for _ in range(2)]
        failures = []
        barrier = threading.Barrier(3)
        def copies(index, queue):
            try:
                barrier.wait()
                for iteration in range(32):
                    expected = bytes((j * 29 + iteration * 13 + index * 17) % 251 for j in range(4097))
                    source = ctypes.create_string_buffer(expected)
                    assert helper.queue_test_copy(queue, targets[index].data_ptr(), source, len(expected)) == 0
                    actual = bytes(targets[index].cpu().tolist())
                    assert actual == expected
                    (args.output / f'thread-{index}-{iteration}-actual.bin').write_bytes(actual)
                    checks.append({'thread': index, 'iteration': iteration, 'bytes': 4097,
                                   'sha256': hashlib.sha256(actual).hexdigest()})
            except BaseException as error:
                failures.append(repr(error))
        def snapshots():
            try:
                barrier.wait()
                for _ in range(64):
                    assert helper.queue_test_sync() == 0
            except BaseException as error:
                failures.append(repr(error))
        workers = [threading.Thread(target=copies, args=(0, pointer)),
                   threading.Thread(target=copies, args=(1, int(secondary.sycl_queue))),
                   threading.Thread(target=snapshots)]
        for worker in workers: worker.start()
        for worker in workers: worker.join(timeout=60)
        assert not any(worker.is_alive() for worker in workers) and not failures, failures
        del targets
        gc.collect()
        torch.xpu.synchronize()
        torch.xpu.empty_cache()
        assert len(checks) == 64
    final = control.get_xpu_vmm_stats()
    native_sha256 = hashlib.sha256(Path(control.lib._name).read_bytes()).hexdigest()
    control.deinit()
    return {'status': 'passed', 'mode': args.mode, 'source_revision': manifest['source']['revision'],
            'native_sha256': native_sha256,
            'wrong_context_result': rejected, 'wrong_context_rebind_delta': rebind_delta,
            'wrong_context_gpu_memory_touched': False, 'checks': sorted(checks, key=lambda r:(r['thread'],r['iteration'])),
            'final_stats': final, 'torch': torch.__version__, 'device_id': '0xE223',
            'device_uuid': str(props.uuid), 'public_compiler_available': False}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['before', 'candidate'], required=True)
    parser.add_argument('--helper', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = run(args)
    (args.output / 'probe.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)
