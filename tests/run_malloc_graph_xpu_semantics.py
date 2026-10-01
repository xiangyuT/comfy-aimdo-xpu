"""Linux native-owner semantic ports; run each case in a fresh XPU process.

This uses the private diagnostic route, not public record(). CUDA tests remain
unchanged. Every observed tensor is compared completely against a nonuniform
reference and retained as a compressed complete capture.
"""
import argparse
import ctypes
import gc
import hashlib
import importlib.metadata as metadata
import importlib.util
import json
from pathlib import Path
import sys
import traceback
from types import SimpleNamespace
import zlib

M = 1024 * 1024
CASES = ('empty', 'empty_subgraph', 'different_subgraph', 'optional_subgraph',
         'subgraph_order', 'variable_subgraph_replay', 'deep_nesting',
         'subgraph_phases', 'reordered_free', 'nested_stats', 'long_run',
         'multiple_graphs', 'small', 'odd_sizes', 'fragmentation')


def digest(data):
    return hashlib.sha256(data).hexdigest()


def expected_bytes(size, seed):
    # Period251 preserves the per-byte formula while avoiding huge Python lists.
    period = bytes((position * (17 + seed % 11) + seed * 13) % 251
                   for position in range(251))
    return (period * ((size + 250) // 251))[:size]


class Allocation:
    def __init__(self, suite, size):
        self.suite = suite
        self.seed = suite.next_seed
        suite.next_seed += 1
        self.expected = expected_bytes(size, self.seed)
        with suite.owner.compiler_scope(suite.stream):
            self.tensor = suite.torch.empty(size, dtype=suite.torch.uint8, device='xpu:0')
        self.pointer = self.tensor.data_ptr()
        if size:
            assert suite.owner.is_compiler_owner(self.pointer)
            self.tensor.copy_(suite.torch.frombuffer(bytearray(self.expected), dtype=suite.torch.uint8))
        else:
            assert self.pointer == 0
        self.check()

    def check(self):
        self.suite.stream.synchronize()
        actual = self.tensor.cpu().numpy().tobytes()
        assert actual == self.expected, 'complete tensor contents changed'
        name = f'payload-{len(self.suite.checks):04d}.zlib'
        (self.suite.output / name).write_bytes(zlib.compress(actual, level=1))
        self.suite.checks.append({'file': name, 'bytes': len(actual), 'seed': self.seed,
                                  'sha256': digest(actual)})


class Suite:
    def __init__(self, torch, owner, output):
        self.torch, self.owner, self.output = torch, owner, output
        self.stream = torch.xpu.current_stream(0)
        self.next_seed = 1
        self.checks = []
        self.graphs = []
        self.stats = []

    def graph(self):
        graph = self.owner.record_diagnostic(self.stream)
        self.graphs.append(graph)
        return graph

    def alloc(self, size):
        return Allocation(self, size)

    def stat(self, graph, expected):
        observed = (graph.peak_used, graph.virtual_bytes, graph.physical_bytes)
        self.stats.append({'observed': observed, 'upstream_expected': expected})
        assert observed == expected, (observed, expected)


def empty(s):
    g = s.graph()
    for iteration in range(2):
        if iteration:
            g.push()
        value = s.alloc(0)
        del value
        assert not g.pop()
        s.stat(g, (0, 0, 0))


def empty_subgraph(s):
    g = s.graph()
    for iteration in range(2):
        if iteration:
            g.push()
        for _ in range(2):
            g.push('empty')
            assert not g.pop()
        assert not g.pop()
        s.stat(g, (0, 0, 0))


def different_subgraph(s):
    g = s.graph()
    g.push('first')
    assert not g.pop() and not g.pop()
    for name, broken in (('second', True), ('first', False), ('second', False)):
        g.push()
        g.push(name)
        assert not g.pop()
        assert g.pop() == broken


def named_sequence(s, sequences):
    g = s.graph()
    for iteration, names in enumerate(sequences):
        if iteration:
            g.push()
        for name in names:
            g.push(name)
            assert not g.pop()
        assert not g.pop()
    s.stat(g, (0, 0, 0))


def optional_subgraph(s):
    named_sequence(s, (('first', 'second'), ('first',)))


def subgraph_order(s):
    named_sequence(s, (('first', 'second'), ('second', 'first')))


def variable_subgraph_replay(s):
    g = s.graph()
    for iteration, count in enumerate((2, 0, 1, 3)):
        if iteration:
            g.push()
        for _ in range(count):
            g.push('inner')
            value = s.alloc(8 * M)
            del value
            assert not g.pop()
        assert not g.pop()


def deep_nesting(s):
    g = s.graph()
    pointer = None
    for iteration in range(2):
        if iteration:
            g.push()
        g.push('outer')
        g.push('inner')
        value = s.alloc(8 * M)
        if pointer is None:
            pointer = value.pointer
        assert value.pointer == pointer
        del value
        assert not g.pop() and not g.pop() and not g.pop()


def subgraph_phases(s):
    g = s.graph()
    for iteration, phases in enumerate((('inner', None, 'inner'), (None, 'inner', 'inner'))):
        if iteration:
            g.push()
        for name in phases:
            if name is not None:
                g.push(name)
            value = s.alloc(8 * M)
            del value
            if name is not None:
                assert not g.pop()
        assert not g.pop()


def reordered_free(s):
    g = s.graph()
    for iteration in range(2):
        if iteration:
            g.push()
        first = s.alloc(8 * M)
        second = s.alloc(16 * M)
        first.check()  # Check adjacent live storage after another tensor was written.
        if iteration:
            del second, first
        else:
            del first, second
        assert g.pop() == bool(iteration)


def nested_stats(s):
    g = s.graph()
    for iteration in range(2):
        if iteration:
            g.push()
        outer = s.alloc(8 * M)
        for _ in range(2):
            g.push('inner')
            first = s.alloc(8 * M)
            second = s.alloc(8 * M)
            outer.check()
            first.check()
            del first, second
            assert not g.pop()
        del outer
        assert not g.pop()
        s.stat(g, (24 * M, 24 * M, 24 * M))


def long_run(s):
    g = s.graph()
    pointer = None
    for iteration in range(101):  # One record plus100 upstream replay iterations.
        if iteration:
            g.push()
        value = s.alloc(8 * M)
        pointer = pointer or value.pointer
        assert value.pointer == pointer
        del value
        assert not g.pop()
        s.stat(g, (8 * M, 8 * M, 8 * M))


def multiple_graphs(s):
    first_graph = s.graph()
    value = s.alloc(8 * M)
    first_pointer = value.pointer
    del value
    assert not first_graph.pop()
    second_graph = s.graph()
    value = s.alloc(16 * M)
    second_pointer = value.pointer
    del value
    assert not second_graph.pop()
    for graph, size, pointer in ((first_graph, 8 * M, first_pointer),
                                  (second_graph, 16 * M, second_pointer)):
        graph.push()
        value = s.alloc(size)
        assert value.pointer == pointer
        del value
        assert not graph.pop()


def small(s):
    g = s.graph()
    pointers = None
    replacement_pointer = None
    for iteration in range(2):
        if iteration:
            g.push()
        first, second = s.alloc(M), s.alloc(2 * M)
        observed = first.pointer, second.pointer
        pointers = pointers or observed
        assert observed == pointers
        first.check()
        del first
        replacement = s.alloc(M // 2)
        replacement_pointer = replacement_pointer or replacement.pointer
        assert replacement.pointer == replacement_pointer == pointers[0]
        second.check()
        del second, replacement
        assert not g.pop()
        s.stat(g, (3 * M, 8 * M, 8 * M))


def odd_sizes(s):
    g = s.graph()
    pointers = []
    for iteration in range(2):
        if iteration:
            g.push()
        for index, size in enumerate((8 * M + 1, 24 * M + 1)):
            value = s.alloc(size)
            if iteration:
                assert value.pointer == pointers[index]
            else:
                pointers.append(value.pointer)
            del value
        assert not g.pop()
        s.stat(g, (32 * M, 48 * M, 32 * M))


def fragmentation(s):
    g = s.graph()
    pointers = None
    extended_pointer = None
    for iteration in range(2):
        if iteration:
            g.push()
        first, hole, last = s.alloc(8 * M), s.alloc(16 * M), s.alloc(8 * M)
        observed = first.pointer, hole.pointer, last.pointer
        pointers = pointers or observed
        assert observed == pointers
        first.check()
        hole.check()
        del hole
        extended = s.alloc(24 * M)
        extended_pointer = extended_pointer or extended.pointer
        assert extended.pointer == extended_pointer
        first.check()
        last.check()
        del first, last, extended
        assert not g.pop()
        s.stat(g, (40 * M, 56 * M, 40 * M))


def vmm(control):
    values = (ctypes.c_uint64 * 7)()
    control.lib.xpu_get_vmm_ownership.argtypes = [ctypes.POINTER(ctypes.c_uint64), ctypes.c_size_t]
    control.lib.xpu_get_vmm_ownership.restype = ctypes.c_bool
    assert control.lib.xpu_get_vmm_ownership(values, 7)
    return list(values)


def run(case, output, plan):
    assert 'torch' not in sys.modules
    entry = Path('/llm/ComfyUI/custom_nodes/ComfyUI-OmniXPU/runtime_bootstrap.py')
    spec = importlib.util.spec_from_file_location('xpu_semantics_bootstrap', entry)
    runtime = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = runtime
    spec.loader.exec_module(runtime)
    from comfy_aimdo import control  # Official entry must exist before provider activation.
    dist = metadata.distribution('comfy-aimdo-xpu-runtime')
    manifest = json.loads(Path(dist.locate_file('comfy_aimdo_xpu_runtime/provider.json')).read_text())
    assert manifest['source']['revision'] == plan['native_source_revision']
    provider = runtime._validate_manifest(SimpleNamespace(name='comfy_aimdo.xpu', dist=dist), manifest)
    runtime.bootstrap(providers_override={provider.provider_id: provider}, dynamic_vram_override=True)
    from comfy_aimdo import native_owner
    import torch
    assert torch.__version__ == plan['torch'] and control.init_devices([0])
    props = torch.xpu.get_device_properties(0)
    assert torch.xpu.device_count() == 1 and props.device_id == 0xE223
    assert str(props.uuid) == plan['device_uuid']
    native_hash = digest(Path(control.lib._name).read_bytes())
    assert native_hash == plan['native_sha256']
    assert not control.get_memory_compiler_capability()['available']
    assert native_owner.snapshot() == [0] * 17 and vmm(control) == [0] * 7
    suite = Suite(torch, native_owner, output)
    error = None
    try:
        globals()[case](suite)
    except BaseException as caught:
        error = {'class': type(caught).__name__, 'message': str(caught), 'traceback': traceback.format_exc()}
    # Checked close handles an active root and a completed root; abort requires
    # an active graph. Preserve a semantic failure while still checking cleanup.
    for graph in suite.graphs:
        assert graph.close()
    suite.graphs.clear()
    gc.collect()
    torch.xpu.synchronize()
    torch.xpu.empty_cache()
    final = native_owner.snapshot()
    final_vmm = vmm(control)
    assert final[0] == final[16] == 0 and final[1] == final[2] and final[12] == final[13]
    assert final_vmm == [0] * 7
    control.deinit()
    return {'status': 'passed' if error is None else 'failed', 'case': case,
            'native_source_revision': manifest['source']['revision'], 'native_sha256': native_hash,
            'torch': torch.__version__, 'device_uuid': str(props.uuid), 'physical_xpu': 0,
            'checks': suite.checks, 'stats': suite.stats, 'error': error,
            'final': final, 'final_vmm': final_vmm, 'deinit_succeeded': True,
            'public_compiler_available': False}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--case', choices=CASES, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--plan', type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    try:
        result = run(args.case, args.output_dir, json.loads(args.plan.read_text()))
    except BaseException as error:
        result = {'status': 'failed', 'case': args.case, 'harness_or_cleanup_error': repr(error),
                  'traceback': traceback.format_exc()}
    (args.output_dir / 'probe.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({key: value for key, value in result.items() if key != 'checks'}), flush=True)
    raise SystemExit(0 if result['status'] == 'passed' else 2)
