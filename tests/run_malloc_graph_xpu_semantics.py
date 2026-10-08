"""Linux native-owner semantic ports; run each case in a fresh XPU process.

This uses the private diagnostic route, not public record(). CUDA tests remain
unchanged. Every observed tensor is compared completely against a nonuniform
reference and retained as a compressed complete capture.
"""
import argparse
import contextlib
import ctypes
import gc
import hashlib
import importlib.metadata as metadata
import importlib.util
import json
import os
from pathlib import Path
import sys
import threading
import traceback
from types import SimpleNamespace
import zlib

M = 1024 * 1024
CASES = ('empty', 'empty_subgraph', 'different_subgraph', 'optional_subgraph',
         'subgraph_order', 'variable_subgraph_replay', 'deep_nesting',
         'subgraph_phases', 'reordered_free', 'nested_stats', 'long_run',
         'multiple_graphs', 'small', 'odd_sizes', 'fragmentation', 'off_stream',
         'different_stream', 'free_external', 'free_external_subgraph',
         'destructor', 'leak', 'leak_subgraph', 'rogue_event_splice',
         'rogue_reuse', 'small_rogue', 'rogue_alias', 'abort', 'pause',
         'basic', 'nested', 'iterate', 'iterate_break', 'nested_iterate',
         'extra_subgraph', 'small_branch', 'aliasing', 'branch_reuse',
         'changed_size', 'extra_allocation', 'missing_allocation', 'branch_join',
         'assert_break', 'free_outer_subgraph', 'free_rogue_outer')


def digest(data):
    return hashlib.sha256(data).hexdigest()


def expected_bytes(size, seed):
    # Period251 preserves the per-byte formula while avoiding huge Python lists.
    period = bytes((position * (17 + seed % 11) + seed * 13) % 251
                   for position in range(251))
    return (period * ((size + 250) // 251))[:size]


class Allocation:
    def __init__(self, suite, size, compiler=True, *, scoped=True):
        self.suite = suite
        self.seed = suite.next_seed
        suite.next_seed += 1
        self.expected = expected_bytes(size, self.seed)
        self.overrides = []
        self.stream = suite.torch.xpu.current_stream(0)
        active = next((g for g in reversed(suite.graphs) if g.depth), None)
        target = active.graph._stream if active is not None else suite.stream
        scope = (suite.owner.compiler_scope(target)
                 if scoped and (active is not None or compiler)
                 else contextlib.nullcontext())
        with scope:
            self.tensor = suite.torch.empty(size, dtype=suite.torch.uint8, device='xpu:0')
        self.pointer = self.tensor.data_ptr()
        self.compiler = bool(suite.owner.is_compiler_owner(self.pointer)) if size else False
        if size:
            assert suite.owner.is_compiler_owner(self.pointer) == compiler
            self.tensor.copy_(suite.torch.frombuffer(bytearray(self.expected), dtype=suite.torch.uint8))
        else:
            assert self.pointer == 0
        self.check()

    def fill_region(self, start, end, value):
        assert 0 <= start <= end <= len(self.expected) and 0 <= value <= 255
        self.tensor[start:end].fill_(value)
        self.expected = self.expected[:start] + bytes([value]) * (end - start) + self.expected[end:]
        self.overrides.append({'start': start, 'end': end, 'value': value})

    def check(self):
        self.stream.synchronize()
        actual = self.tensor.cpu().numpy().tobytes()
        assert actual == self.expected, 'complete tensor contents changed'
        name = f'payload-{len(self.suite.checks):04d}.zlib'
        (self.suite.output / name).write_bytes(zlib.compress(actual, level=1))
        self.suite.checks.append({'file': name, 'bytes': len(actual), 'seed': self.seed,
                                  'sha256': digest(actual), 'compiler_owner': self.compiler,
                                  'pointer': self.pointer, 'overrides': list(self.overrides)})


VMM_CHANGE_COUNTERS = ('virtual_reserve_calls', 'virtual_reserve_bytes',
                       'physical_create_calls', 'physical_create_bytes',
                       'map_calls', 'map_bytes', 'unmap_calls', 'unmap_bytes',
                       'physical_release_calls')


class ObservedGraph:
    """Observe completed root replay without changing graph operations."""
    def __init__(self, suite, graph):
        self.suite, self.graph = suite, graph
        self.depth = 1  # record_diagnostic() starts the first root.
        self.before = None
        self.before_skipped = 0
        self.nested_break = False

    def __getattr__(self, name):
        return getattr(self.graph, name)

    def push(self, name=None):
        self.graph.push(name)
        if name is None:
            assert self.depth == 0
            self.depth = 1
            self.before = self.suite.control.get_xpu_vmm_stats()
            self.before_skipped = self.graph.skipped_replays
            self.nested_break = False
        else:
            self.depth += 1

    def pop(self):
        broken = self.graph.pop()
        self.depth -= 1
        if self.depth and broken:
            self.nested_break = True
        if self.depth == 0 and self.before is not None:
            after = self.suite.control.get_xpu_vmm_stats()
            delta = {key: after[key] - self.before[key] for key in VMM_CHANGE_COUNTERS}
            skipped = self.graph.skipped_replays > self.before_skipped
            stable = (not broken and not self.nested_break and
                      self.graph.rogue_count == 0 and not skipped)
            self.suite.replays.append({'stable': stable, 'broken': broken,
                                       'nested_break': self.nested_break,
                                       'skipped': skipped, 'vmm_delta': delta})
            self.before = None
            if stable:
                assert all(value == 0 for value in delta.values()), delta
        return broken

    def abort(self):
        self.graph.abort()
        self.depth = 0
        self.before = None

    def iterate(self, name=None):
        before = len(self.graph._scopes)
        broken = self.graph.iterate(name)
        self.depth += len(self.graph._scopes) - before
        if broken:
            self.nested_break = True
        return broken


class Suite:
    def __init__(self, torch, owner, control, output):
        self.torch, self.owner, self.control, self.output = torch, owner, control, output
        self.stream = torch.xpu.current_stream(0)
        self.next_seed = 1
        self.checks = []
        self.graphs = []
        self.stats = []
        self.replays = []
        self.lifetime = []
        self.guards = []

    def graph(self, *, assert_graph_breaks=False):
        graph = ObservedGraph(self, self.owner.record_diagnostic(
            self.stream, assert_graph_breaks=assert_graph_breaks))
        self.graphs.append(graph)
        return graph

    def alloc(self, size, compiler=True):
        return Allocation(self, size, compiler)

    def stat(self, graph, expected):
        observed = (graph.peak_used, graph.virtual_bytes, graph.physical_bytes)
        self.stats.append({'observed': observed, 'upstream_expected': expected})
        assert observed == expected, (observed, expected)

    def observe(self, label):
        row = {'label': label, 'proxy': self.owner.snapshot(),
               'graphs': self.owner.graph_ownership_snapshot(),
               'vmm': vmm(self.control),
               'total_vram': self.control.get_total_vram_usage()}
        self.lifetime.append(row)
        return row

    def destroy_graph(self, graph):
        # Drop the final facade reference, exercising MallocGraph.__del__.
        # The suite must not keep a strong reference that hides destruction.
        self.graphs.remove(graph)
        graph.graph = None
        graph.depth = 0
        gc.collect()
        self.observe('graph_destructor')

    def reject(self, label, operation, message=None):
        try:
            operation()
        except RuntimeError as error:
            if message is not None:
                assert message in str(error), str(error)
            self.guards.append({'label': label, 'class': type(error).__name__, 'message': str(error)})
        else:
            raise AssertionError(label + ': expected RuntimeError was not raised')


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


def off_stream(s):
    g = s.graph()
    other = s.torch.xpu.Stream(device=0)
    with s.torch.xpu.stream(other):
        ordinary = s.alloc(8 * M, compiler=False)
        del ordinary
        with g.use_stream(other):
            compiled = s.alloc(8 * M)
            pointer = compiled.pointer
            del compiled
    assert not g.pop()
    other.synchronize()
    s.stat(g, (8 * M, 8 * M, 8 * M))
    g.push()
    with s.torch.xpu.stream(other), g.use_stream(other):
        compiled = s.alloc(8 * M)
        assert compiled.pointer == pointer
        del compiled
    assert not g.pop()


def different_stream(s):
    g = s.graph()
    value = s.alloc(8 * M)
    pointer = value.pointer
    del value
    assert not g.pop()
    other = s.torch.xpu.Stream(device=0)
    with s.torch.xpu.stream(other):
        g.push()
        value = s.alloc(8 * M, compiler=False)
        del value
        assert not g.pop()
    other.synchronize()
    assert g.skipped_replays == 1
    g.push()
    value = s.alloc(8 * M)
    assert value.pointer == pointer
    del value
    assert not g.pop() and g.skipped_replays == 1


def free_external(s):
    g = s.graph()
    with s.owner.compiler_scope(s.stream), s.owner.paused_graph_scope(g.graph):
        value = Allocation(s, 8 * M, compiler=False, scoped=False)
    del value
    assert not g.pop()
    g.push()
    assert not g.pop()


def free_external_subgraph(s):
    value = s.alloc(8 * M, compiler=False)
    g = s.graph()
    g.push('inner')
    del value
    assert not g.pop() and not g.pop()
    g.push()
    g.push('inner')
    assert not g.pop() and not g.pop()


def destructor(s):
    warmup = s.alloc(1, compiler=False)
    del warmup
    s.torch.xpu.synchronize()
    baseline = s.control.get_total_vram_usage()
    g = s.graph()
    value = s.alloc(8 * M)
    del value
    assert not g.pop()
    assert s.observe('completed_graph')['total_vram'] == baseline + 8 * M
    s.destroy_graph(g)
    assert s.control.get_total_vram_usage() == baseline
    assert s.owner.graph_ownership_snapshot() == {'live': 0, 'deferred': 0}


def leak(s):
    g = s.graph()
    value = s.alloc(8 * M)
    pointer = value.pointer
    g.pop()
    assert g.rogue_count == 1
    g.push()
    replacement = s.alloc(8 * M)
    assert replacement.pointer != pointer
    value.check()
    del replacement
    g.pop()
    assert g.rogue_count == 1
    s.destroy_graph(g)
    value.check()
    holder = [value]
    del value
    thread = threading.Thread(target=lambda: holder.pop())
    thread.start()
    thread.join(timeout=30)
    assert not thread.is_alive()
    gc.collect()
    s.observe('foreign_escaped_free')


def leak_subgraph(s):
    g = s.graph()
    g.push('inner')
    value = s.alloc(8 * M)
    pointer = value.pointer
    g.pop()
    replacement = s.alloc(8 * M)
    assert replacement.pointer != pointer
    value.check()
    del replacement
    g.pop()
    g.push()
    g.push('inner')
    g.pop()  # The missing inner allocation follows the original break path.
    replacement = s.alloc(8 * M)
    assert replacement.pointer != pointer
    value.check()
    del replacement
    g.pop()
    s.destroy_graph(g)
    value.check()
    del value
    gc.collect()


def rogue_event_splice(s):
    g = s.graph()
    rogue = s.alloc(8 * M)
    tail = s.alloc(16 * M)
    tail_pointer = tail.pointer
    rogue.check()
    del tail
    assert not g.pop()
    g.push()
    tail = s.alloc(16 * M)
    assert tail.pointer == tail_pointer and tail.pointer != rogue.pointer
    rogue.check()
    del tail
    assert not g.pop()
    s.destroy_graph(g)
    rogue.check()
    del rogue
    gc.collect()


def rogue_reuse(s):
    g = s.graph()
    value = s.alloc(8 * M)
    pointer = value.pointer
    g.pop()
    value.check()
    holder = [value]
    del value
    thread = threading.Thread(target=lambda: holder.pop())
    thread.start()
    thread.join(timeout=30)
    assert not thread.is_alive()
    s.observe('foreign_rogue_free')
    g.push()
    value = s.alloc(8 * M)
    assert value.pointer == pointer
    del value
    g.pop()
    s.destroy_graph(g)


def small_rogue(s):
    g = s.graph()
    first, second = s.alloc(M), s.alloc(2 * M)
    pointers = (first.pointer, second.pointer)
    first.check()
    g.pop()
    assert g.rogue_count == 2
    g.push()
    replacement_first, replacement_second = s.alloc(M), s.alloc(2 * M)
    assert replacement_first.pointer not in pointers
    assert replacement_second.pointer not in pointers
    first.check()
    second.check()
    del replacement_first, replacement_second
    g.pop()
    assert g.rogue_count == 2
    s.destroy_graph(g)
    first.check()
    second.check()
    del first
    gc.collect()
    second.check()
    del second
    gc.collect()


def rogue_alias(s):
    g = s.graph()
    for size in (8 * M, 16 * M):
        g.push('large')
        value = s.alloc(size)
        del value
        g.pop()
    g.push('large')
    large = s.alloc(8 * M)
    g.pop()
    g.push('large')
    value = s.alloc(16 * M)
    large.check()
    del value
    g.pop()
    for size in (M, 2 * M):
        g.push('small')
        value = s.alloc(size)
        del value
        g.pop()
    g.push('small')
    small_value = s.alloc(M)
    g.pop()
    g.push('small')
    value = s.alloc(2 * M)
    assert value.pointer // (8 * M) != small_value.pointer // (8 * M)
    large.check()
    small_value.check()
    del value
    g.pop()
    g.pop()
    s.destroy_graph(g)
    large.check()
    small_value.check()
    del large, small_value
    gc.collect()


def abort(s):
    warmup = s.alloc(1, compiler=False)
    del warmup
    s.torch.xpu.synchronize()
    s.torch.xpu.empty_cache()
    baseline = s.control.get_total_vram_usage()
    g = s.graph(assert_graph_breaks=True)
    root = s.alloc(M)
    g.push('block')
    nested = s.alloc(8 * M)
    root.check()
    g.abort()
    g.abort()
    assert g.rogue_count == 2
    s.destroy_graph(g)
    replacement = s.alloc(8 * M, compiler=False)
    root.check()
    nested.check()
    del replacement
    holder = [nested]
    del nested
    thread = threading.Thread(target=lambda: holder.pop())
    thread.start()
    thread.join(timeout=30)
    assert not thread.is_alive()
    del root
    gc.collect()
    g = s.graph()
    g.push('block')
    value = s.alloc(8 * M)
    pointer = value.pointer
    del value
    assert not g.pop() and not g.pop()
    g.push()
    g.push('block')
    value = s.alloc(8 * M)
    assert value.pointer == pointer
    s.destroy_graph(g)
    replacement = s.alloc(8 * M, compiler=False)
    value.check()
    del replacement, value
    gc.collect()
    s.torch.xpu.synchronize()
    s.torch.xpu.empty_cache()
    # Drain ordinary native cache on both sides of the compiler lifetime check.
    assert s.control.get_total_vram_usage() == baseline
    s.observe('abort_reentry_cleanup')


def pause(s):
    g = s.graph()
    pointers = None
    for iteration in range(2):
        if iteration:
            g.push()
        first = s.alloc(8 * M)
        if not iteration:
            with s.owner.compiler_scope(s.stream), s.owner.paused_graph_scope(g.graph):
                temporary = Allocation(s, 8 * M, compiler=False, scoped=False)
                first.check()
                del temporary
        second = s.alloc(8 * M)
        observed = first.pointer, second.pointer
        pointers = pointers or observed
        assert observed == pointers
        first.check()
        del second, first
        assert not g.pop()
        assert g.physical_bytes == 16 * M


def basic(s):
    g = s.graph()
    pointers = None
    for iteration in range(2):
        if iteration:
            outside = s.alloc(8 * M, compiler=False)
            del outside
            g.push()
        first, second = s.alloc(8 * M), s.alloc(16 * M)
        observed = first.pointer, second.pointer
        pointers = pointers or observed
        assert observed == pointers
        first.check()
        del first, second
        spanning = s.alloc(24 * M)
        assert spanning.pointer == pointers[0]
        del spanning
        s.stat(g, (24 * M, 24 * M, 24 * M))
        g.pop()


def nested(s):
    g = s.graph()

    def inner():
        g.push('inner')
        first, second = s.alloc(8 * M), s.alloc(8 * M)
        first.check()
        pointers = first.pointer, second.pointer
        del first
        third = s.alloc(16 * M)
        pointers += (third.pointer,)
        second.check()
        del second, third
        g.pop()
        return pointers

    def outer():
        first, second = s.alloc(8 * M), s.alloc(16 * M)
        pointers = first.pointer, second.pointer
        first.check()
        del second
        inner_pointers = inner()
        assert inner() == inner_pointers
        assert inner() == inner_pointers
        first.check()
        last = s.alloc(8 * M)
        first.check()
        pointer = last.pointer
        del first, last
        return pointers, inner_pointers, pointer

    pointers = outer()
    g.pop()
    g.push()
    assert outer() == pointers
    g.pop()


def iterate(s):
    g = s.graph()
    pointer = None
    for iteration, repeats in enumerate((3, 2)):
        if iteration:
            g.push()
        for _ in range(repeats):
            assert not g.iterate('block')
            value = s.alloc(M)
            pointer = pointer or value.pointer
            assert value.pointer == pointer
            del value
        assert not g.iterate() and not g.pop()


def iterate_break(s):
    g = s.graph()
    for iteration, size in enumerate((M, 2 * M)):
        if iteration:
            g.push()
        assert not g.iterate('block')
        value = s.alloc(size)
        del value
        assert g.iterate() == bool(iteration)
        assert not g.pop()


def nested_iterate(s):
    g = s.graph()
    outer_pointer = inner_pointer = None
    for iteration, inner_repeats in enumerate((2, 3)):
        if iteration:
            g.push()
        g.push('wrapper')
        for _ in range(2):
            assert not g.iterate('outer')
            outer = s.alloc(8 * M)
            outer_pointer = outer_pointer or outer.pointer
            assert outer.pointer == outer_pointer
            for _ in range(inner_repeats):
                assert not g.iterate('inner')
                inner = s.alloc(8 * M)
                inner_pointer = inner_pointer or inner.pointer
                assert inner.pointer == inner_pointer
                outer.check()
                del inner
            assert not g.iterate()
            del outer
        assert not g.iterate() and not g.pop() and not g.pop()


def extra_subgraph(s):
    g = s.graph()
    for iteration, names in enumerate((('first',), ('first', 'second'), ('first',))):
        if iteration:
            g.push()
        for name in names:
            g.push(name)
            assert not g.pop()
        assert g.pop() == (iteration == 1)


def small_branch(s):
    g = s.graph()
    pointer = None
    for iteration in range(3):
        if iteration:
            g.push()
        first = s.alloc(M)
        pointer = pointer or first.pointer
        assert first.pointer == pointer
        if iteration == 1:
            second = s.alloc(2 * M)
            assert second.pointer != pointer
            first.check()
            del second
        del first
        g.pop()


def aliasing(s):
    g = s.graph()
    pointers = None
    spanning_pointer = None
    for iteration in range(2):
        if iteration:
            g.push()
        live, freed = s.alloc(8 * M), s.alloc(16 * M)
        freed_pointer = freed.pointer
        live.check()
        del freed
        alias = s.alloc(24 * M)
        assert alias.pointer != freed_pointer
        observed = live.pointer, alias.pointer
        pointers = pointers or observed
        assert observed == pointers
        live.check()
        del alias, live
        spanning = s.alloc(48 * M)
        spanning_pointer = spanning_pointer or spanning.pointer
        assert spanning.pointer == spanning_pointer
        if not iteration:
            spanning.fill_region(8 * M, 16 * M, 1)
            spanning.fill_region(24 * M, 32 * M, 2)
            s.observe('before_alias_reductions')
            # Upstream records allocations made by these operators as well.
            # Scope their real Torch outputs; backend raw workspaces retain
            # the diagnostic proxy's separately declared native ownership.
            with s.owner.compiler_scope(s.stream):
                assert spanning.tensor[8 * M:16 * M].min().item() == 1
                assert spanning.tensor[24 * M:32 * M].min().item() == 2
            s.observe('after_alias_reductions')
            spanning.check()
        del spanning
        g.pop()
        if not iteration:
            s.stat(g, (56 * M, 104 * M, 56 * M))


def branch_reuse(s):
    g = s.graph()
    value = s.alloc(8 * M)
    pointer = value.pointer
    del value
    assert not g.pop()
    g.push()
    assert g.pop()
    g.push()
    value = s.alloc(8 * M)
    assert value.pointer == pointer
    del value
    assert not g.pop()


def changed_size(s):
    g = s.graph()
    original = None
    for iteration, size in enumerate((8 * M, 16 * M, 8 * M)):
        if iteration:
            g.push()
        value = s.alloc(size)
        if original is None:
            original = value.pointer
        assert (value.pointer == original) == (iteration != 1)
        del value
        assert g.pop() == (iteration == 1)


def extra_allocation(s):
    g = s.graph()
    pointer = None
    for iteration in range(3):
        if iteration:
            g.push()
        first = s.alloc(8 * M)
        pointer = pointer or first.pointer
        assert first.pointer == pointer
        if iteration == 1:
            second = s.alloc(8 * M)
            assert second.pointer != pointer
            first.check()
            del first, second
        else:
            del first
        g.pop()


def missing_allocation(s):
    g = s.graph()
    for iteration, repeats in enumerate((2, 1, 2)):
        if iteration:
            g.push()
        for _ in range(repeats):
            value = s.alloc(8 * M)
            del value
        assert g.pop() == (iteration == 1)


def branch_join(s):
    g = s.graph()
    pointers = None
    for iteration, size in enumerate((8 * M, 16 * M, 8 * M)):
        if iteration:
            g.push()
        outer = s.alloc(8 * M)
        g.push('inner')
        inner = s.alloc(size)
        if pointers is None:
            outer_pointer, inner_pointer = outer.pointer, inner.pointer
        assert outer.pointer == outer_pointer
        assert inner.pointer != outer_pointer
        if iteration == 2:
            assert inner.pointer == inner_pointer
        outer.check()
        del inner
        assert g.pop() == (iteration == 1)
        tail = s.alloc(8 * M)
        outer.check()
        pointers = pointers or (outer_pointer, inner_pointer, tail.pointer)
        assert tail.pointer == pointers[2]
        del tail, outer
        assert not g.pop()


def assert_break(s):
    g = s.graph(assert_graph_breaks=True)
    value = s.alloc(8 * M)
    del value
    g.pop()
    g.push()
    s.reject('assert_graph_breaks', lambda: s.alloc(16 * M))
    s.destroy_graph(g)


def free_outer_subgraph(s):
    g = s.graph()
    value = s.alloc(8 * M)
    g.push('inner')
    del value
    s.reject('outer_allocation_free_in_subgraph', g.pop, 'aimdo memory compile error')
    s.destroy_graph(g)


def free_rogue_outer(s):
    g = s.graph()
    g.push('inner')
    freed, survivor = s.alloc(8 * M), s.alloc(8 * M)
    survivor.check()
    g.pop()
    holder = [freed]
    del freed
    thread = threading.Thread(target=lambda: holder.pop())
    thread.start()
    thread.join(timeout=30)
    assert not thread.is_alive()
    s.reject('foreign_rogue_free_in_outer_scope', g.pop, 'aimdo memory compile error')
    s.destroy_graph(g)
    survivor.check()
    del survivor
    gc.collect()


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
    driver = Path(os.environ['ZE_ENABLE_ALT_DRIVERS']).resolve()
    driver_sha = digest(driver.read_bytes())
    assert driver_sha == plan['driver_sha256']
    driver_mappings = [line for line in Path('/proc/self/maps').read_text().splitlines()
                       if line.rstrip().endswith(str(driver))]
    assert driver_mappings, 'selected driver is not loaded in the device process'
    avoid_alias = os.environ.get('AIMDO_XPU_GRAPH_AVOID_ALIAS', '0') == '1'
    assert avoid_alias == plan['avoid_physical_alias']
    native_hash = digest(Path(control.lib._name).read_bytes())
    assert native_hash == plan['native_sha256']
    assert not control.get_memory_compiler_capability()['available']
    assert native_owner.snapshot() == [0] * 17 and vmm(control) == [0] * 7
    suite = Suite(torch, native_owner, control, output)
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
            'driver_sha256': driver_sha, 'driver_path': str(driver), 'driver_mappings': driver_mappings,
            'avoid_physical_alias': avoid_alias,
            'checks': suite.checks, 'stats': suite.stats, 'replays': suite.replays,
            'lifetime': suite.lifetime, 'guards': suite.guards, 'error': error,
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
