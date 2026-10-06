"""Device-neutral checks of graph-replay evidence classification."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

spec = importlib.util.spec_from_file_location(
    'xpu_semantic_observation', Path(__file__).with_name('run_malloc_graph_xpu_semantics.py'))
semantics = importlib.util.module_from_spec(spec)
spec.loader.exec_module(semantics)


class Graph:
    rogue_count = 0
    skipped_replays = 0

    def __init__(self, outcomes):
        self.outcomes = iter(outcomes)

    def push(self, name=None):
        pass

    def pop(self):
        return next(self.outcomes)


def observation(outcomes):
    counters = dict.fromkeys(semantics.VMM_CHANGE_COUNTERS, 0)
    suite = SimpleNamespace(control=SimpleNamespace(
        get_xpu_vmm_stats=lambda: dict(counters)), replays=[])
    return semantics.ObservedGraph(suite, Graph(outcomes)), suite, counters


def test_child_break_does_not_become_stable_root_replay():
    graph, suite, counters = observation((False, True, False))
    assert not graph.pop()
    graph.push()
    graph.push('inner')
    assert graph.pop()
    counters['physical_create_calls'] = 2
    counters['map_calls'] = 2
    assert not graph.pop()
    row = suite.replays[0]
    assert row['broken'] is False and row['nested_break'] is True
    assert row['stable'] is False and row['vmm_delta']['map_calls'] == 2


def test_unchanged_root_still_rejects_vmm_mutation():
    graph, suite, counters = observation((False, False))
    graph.pop()
    graph.push()
    counters['map_calls'] = 1
    with pytest.raises(AssertionError):
        graph.pop()


def test_excluded_root_does_not_count_as_compiler_replay():
    graph, suite, counters = observation((False, False))
    graph.pop()
    graph.push()
    graph.graph.skipped_replays = 1
    assert not graph.pop()
    assert suite.replays[0]['skipped'] is True
    assert suite.replays[0]['stable'] is False
