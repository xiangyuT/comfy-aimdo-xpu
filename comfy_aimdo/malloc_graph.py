import contextlib
import ctypes
from pathlib import Path
import threading

from . import control

if Path(control.__file__).resolve().parent != Path(__file__).resolve().parent:
    raise ImportError("AIMDO malloc_graph and control must come from the same provider")


class MallocGraph:
    def __init__(self, handle, stream, *, owner_thread=None, native_lib=None):
        self._handle = handle
        self._stream = stream
        self._scopes = []
        self._owner_thread = owner_thread
        self._native_lib = native_lib
        self._lifecycle_lock = threading.RLock()

    def _check_owner(self):
        if not self._handle:
            raise RuntimeError("AIMDO memory graph is closed")
        if self._owner_thread is not None:
            if threading.current_thread() is not self._owner_thread:
                raise RuntimeError("AIMDO diagnostic graph requires its owner thread")
            from . import native_owner
            native_owner.drain_deferred_graphs()

    def _call(self, function, *args):
        with self._lifecycle_lock:
            self._check_owner()
            result = function(self._handle, *args)
        if not result:
            raise RuntimeError("aimdo memory compile error")
        return result

    def push(self, name=None):
        self._call(control.lib.malloc_graph_push, name.encode() if name is not None else None)
        if name is not None:
            self._scopes.append(None)

    def pop(self):
        broken = self._call(control.lib.malloc_graph_pop) == 2
        if self._scopes:
            self._scopes.pop()
        return broken

    def abort(self):
        self._call(control.lib.malloc_graph_abort)
        self._scopes.clear()

    def pause(self, sync=False):
        self._call(control.lib.malloc_graph_pause, True, sync)

    def resume(self, sync=False):
        self._call(control.lib.malloc_graph_pause, False, sync)

    @contextlib.contextmanager
    def use_stream(self, stream):
        with self._lifecycle_lock:
            previous = self._stream
            pointer = (int(stream.sycl_queue) if control.implementation == "xpu"
                       else int(stream.cuda_stream))
            self._call(control.lib.malloc_graph_set_stream, ctypes.c_void_p(pointer))
            self._stream = stream
            try:
                yield
            finally:
                pointer = (int(previous.sycl_queue) if control.implementation == "xpu"
                           else int(previous.cuda_stream))
                self._call(control.lib.malloc_graph_set_stream, ctypes.c_void_p(pointer))
                self._stream = previous

    def iterate(self, name=None):
        broken = False
        if name is None:
            if self._scopes:
                broken = self.pop()
            return broken
        if self._scopes and self._scopes[-1] == name:
            broken = self.pop()
        self.push(name)
        self._scopes[-1] = name
        return broken

    def _stat(self, which):
        with self._lifecycle_lock:
            self._check_owner()
            return control.lib.malloc_graph_stat(self._handle, which)

    peak_used = property(lambda self: self._stat(0))
    virtual_bytes = property(lambda self: self._stat(1))
    physical_bytes = property(lambda self: self._stat(2))
    rogue_count = property(lambda self: self._stat(3))

    def close(self):
        """Close on the owner thread, or queue a diagnostic close to that thread."""
        with self._lifecycle_lock:
            handle = self._handle
            if not handle:
                return True
            if self._owner_thread is not None:
                from . import native_owner

                if threading.current_thread() is not self._owner_thread:
                    native_owner._defer_graph_destroy(
                        self._owner_thread, self._native_lib, handle
                    )
                    self._handle = None
                    return False
                native_owner.drain_deferred_graphs()
                if native_owner._destroy_graph_terminal(self._native_lib, handle):
                    raise RuntimeError(
                        "terminal AIMDO graph destroy error; process must exit"
                    )
                if not native_owner._destroy_graph_checked(self._native_lib, handle):
                    if native_owner._destroy_graph_terminal(self._native_lib, handle):
                        raise RuntimeError(
                            "terminal AIMDO graph destroy error; process must exit"
                        )
                    raise RuntimeError("AIMDO diagnostic graph destroy failed")
            else:
                if control.lib is None:
                    raise RuntimeError("AIMDO library is no longer initialized")
                control.lib.malloc_graph_destroy(handle)
            self._handle = None
            return True

    def __del__(self):
        handle = getattr(self, "_handle", None)
        if not handle:
            return
        try:
            self.close()
        except Exception:
            owner = getattr(self, "_owner_thread", None)
            library = getattr(self, "_native_lib", None)
            if owner is not None and library is not None:
                try:
                    from . import native_owner
                    native_owner._defer_graph_destroy(owner, library, handle)
                    self._handle = None
                except Exception:
                    pass


def record(stream, assert_graph_breaks=False):
    if (control.implementation == "xpu"
            or getattr(getattr(stream, "device", None), "type", None) == "xpu"):
        raise NotImplementedError("AIMDO memory compiler is not yet supported on XPU")
    handle = control.lib.malloc_graph_create(
        control.get_devctx(stream.device.index), ctypes.c_void_p(stream.cuda_stream),
        assert_graph_breaks,
    )
    if not handle:
        raise RuntimeError("aimdo memory compile error: could not start recording")
    return MallocGraph(handle, stream)
