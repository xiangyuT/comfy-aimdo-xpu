from pathlib import Path
import re
import shutil
import subprocess

import pytest


SOURCE = (Path(__file__).parents[1] / "src" / "model-vbar.c").read_text(
    encoding="utf-8"
)
BUILD_SCRIPT = (
    Path(__file__).parents[1] / "scripts" / "build-linux-xpu.sh"
).read_text(encoding="utf-8")


def test_linux_mod1_does_not_read_windows_retirement_fields():
    mod1 = SOURCE.split("static inline bool mod1", 1)[1].split(
        "if (do_free)", 1
    )[0]
    windows_branch, linux_branch = mod1.split("#else", 1)

    assert "rp->evicting" in windows_branch
    assert "rp->retire_unknown" in windows_branch
    assert "rp->evicting" not in linux_branch
    assert "rp->retire_unknown" not in linux_branch
    assert "do_unpin || rp->pin_count == 0" in linux_branch


def test_platform_neutral_entry_points_are_declared_before_first_use():
    range_declaration = SOURCE.index("int aimdo_vbar_describe_range(")
    range_wrapper = SOURCE.index("int aimdo_vbar_describe_address(")
    stream_declaration = SOURCE.index("void vbar_unpin_stream(")
    stream_wrapper = SOURCE.index("void vbar_unpin(")

    assert range_declaration < range_wrapper
    assert stream_declaration < stream_wrapper


def test_linux_xpu_runtime_binds_its_own_native_functions():
    assert "-Wl,-Bsymbolic-functions" in BUILD_SCRIPT


@pytest.mark.parametrize("platform_macro", ("_WIN32", "_WIN64", None))
def test_graph_stubs_preserve_windows_fallback_without_shadowing_linux_core(tmp_path, platform_macro):
    compiler = shutil.which("gcc")
    if compiler is None or shutil.which("nm") is None:
        pytest.skip("portable Windows-branch symbol check requires gcc/nm")
    root = Path(__file__).parents[1]
    # Exercise Windows preprocessing on the host; this is not a Windows build.
    (tmp_path / "BaseTsd.h").write_text("typedef __PTRDIFF_TYPE__ SSIZE_T;\n")
    command = [compiler, "-std=c11", "-DAIMDO_XPU", "-ffunction-sections", "-fdata-sections",
               "-I" + str(tmp_path), "-I" + str(root / "src")]
    if platform_macro:
        command += ["-D" + platform_macro, "-D__declspec(x)=", "-D__stdcall="]
    object_path = tmp_path / "stubs.o"
    subprocess.run([*command, "-c", str(root / "src-xpu/stubs.c"), "-o", str(object_path)], check=True)
    symbols = subprocess.check_output(["nm", "-g", "--defined-only", str(object_path)], text=True)
    names = {line.split()[-1] for line in symbols.splitlines()}
    graph_names = {"malloc_graph_alloc", "malloc_graph_free", "malloc_graph_sync_paused", "free_rogue"}
    if not platform_macro:
        assert not names & graph_names
        return
    assert graph_names <= names
    harness = tmp_path / "fallback.c"
    harness.write_text('''#include "plat.h"
int main(void) {
    CUdeviceptr pointer = 123;
    int result = 99;
    assert(!malloc_graph_alloc(&pointer, 8, NULL) && pointer == 123);
    assert(!malloc_graph_free(pointer, NULL, &result) && result == 99);
    assert(!malloc_graph_sync_paused());
    assert(!free_rogue(pointer, &result) && result == 99);
    return 0;
}
''')
    binary = tmp_path / "fallback"
    subprocess.run([*command, str(harness), str(object_path), "-Wl,--gc-sections", "-o", str(binary)], check=True)
    subprocess.run([str(binary)], check=True)


def test_windows_loader_imports_cover_vmm_driver_calls():
    root = Path(__file__).parents[1]
    vmm = (root / "src-xpu/vmm-manager.h").read_text()
    needed = set(re.findall(r"decltype\(&([A-Za-z0-9_]+)\)", vmm))
    exported = set((root / "src-xpu/ze_loader.def").read_text().split())
    assert needed <= exported, sorted(needed - exported)
