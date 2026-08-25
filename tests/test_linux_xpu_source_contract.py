from pathlib import Path


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
