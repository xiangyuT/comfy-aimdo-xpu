import os
import sys
from setuptools import setup, Distribution

# This repository is not a branch of the upstream release line. It is upstream
# plus a Windows/XPU port, published under the same distribution name, so the
# version has to say which upstream release it carries and which tree built it.
#
# setuptools-scm's default answers neither. `guess-next-dev` produces
# `0.4.14.dev41`, which asserts a release that does not exist and counts commits
# on a branch upstream will never ship, so an unrelated branch at the same
# distance yields the same version for different code. The count also advances
# for commits that change no shipped file.
#
# A PEP 440 local version identifier is the construct for this case: a
# modification of an upstream release. The base names the newest release in the
# tree, and the local segment names the port and the commit.
#
# These live here rather than in pyproject.toml because setuptools-scm resolves
# a scheme name through the entry points of installed distributions, and a
# module in an unbuilt source tree is not importable from the build backend.

def _upstream_release(version):
    """The newest upstream release this tree contains."""
    return str(version.tag)

def _port_build(version):
    """Identify the port and the exact commit that built the artifact."""
    parts = ["xpu"]
    if version.node:
        parts.append(version.node[:8])
    if version.dirty:
        parts.append("dirty")
    return "+" + ".".join(parts)

# This trick forces the wheel to be labeled with the platform (e.g., win_amd64)
# instead of "any", which is required for binary DLLs.
class BinaryDistribution(Distribution):
    def has_ext_modules(self):
        return any(os.path.exists(path) for path in (
            "comfy_aimdo/aimdo.so",
            "comfy_aimdo/aimdo_xpu.so",
            "comfy_aimdo/aimdo.dll",
            "comfy_aimdo/aimdo_xpu.dll",
        ))
    def get_tag(self):
        t = super().get_tag()
        return ("cp39", "abi3", t[2])

setup(
    distclass=BinaryDistribution,
    options={"bdist_wheel": {"py_limited_api": "cp39"}},
    use_scm_version={
        "version_file": "comfy_aimdo/_version.py",
        "version_scheme": _upstream_release,
        "local_scheme": _port_build,
    },
)
