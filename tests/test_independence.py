"""This package must stand alone.

The whole reason `chunk` and `loops` were moved out of `polars_kir_hmm` is that they need no
trained model once the gene is supplied. That claim is easy to make and easy to break — one
convenience import of a helper and the dependency is back, silently, because the HMM package is
installed in every development environment.

So it is asserted rather than asserted-in-prose: these tests fail if this package acquires a
runtime dependency on the HMM stack, and they are the tests to read first if someone is wondering
why an import was rejected in review.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

SOURCE_ROOT = Path(__file__).resolve().parents[1]

#: Packages this one must not import at runtime. `histo_kir` is a *dev* dependency: the vendoring
#: scripts and the parity tests read it, but the built wheel must not.
FORBIDDEN = ("polars_kir_hmm", "histo_kir_hmm", "histo_kir")


def test_importing_this_package_does_not_import_the_hmm_stack():
    """The load-bearing check, run in a fresh interpreter.

    Checking `sys.modules` inside the test session would prove nothing: pytest has already
    imported `histo_kir` for the parity fixtures, so the names are present regardless. A
    subprocess with a clean interpreter is the only honest way to ask the question.
    """
    program = (
        "import sys\n"
        "import polars_kir_regions\n"
        f"leaked = [name for name in {FORBIDDEN!r} if name in sys.modules]\n"
        "print(';'.join(leaked))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", program],
        capture_output=True,
        text=True,
        check=True,
    )
    leaked = [name for name in result.stdout.strip().split(";") if name]
    assert not leaked, (
        f"importing polars_kir_regions pulled in {leaked}; this package must not depend on the "
        "HMM stack at runtime"
    )


def test_the_expressions_run_without_the_hmm_stack_importable():
    """Both expressions, in an interpreter where importing the HMM packages is made to fail.

    Stronger than the import check: a lazy `import polars_kir_hmm` inside a function would survive
    that one and die here. The blocker raises on any attempt, so the run either completes with no
    such import or fails loudly.
    """
    program = (
        "import sys\n"
        "\n"
        "class Blocker:\n"
        f"    forbidden = set({FORBIDDEN!r})\n"
        "    def find_module(self, name, path=None):\n"
        "        return self.find_spec(name, path)\n"
        "    def find_spec(self, name, path=None, target=None):\n"
        "        if name.split('.')[0] in self.forbidden:\n"
        "            raise ImportError('blocked by the independence test: ' + name)\n"
        "        return None\n"
        "\n"
        "sys.meta_path.insert(0, Blocker())\n"
        "import polars as pl\n"
        "import polars_kir_regions  # noqa: F401\n"
        "\n"
        "# A real KIR3DL1 ectodomain fragment would need a fixture; a short sequence is enough to\n"
        "# prove the kernels run, since correctness is covered by the other suites.\n"
        "frame = pl.DataFrame({'g': ['KIR3DL1'], 'sequence': ['HMGGQDKPFLSA']})\n"
        "chunked = frame.with_columns(c=pl.col('sequence').kir_regions.chunk(gene='g'))\n"
        "looped = frame.with_columns(l=pl.col('sequence').kir_regions.loops(gene='g'))\n"
        "print(chunked.unnest('c')['provenance'][0], looped.unnest('l')['gene'][0])\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", program],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"the expressions failed with the HMM stack blocked:\n{result.stderr}"
    )
    assert result.stdout.strip(), "the run produced no output"


@pytest.mark.parametrize("forbidden", FORBIDDEN)
def test_no_rust_or_python_source_references_the_hmm_packages(forbidden):
    """A grep over the shipped source, so a new import is caught at review time.

    Scoped to what ships: `src/` and `python/`. The vendoring scripts under `codegen/` and the
    parity tests legitimately read `histo_kir`, which is why they are excluded — that is the
    build-time/runtime line this package draws.
    """
    offenders = []
    for directory in ("src", "python"):
        for path in (SOURCE_ROOT / directory).rglob("*"):
            if path.suffix not in {".rs", ".py"} or not path.is_file():
                continue
            for number, line in enumerate(path.read_text().splitlines(), start=1):
                stripped = line.strip()
                # Prose mentions are fine and frequent -- the docs explain the split. Only an
                # actual import is a dependency.
                if stripped.startswith("//") or stripped.startswith("#"):
                    continue
                # Match an import *statement*, not a mention: `use histo_kir::` in Rust, or a
                # leading `import`/`from` in Python. A package name inside a string literal --
                # the docs cite `histo_kir/data/...` paths constantly -- is prose, not a
                # dependency, and an earlier version of this test failed on exactly that.
                is_import = (
                    stripped.startswith(f"use {forbidden}")
                    or stripped.startswith(f"import {forbidden}")
                    or stripped.startswith(f"from {forbidden}")
                )
                if is_import:
                    offenders.append(f"{path.relative_to(SOURCE_ROOT)}:{number}: {stripped}")
    assert not offenders, (
        f"shipped source imports {forbidden}:\n  " + "\n  ".join(offenders)
    )


def test_the_crate_does_not_depend_on_the_hmm_crate():
    """`Cargo.toml` must not list `polars-kir-hmm`, and must not need the model-loading crates."""
    manifest = (SOURCE_ROOT / "Cargo.toml").read_text()
    assert "polars-kir-hmm" not in manifest
    # `zip` and `rayon` existed only to load and score models. Their absence is what makes the
    # dependency reduction real rather than nominal, so it is pinned.
    for crate in ("zip", "rayon"):
        assert f'\n{crate} = ' not in manifest, (
            f"{crate} is back in Cargo.toml; it was only needed for model loading/scoring"
        )


def test_the_wheel_carries_its_own_vendored_data():
    """The data must be in the package directory, not read from a sibling checkout at runtime."""
    import polars_kir_regions

    assert (polars_kir_regions.region_dir() / "regions.json").exists()
    assert (polars_kir_regions.loop_dir() / "loop_windows.json").exists()
