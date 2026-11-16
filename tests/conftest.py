"""Shared fixtures. Real committed data only -- never mocks or synthetic sequences.

That is a project-wide rule and it matters here: the sequences that break a KIR classifier are
real deposited chains with disordered loops and truncated constructs, and real IPD-KIR alleles
whose ectodomains collide across genes. A synthetic sequence exercises none of that.
"""

from __future__ import annotations

import json
import pathlib
from pathlib import Path

import pytest

DATA = Path(__file__).parent / "data"


@pytest.fixture(scope="session")
def alleles() -> list[dict]:
    """All 1983 IPD-KIR 2.15.0 protein alleles, full-length precursor translations."""
    return json.loads((DATA / "ipdkir_alleles.json").read_text())["alleles"]


@pytest.fixture(scope="session")
def expressed_alleles(alleles) -> list[dict]:
    """The 1958 expressed alleles.

    IPD-KIR marks non-expressed alleles with a trailing `N`; their translations are frameshifted
    fragments at 8-34% of reference length, so rejecting them is correct behaviour and mixing them
    in understates accuracy (`histo_kir_hmm` invariant 9).
    """
    return [a for a in alleles if not a["is_null"]]


@pytest.fixture(scope="session")
def structural() -> dict:
    """The framed-structure corpus: 54 KIR chains, 22 non-KIR chains, tied ectodomains."""
    return json.loads((DATA / "structural_chains.json").read_text())


@pytest.fixture(scope="session")
def kir_chains(structural) -> list[dict]:
    """The 54 observed KIR chain sequences, with the gene from `loop_annotations.csv`."""
    return structural["kir_chains"]


@pytest.fixture(scope="session")
def non_kir_chains(structural) -> list[dict]:
    """All 122 non-KIR chains across the 54 framed files.

    This is the scope `histo_kir_hmm/VERIFICATION_REPORT.md` reports as 122/122 rejected: 32 MHC
    class I heavy chains, 32 beta-2 microglobulins, 32 peptides and 26 other partners. Only 60 of
    the sequences are distinct -- the same heavy chain, b2m and peptide recur across entries -- and
    the duplicates are kept so the count matches the reference's.
    """
    return structural["non_kir_chains"]


@pytest.fixture(scope="session")
def non_kir_curated(structural) -> list[dict]:
    """The 22 curated negatives: one per structural partner class, plus the UniProt hard cases.

    LILRB1-4, LILRA5, LAIR1, FCGR2A, TYROBP, NKp44, NKp46, CD226 and NKG2D appear in no structure
    file, so they are only reachable through this fixture. LILRB1 is the load-bearing one: a
    four-Ig-domain MHC class I-binding NK receptor in the same superfamily as KIR.
    """
    return structural["non_kir_curated"]


@pytest.fixture(scope="session")
def tied_ectodomains(structural) -> dict[str, str]:
    """Allele -> ectodomain for alleles whose ectodomain collides with another gene's."""
    return structural["tied_ectodomains"]


@pytest.fixture(scope="session")
def region_parquet():
    """The Phase 0 region table, as the normative boundaries.

    Read from `histo_kir/data/` in the repo rather than vendored into `tests/data/`: it is Phase
    0's deliverable and this package's build input, so a test that disagreed with it would be
    reporting drift between the vendored copy and its source -- which is exactly what we want to
    detect.
    """
    import polars as pl

    path = (
        pathlib.Path(__file__).resolve().parents[2]
        / "histo_kir"
        / "data"
        / "kir_allele_regions.parquet"
    )
    if not path.exists():
        pytest.skip(f"the region table is not at {path}")
    return pl.read_parquet(path)


@pytest.fixture(scope="session")
def gene_reference():
    """`histo_kir`'s gene reference: structurally-anchored mature sequences and coordinates."""
    import polars as pl

    path = (
        pathlib.Path(__file__).resolve().parents[2]
        / "histo_kir"
        / "data"
        / "kir_gene_reference.csv"
    )
    if not path.exists():
        pytest.skip(f"the gene reference is not at {path}")
    return pl.read_csv(path)


@pytest.fixture(scope="session")
def numbering():
    """`histo_kir.numbering`, or a skip -- the reference for the alignment transfer.

    A DEV dependency only. This package imports nothing from `histo_kir` at runtime; the parity
    tests use it to check that the transcribed aligner still reproduces the reference's mappings.
    """
    pytest.importorskip("histo_kir", reason="parity requires histo_kir")
    from histo_kir import numbering

    return numbering
