"""The `loops` expression, against real committed fixtures only.

Two things are load-bearing here and each gets a test that fails loudly rather than quietly.

**The numbering frame.** The loop windows are mature numbering counted from the signal-peptide
cleavage site; the exon-derived leader is 2 residues longer in all 15 genes, so using it as the
origin shifts every window by 2 and still returns the right number of plausible residues. Nothing
downstream would flag it. `test_frame_is_the_cleavage_site` asserts BOTH directions — the right
origin reproduces every published window and the wrong one reproduces none — because a test that
only confirms the right answer would still pass if the two origins ever converged.

**The gene is an input.** This package has no classifier. Every test supplies the gene, and
`test_unknown_gene_*` pins that an unusable gene yields nulls and a note rather than a guess.
"""
from __future__ import annotations

import csv
import gzip
import json
from pathlib import Path

import polars as pl
import pytest

import polars_kir_regions  # noqa: F401  (registers the .kir_regions namespace)
from polars_kir_regions import loop_windows

DATA = Path(__file__).parent / "data"
KIRS = Path(__file__).resolve().parent.parent.parent

ELEMENTS = ("A'B", "CC'", "EF", "hinge", "BC", "FG", "D0a", "D0b", "CC'ext")
FIELDS = ("sequence", "status", "start", "end", "n_observed", "n_expected", "matches_reference")
D1_LOOPS = ("A'B", "CC'", "EF", "CC'ext")
KIR3D_ONLY = ("D0a", "D0b", "CC'ext")


@pytest.fixture(scope="session")
def golden() -> dict:
    """The histo_kir-derived expected records: 1983 alleles and 54 deposited chains."""
    with gzip.open(DATA / "loops_golden.json.gz", "rt") as handle:
        return json.load(handle)


@pytest.fixture(scope="session")
def windows() -> pl.DataFrame:
    return loop_windows()


@pytest.fixture(scope="session")
def gene_reference() -> dict:
    with open(KIRS / "histo_kir/data/kir_gene_reference.csv") as handle:
        return {row["gene"]: row for row in csv.DictReader(handle)}


@pytest.fixture(scope="session")
def vendored() -> dict:
    path = (Path(polars_kir_regions.loop_dir()) / "loop_windows.json")
    return json.loads(path.read_text())


def slice_loops(sequences: list[str], genes: list[str]) -> pl.DataFrame:
    """Run the expression with the gene supplied, as this package always requires."""
    return pl.DataFrame({"gene": genes, "sequence": sequences}).with_columns(
        loops=pl.col("sequence").kir_regions.loops(gene="gene")
    )


# --------------------------------------------------------------------------------------------
# The frame
# --------------------------------------------------------------------------------------------


def test_frame_is_the_cleavage_site(vendored, alleles, gene_reference):
    """Every published window is reproduced from the cleavage site, and NONE from the exon leader.

    Both directions are asserted. The exon-leader origin is wrong by exactly 2 in all 15 genes,
    which is invisible in the output — the slice is the right length and looks like a loop — so
    only a paired assertion catches it.
    """
    by_allele = {a["allele"]: a["sequence"] for a in alleles}
    regions = pl.read_parquet(KIRS / "histo_kir/data/kir_allele_regions.parquet")
    exon_leader = {r["allele"]: r["leader_end_exon_frame"]
                   for r in regions.select("allele", "leader_end_exon_frame").iter_rows(named=True)}

    from_cleavage = from_leader = total = 0
    for gene, entry in vendored["genes"].items():
        allele = entry["reference_allele"]
        precursor = by_allele[allele]
        signal = entry["signal_peptide_length"]
        assert signal == int(gene_reference[gene]["signal_peptide_length"]), gene
        # The +2 that makes the wrong origin wrong.
        assert exon_leader[allele] - signal == 2, gene
        for element in entry["elements"]:
            if element["kind"] != "published_loop":
                continue
            start, end = element["mature_start"], element["mature_end"]
            total += 1
            from_cleavage += precursor[signal + start - 1 : signal + end] == element["sequence"]
            from_leader += (
                precursor[exon_leader[allele] + start - 1 : exon_leader[allele] + end]
                == element["sequence"]
            )

    assert total == 81, f"expected 81 published windows, saw {total}"
    assert from_cleavage == total, f"cleavage-site origin reproduced only {from_cleavage}/{total}"
    assert from_leader == 0, (
        f"the exon-leader origin reproduced {from_leader}/{total} windows; it should reproduce "
        "none. If this fires, the two origins have converged and the frame is no longer pinned."
    )


def test_vendored_windows_match_their_own_coordinates(vendored, gene_reference):
    """Each recorded window sequence equals the slice its own coordinates name.

    This is the check that would catch a future histo_kir release shifting the frame: it compares
    the table against itself in the precursor frame, so a re-vendored table in the wrong frame
    fails here rather than silently producing shifted loops.
    """
    checked = 0
    for gene, entry in vendored["genes"].items():
        signal = entry["signal_peptide_length"]
        assert signal == int(gene_reference[gene]["signal_peptide_length"]), gene
        for element in entry["elements"]:
            start = signal + element["mature_start"] - 1
            end = signal + element["mature_end"]
            assert entry["precursor"][start:end] == element["sequence"], (gene, element["name"])
            checked += 1
    assert checked == 93, f"expected 93 windows, checked {checked}"


def test_every_element_inside_ectodomain(vendored, gene_reference):
    """All nine elements lie inside D0+D1+D2, so the ectodomain scope costs no loop.

    Recomputes the total, the per-kind breakdown and the inside count from the shipped table
    rather than asserting a remembered figure.
    """
    regions = pl.read_parquet(KIRS / "histo_kir/data/kir_allele_regions.parquet")
    d2_end = {r["allele"]: r["D2_end"]
              for r in regions.select("allele", "D2_end").iter_rows(named=True)}

    by_kind: dict[str, int] = {}
    checked = inside = 0
    for gene, entry in vendored["genes"].items():
        signal = entry["signal_peptide_length"]
        mature_d2_end = d2_end[entry["reference_allele"]] - signal
        for element in entry["elements"]:
            by_kind[element["kind"]] = by_kind.get(element["kind"], 0) + 1
            checked += 1
            inside += element["mature_end"] <= mature_d2_end

    assert checked == 93, f"expected 93 element windows, counted {checked}"
    assert by_kind == {"published_loop": 81, "d0_patch": 8, "ccprime_extension": 4}, by_kind
    assert inside == checked, f"only {inside}/{checked} elements end at or before the D2 end"


# --------------------------------------------------------------------------------------------
# Parity with histo_kir
# --------------------------------------------------------------------------------------------


def compare_against_golden(frame: pl.DataFrame, expected: dict, key: str) -> tuple[int, list]:
    exprs = [pl.col(key)]
    for element in ELEMENTS:
        for field in FIELDS:
            exprs.append(
                pl.col("loops").struct.field(element).struct.field(field)
                .alias(f"{element}|{field}")
            )
    agree = 0
    disagree = []
    for row in frame.select(exprs).iter_rows(named=True):
        record = expected[row[key]]
        for element in ELEMENTS:
            want = record["elements"].get(element)
            if want is None:
                # Absent from this architecture: a NULL STRUCT, not a struct of nulls.
                if row[f"{element}|status"] is None:
                    agree += 1
                else:
                    disagree.append((row[key], element, "emitted but architecturally absent"))
                continue
            bad = {f: (row[f"{element}|{f}"], want[f])
                   for f in FIELDS if row[f"{element}|{f}"] != want[f]}
            if bad:
                disagree.append((row[key], element, bad))
            else:
                agree += 1
    return agree, disagree


def test_gene_reference_alleles_match_histo_kir(golden, gene_reference, vendored):
    """The 15 gene reference alleles reproduce histo_kir's records exactly."""
    referenced = [entry["reference_allele"] for entry in vendored["genes"].values()]
    by_allele = {a: golden["alleles"][a] for a in referenced}
    frame = slice_loops(
        [vendored["genes"][g]["precursor"] for g in sorted(vendored["genes"])],
        sorted(vendored["genes"]),
    ).with_columns(allele=pl.Series(referenced))
    agree, disagree = compare_against_golden(frame, by_allele, "allele")
    assert not disagree, disagree[:5]
    assert agree == 15 * 9, agree


@pytest.mark.slow
def test_full_allele_corpus_matches_histo_kir(golden, alleles):
    """All 1983 IPD-KIR alleles, every field, against the histo_kir-derived golden."""
    frame = slice_loops([a["sequence"] for a in alleles],
                        [a["gene"] for a in alleles]).with_columns(
        allele=pl.Series([a["allele"] for a in alleles])
    )
    agree, disagree = compare_against_golden(frame, golden["alleles"], "allele")
    assert not disagree, disagree[:5]
    assert agree == len(alleles) * len(ELEMENTS), agree


def test_deposited_chains_match_histo_kir(golden, structural):
    """Every deposited KIR chain, including the truncated and disordered ones."""
    chains = structural["kir_chains"]
    frame = slice_loops([c["sequence"] for c in chains],
                        [c["gene"] for c in chains]).with_columns(
        file=pl.Series([c["file"] for c in chains])
    )
    agree, disagree = compare_against_golden(frame, golden["chains"], "file")
    assert not disagree, disagree[:5]
    assert agree == len(chains) * len(ELEMENTS), agree


# --------------------------------------------------------------------------------------------
# The gene is an input, never an inference
# --------------------------------------------------------------------------------------------


def test_literal_gene_equals_gene_column(structural):
    """A literal gene and the same gene in a column must give identical results."""
    chains = [c for c in structural["kir_chains"] if c["gene"] == "KIR3DL1"]
    assert chains, "no KIR3DL1 chains in the fixture"
    sequences = [c["sequence"] for c in chains]

    literal = pl.DataFrame({"sequence": sequences}).with_columns(
        loops=pl.col("sequence").kir_regions.loops(gene="KIR3DL1")
    )
    column = slice_loops(sequences, ["KIR3DL1"] * len(sequences))
    for a, b in zip(literal.iter_rows(named=True), column.iter_rows(named=True)):
        for element in ELEMENTS:
            x, y = a["loops"][element], b["loops"][element]
            assert (x is None) == (y is None), element
            if x is not None:
                assert x["sequence"] == y["sequence"], element
                assert x["start"] == y["start"], element


def test_unknown_gene_yields_nulls_and_a_note(structural):
    """An unrecognised gene must produce nulls plus an explanatory note, never a guess."""
    chain = structural["kir_chains"][0]
    out = slice_loops([chain["sequence"]], ["KIR9DL9"])
    row = out["loops"][0]
    assert row["gene"] is None
    assert row["architecture"] is None
    # `"none"` rather than null, matching `chunk`'s PROVENANCE_NONE: the row was processed and
    # nothing was established, which is a different statement from "not asked".
    assert row["provenance"] == "none"
    for element in ELEMENTS:
        assert row[element] is None, element
    assert row["notes"], "an unknown gene must explain itself"
    assert "KIR9DL9" in row["notes"][0]


def test_null_gene_yields_nulls_and_a_note(structural):
    """A null gene is not an error and not a guess: nulls plus a note."""
    chain = structural["kir_chains"][0]
    out = pl.DataFrame({"gene": [None], "sequence": [chain["sequence"]]},
                       schema={"gene": pl.String, "sequence": pl.String}).with_columns(
        loops=pl.col("sequence").kir_regions.loops(gene="gene")
    )
    row = out["loops"][0]
    assert row["gene"] is None
    for element in ELEMENTS:
        assert row[element] is None, element
    assert row["notes"] and "no gene supplied" in row["notes"][0]


def test_tie_set_is_reported_not_arbitrated(structural):
    """A `;`-separated gene value must be refused, not silently resolved to the first name.

    127 IPD-KIR alleles have ectodomains that collide across gene pairs, so a tie is a real state
    upstream. This package does not arbitrate it — taking the first name would be a wrong answer
    presented as a confident one.
    """
    chain = next(c for c in structural["kir_chains"] if c["gene"] == "KIR2DL2")
    out = slice_loops([chain["sequence"]], ["KIR2DL2;KIR2DL3"])
    row = out["loops"][0]
    assert row["gene"] is None
    for element in ELEMENTS:
        assert row[element] is None, element
    assert row["notes"] and "more than one gene" in row["notes"][0]

    # And the same sequence with either gene alone does resolve, so the refusal is about the tie.
    for gene in ("KIR2DL2", "KIR2DL3"):
        resolved = slice_loops([chain["sequence"]], [gene])["loops"][0]
        assert resolved["gene"] == gene
        assert resolved["FG"] is not None


def test_null_sequence_yields_a_null_struct(structural):
    """A null sequence gives a null struct, not a struct of nulls."""
    out = pl.DataFrame({"gene": ["KIR3DL1"], "sequence": [None]},
                       schema={"gene": pl.String, "sequence": pl.String}).with_columns(
        loops=pl.col("sequence").kir_regions.loops(gene="gene")
    )
    assert out["loops"][0] is None


# --------------------------------------------------------------------------------------------
# Architecture, nulls and provenance
# --------------------------------------------------------------------------------------------


def test_d0d2_genes_have_no_d1_loops(vendored, structural):
    """The three D1 loops do not exist for a D0D2 receptor, and D0 windows stay undetermined."""
    d0d2 = [g for g, e in vendored["genes"].items() if e["architecture"] == "D0D2"]
    assert d0d2, "no D0D2 gene in the table"
    chain = next(c for c in structural["kir_chains"] if c["gene"] in d0d2)
    row = slice_loops([chain["sequence"]], [chain["gene"]])["loops"][0]
    for element in ("A'B", "CC'", "EF"):
        assert row[element] is None, element
    for element in KIR3D_ONLY:
        assert row[element] is None, element
    for element in ("hinge", "BC", "FG"):
        assert row[element] is not None, element
    assert any("D0D2" in note for note in row["notes"])


def test_kir3d_only_elements_only_for_three_domain_genes(vendored, structural):
    """`D0a`, `D0b` and `CC'ext` are emitted only where the architecture is D0D1D2."""
    for chain in structural["kir_chains"]:
        row = slice_loops([chain["sequence"]], [chain["gene"]])["loops"][0]
        three_domain = vendored["genes"][chain["gene"]]["architecture"] == "D0D1D2"
        for element in KIR3D_ONLY:
            if three_domain:
                assert row[element] is not None, (chain["file"], element)
            else:
                assert row[element] is None, (chain["file"], element)


def test_partial_window_is_null_not_truncated(golden, structural):
    """A window observed in part emits a NULL sequence, never a shortened slice.

    `8tui_A` is a lone KIR2DL3 D1 domain: its D2 loops are genuinely absent. Emitting the residues
    that happen to be present would be a plausible-looking wrong answer.
    """
    chain = next(c for c in structural["kir_chains"] if c["file"] == "8tui_A.pdb")
    row = slice_loops([chain["sequence"]], [chain["gene"]])["loops"][0]
    for element in ("hinge", "BC", "FG"):
        got = row[element]
        assert got is not None, element
        assert got["status"] in {"absent", "partial"}, (element, got["status"])
        assert got["sequence"] is None, (element, got["sequence"])
        assert got["start"] is None and got["end"] is None, element
        assert got["n_observed"] < got["n_expected"], element


def test_provenance_distinguishes_annotated_from_inferred(vendored, structural):
    """`annotated` when the sequence IS the gene reference precursor; `inferred` for a transfer."""
    gene = "KIR3DL1"
    reference = vendored["genes"][gene]["precursor"]
    row = slice_loops([reference], [gene])["loops"][0]
    assert row["provenance"] == "annotated"

    chain = next(c for c in structural["kir_chains"] if c["gene"] == gene)
    transferred = slice_loops([chain["sequence"]], [gene])["loops"][0]
    assert transferred["provenance"] == "inferred"


def test_5prime_partial_allele_keeps_its_n_terminal_windows(golden, alleles):
    """A 5'-partial allele must not lose N-terminal windows to a fixed-offset assumption.

    `KIR2DL2*011` begins inside the signal peptide. Stripping `signal_peptide_length` characters
    off the query would consume real mature residues; aligning to the reference PRECURSOR does not.
    """
    allele = next(a for a in alleles if a["allele"] == "KIR2DL2*011")
    row = slice_loops([allele["sequence"]], [allele["gene"]])["loops"][0]
    expected = golden["alleles"]["KIR2DL2*011"]["elements"]
    assert expected["A'B"]["status"] == "complete", "fixture no longer exercises the recovery"
    assert row["A'B"] is not None and row["A'B"]["sequence"] == expected["A'B"]["sequence"]


# --------------------------------------------------------------------------------------------
# The introspection helper
# --------------------------------------------------------------------------------------------


def test_loop_windows_table_shape(windows, vendored):
    """`loop_windows()` exposes every vendored window, with the evidence class visible."""
    assert windows.height == sum(len(e["elements"]) for e in vendored["genes"].values())
    assert windows.height == 93
    assert set(windows["kind"].unique()) == {"published_loop", "d0_patch", "ccprime_extension"}
    assert (windows["length"] == windows["mature_end"] - windows["mature_start"] + 1).all()
    # `predicted_only` must not be uniformly true or false: it is the whole point of the column.
    values = set(windows["predicted_only"].to_list())
    assert values == {True, False}, values


def test_predicted_only_marks_genes_without_structure(windows, structural):
    """`predicted_only` is False exactly for the genes with a deposited chain in the corpus."""
    with_structure = {c["gene"] for c in structural["kir_chains"]}
    for row in windows.iter_rows(named=True):
        assert row["predicted_only"] == (row["gene"] not in with_structure), row["gene"]


def test_expression_and_table_agree_on_the_reference(windows, vendored):
    """The sliced sequence for each gene reference allele equals the table's recorded sequence."""
    for gene, entry in sorted(vendored["genes"].items()):
        row = slice_loops([entry["precursor"]], [gene])["loops"][0]
        for element_row in windows.filter(pl.col("gene") == gene).iter_rows(named=True):
            got = row[element_row["element"]]
            assert got is not None, (gene, element_row["element"])
            assert got["sequence"] == element_row["sequence"], (gene, element_row["element"])
            assert got["matches_reference"], (gene, element_row["element"])


def test_no_hmm_dependency():
    """The package must not IMPORT `polars_kir_hmm` or `histo_kir_hmm`.

    This is the whole reason the package exists, so it is asserted rather than assumed. The test
    parses the import statements rather than grepping the source: the docstrings legitimately name
    `polars_kir_hmm` when pointing a caller who does not know the gene at the classifier, and a
    substring search would fail on that mention while passing a real import written as
    `importlib.import_module`.
    """
    import ast
    import sys

    forbidden = {"polars_kir_hmm", "histo_kir_hmm", "histo_kir"}
    module = sys.modules["polars_kir_regions"]
    tree = ast.parse(Path(module.__file__).read_text())

    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    assert not (imported & forbidden), f"the package imports {imported & forbidden}"

    # The built extension must not link the other package's symbols either.
    plugin = Path(polars_kir_regions.loop_dir()).parent
    natives = list(plugin.glob("_internal*.so")) + list(plugin.glob("_internal*.pyd"))
    assert natives, f"no built extension found in {plugin}"
    for native in natives:
        blob = native.read_bytes()
        assert b"polars_kir_hmm" not in blob, native.name
        assert b"histo_kir_hmm" not in blob, native.name


def test_rust_sources_carry_no_hmm_imports():
    """No `use crate::classify` / `use crate::models` survives in the crate's loop modules.

    The pre-split version of `loops_expr.rs` imported the classifier for its gene-call path. If a
    future edit reintroduces that, this crate stops being standalone — and because `classify.rs`
    does not exist here it would fail to compile, so this test is really a guard against someone
    "fixing" that by copying the module across.
    """
    crate = Path(__file__).resolve().parent.parent / "src"
    if not crate.exists():  # installed wheel, no source tree
        pytest.skip("no crate source tree next to the tests")
    for name in ("loops.rs", "loops_expr.rs", "align.rs", "sequence.rs"):
        source = (crate / name).read_text()
        assert "crate::classify" not in source, name
        assert "crate::models" not in source, name
        assert "crate::hmm" not in source, name
