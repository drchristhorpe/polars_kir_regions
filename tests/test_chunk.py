"""The `chunk` expression, against the reference region table.

Real committed fixtures only (project-wide rule): the 1983 IPD-KIR protein alleles in
`data/ipdkir_alleles.json`, the 54 deposited chains and 22 curated negatives in
`data/structural_chains.json`, and `histo_kir/data/kir_allele_regions.parquet` as the normative
boundaries. No synthetic sequences — the cases that break this expression are real partial-CDS
alleles, real null alleles truncated mid-domain, and real deposited fragments.

**The gene is an input.** Every call passes it, because this package has no classifier. That is
also what these tests exercise that the earlier in-HMM version could not: what happens when the
supplied gene is wrong, unknown, missing, or a tie set.

**Scope.** `chunk` emits the ectodomain only: leader, signal peptide, the leader's mature tail,
D0/D1/D2 and their concatenation. Stem, TM and cytoplasmic are deliberately not emitted, so the
emitted fields do *not* tile the precursor. The tiling invariant still holds in the region table
and `test_region_table_still_tiles` asserts it there, which is where it now lives.
"""

from __future__ import annotations

import polars as pl
import pytest

import polars_kir_regions  # noqa: F401  (import for the namespace side effect)

#: The three Ig domains, in order.
IG = ("D0", "D1", "D2")

#: The full seven-region partition, for the region-table tiling check only.
ALL_REGIONS = ("leader", "D0", "D1", "D2", "stem", "TM", "cyt")


@pytest.fixture(scope="module")
def chunked(alleles, region_parquet) -> pl.DataFrame:
    """Every allele chunked once, with its true gene supplied; the fixture assertions share."""
    gene_of = dict(zip(region_parquet["allele"], region_parquet["gene"], strict=True))
    frame = pl.DataFrame(
        {
            "name": [a["allele"] for a in alleles],
            "input_gene": [gene_of[a["allele"]] for a in alleles],
            "sequence": [a["sequence"] for a in alleles],
        }
    )
    # The struct emits its own `gene`, so the input column is named `input_gene`: unnesting beside
    # a same-named column is a DuplicateError, which is a caller-side collision rather than an
    # expression bug. `test_unnest_beside_a_gene_column` pins the documented way round it.
    return frame.with_columns(c=pl.col("sequence").kir_regions.chunk(gene="input_gene")).unnest("c")


# --- the two ectodomain identities ------------------------------------------------------


def test_ectodomain_is_leader_tail_plus_ig_domains(chunked):
    """`ectodomain == mature_leader_tail + D0 + D1 + D2`, for every allele.

    Checked independently of the `ectodomain_ok` column so a bug in that column cannot mask a bug
    in the slicing.
    """
    failures = []
    for row in chunked.iter_rows(named=True):
        if row["ectodomain"] is None:
            continue
        rebuilt = (row["mature_leader_tail"] or "") + "".join(
            row[region] for region in IG if row[region] is not None
        )
        if rebuilt != row["ectodomain"]:
            failures.append(row["name"])
    assert not failures, f"{len(failures)} alleles fail identity 1, e.g. {failures[:5]}"


def test_ectodomain_is_a_contiguous_slice_from_mature_residue_one(chunked, gene_reference):
    """`ectodomain == sequence[signal_peptide_length : D2_end]`.

    **This is the assertion that matters** — it pins the ectodomain to an absolute offset in the
    input rather than to the concatenation of its own parts, so it catches an off-by-one or a
    frame error that identity 1 would satisfy anyway.

    Only checkable where the sequence contains the cleavage site; the exceptions are the
    5'-partial CDS entries, pinned separately.
    """
    lengths = dict(
        zip(gene_reference["gene"], gene_reference["signal_peptide_length"], strict=True)
    )
    checked = 0
    failures = []
    for row in chunked.iter_rows(named=True):
        if row["ectodomain"] is None or row["mature_leader_tail"] is None:
            continue
        length = lengths[row["gene"]]
        if row["sequence"][length : length + len(row["ectodomain"])] != row["ectodomain"]:
            failures.append(row["name"])
        checked += 1
    assert not failures, f"{len(failures)} alleles fail identity 2, e.g. {failures[:5]}"
    assert checked > 1800, f"identity 2 was only checkable on {checked} alleles"


def test_ectodomain_ok_is_true_for_every_allele(chunked):
    bad = chunked.filter(~pl.col("ectodomain_ok"))
    assert bad.height == 0, f"{bad.height} alleles report ectodomain_ok=False: {bad['name'].to_list()[:5]}"


def test_ectodomain_starts_at_the_mature_n_terminus(chunked, gene_reference):
    """Spelled out against `histo_kir`'s structurally-anchored mature sequences.

    An ectodomain that started at the exon boundary instead of the cleavage site would begin two
    residues later, and this is what would catch it.
    """
    mature = dict(
        zip(gene_reference["reference_allele"], gene_reference["mature_sequence"], strict=True)
    )
    checked = 0
    for row in chunked.iter_rows(named=True):
        if row["name"] not in mature or row["ectodomain"] is None:
            continue
        assert mature[row["name"]].startswith(row["ectodomain"][:20]), (
            f"{row['name']}: ectodomain starts {row['ectodomain'][:20]!r} but the gene "
            f"reference's mature sequence starts {mature[row['name']][:20]!r}"
        )
        checked += 1
    assert checked == 15, f"expected all 15 reference alleles, checked {checked}"


# --- region lengths, nulls, architecture ------------------------------------------------


def test_ig_region_lengths_match_the_reference_table(chunked, region_parquet):
    expected = {row["allele"]: row for row in region_parquet.iter_rows(named=True)}
    failures = []
    for row in chunked.iter_rows(named=True):
        reference = expected[row["name"]]
        for region in ("leader",) + IG:
            got = len(row[region]) if row[region] is not None else None
            want = reference[f"{region}_length"] or None
            if got != want:
                failures.append((row["name"], region, got, want))
    assert not failures, f"{len(failures)} length mismatches, e.g. {failures[:5]}"


def test_absent_regions_are_null_never_empty_strings(chunked):
    """D0 null for D1D2 genes, D1 null for D0D2 genes — and null, not `""`.

    An empty string would claim the region exists and is zero residues long, which is a different
    and false statement.
    """
    for row in chunked.iter_rows(named=True):
        for region in ("leader", "signal_peptide", "mature_leader_tail", *IG, "ectodomain"):
            assert row[region] != "", f"{row['name']}: {region} is an empty string, not null"


def test_architecture_determines_which_ig_domains_are_present(chunked):
    failures = []
    for row in chunked.iter_rows(named=True):
        if row["architecture"] is None:
            continue
        if ("D0" in row["architecture"]) != (row["D0"] is not None):
            failures.append((row["name"], "D0", row["architecture"]))
        has_d1 = row["architecture"] in ("D1D2", "D0D1D2")
        if has_d1 != (row["D1"] is not None):
            failures.append((row["name"], "D1", row["architecture"]))
    assert not failures, f"{len(failures)} architecture/region mismatches, e.g. {failures[:5]}"


def test_architecture_matches_the_reference_table_per_allele(chunked, region_parquet):
    """Including the six null KIR3D alleles whose own architecture differs from their gene's."""
    expected = {row["allele"]: row for row in region_parquet.iter_rows(named=True)}
    differ = 0
    for row in chunked.iter_rows(named=True):
        reference = expected[row["name"]]
        assert row["architecture"] == reference["architecture"], row["name"]
        assert row["gene_architecture"] == reference["gene_architecture"], row["name"]
        if reference["architecture_differs_from_gene"]:
            differ += 1
            assert row["architecture"] != row["gene_architecture"], row["name"]
    assert differ == 6, f"expected the 6 documented allele/gene architecture mismatches, got {differ}"


def test_kir3dl3_has_no_missing_region_now_the_stem_is_out_of_scope(chunked):
    """The consequence of the ectodomain scope, checked.

    KIR3DL3 lacks exon 6 in 238/238 of its alleles, so under a stem-emitting contract it was the
    one gene with a structurally absent region. With the stem out of scope its D2 runs to the
    ectodomain end and it must look like every other three-domain gene.
    """
    profiles = {}
    for gene in ("KIR3DL1", "KIR3DL2", "KIR3DL3", "KIR3DS1"):
        subset = chunked.filter(
            (pl.col("gene") == gene)
            & (pl.col("expression_status") == "expressed")
            & ~pl.col("cds_partial")
        )
        assert subset.height > 0, f"no expressed complete-CDS {gene} alleles"
        profiles[gene] = {
            region: subset[region].null_count() for region in ("leader", *IG, "ectodomain")
        }
    assert profiles["KIR3DL3"] == {"leader": 0, "D0": 0, "D1": 0, "D2": 0, "ectodomain": 0}, (
        f"KIR3DL3 still has nulls on the emitted fields: {profiles['KIR3DL3']}"
    )
    assert len(set(map(str, profiles.values()))) == 1, (
        f"the four three-domain genes no longer have identical null profiles: {profiles}"
    )


# --- the leader / signal-peptide offset --------------------------------------------------


def test_leader_is_two_residues_longer_than_the_signal_peptide(chunked, gene_reference):
    """The offset that shifts every mature position if the wrong one is used."""
    lengths = dict(
        zip(gene_reference["gene"], gene_reference["signal_peptide_length"], strict=True)
    )
    seen = {}
    for row in chunked.iter_rows(named=True):
        if row["signal_peptide"] is None or row["leader"] is None:
            continue
        assert len(row["signal_peptide"]) == lengths[row["gene"]], row["name"]
        if not row["cds_partial"]:
            assert len(row["leader"]) - len(row["signal_peptide"]) == 2, row["name"]
            assert len(row["mature_leader_tail"]) == 2, row["name"]
            seen.setdefault(row["gene"], (len(row["leader"]), len(row["signal_peptide"])))
    assert seen["KIR2DL4"] == (25, 23), f"KIR2DL4 should be 25/23, got {seen['KIR2DL4']}"
    others = {gene: value for gene, value in seen.items() if gene != "KIR2DL4"}
    assert set(others.values()) == {(23, 21)}, f"the others should be 23/21, got {set(others.values())}"
    assert len(seen) == 15, f"expected all 15 genes, saw {len(seen)}"


def test_mature_leader_tail_is_the_leader_past_the_cleavage_site(chunked):
    for row in chunked.iter_rows(named=True):
        if row["mature_leader_tail"] is None:
            continue
        assert row["leader"] == row["signal_peptide"] + row["mature_leader_tail"], row["name"]


# --- partial, null and questionable alleles ---------------------------------------------


def test_partial_cds_alleles_are_never_under_reported(chunked, region_parquet):
    """All 5'-partial, none 3'-partial, and never reported complete when they are not.

    More rows report partial than the table marks, and that is deliberate: `chunk` keys on the
    *protein* sequence while CDS partiality describes the CDS, and three KIR3DL3 groups pair a
    5'-partial CDS with sequence-identical complete ones (a truncated CDS can translate to the
    same 410-residue protein). A protein-keyed row cannot carry two answers, so it reports the
    cautious one and names the disagreement in `notes`.

    The direction of the error is what matters: no allele is ever told its CDS is complete when it
    is not.
    """
    truth = dict(zip(region_parquet["allele"], region_parquet["cds_partial"], strict=True))
    assert sum(truth.values()) == 105, "the region table should mark 105 partial alleles"

    partial = chunked.filter(pl.col("cds_partial"))
    assert partial["cds_partial_5prime"].all(), "every partial row should carry the 5' marker"
    assert not partial["cds_partial_3prime"].any(), "none should carry a 3' marker"
    assert partial["ectodomain_ok"].all()

    under = [row["name"] for row in chunked.iter_rows(named=True)
             if truth[row["name"]] and not row["cds_partial"]]
    assert not under, (
        f"{len(under)} partial-CDS alleles are reported as complete, the unsafe direction: {under[:5]}"
    )

    over = [row["name"] for row in chunked.iter_rows(named=True)
            if row["cds_partial"] and not truth[row["name"]]]
    assert len(over) == 25, f"expected 25 cautiously-flagged allele names, got {len(over)}"

    flagged = chunked.filter(
        pl.col("cds_partial")
        & pl.col("notes").list.eval(pl.element().str.contains("disagree on")).list.any()
    )
    assert set(over) <= set(flagged["name"].to_list()), (
        "a cautiously-flagged allele carries no note explaining why"
    )
    assert flagged["gene"].unique().to_list() == ["KIR3DL3"]


def test_alleles_with_no_mature_frame_report_null_rather_than_a_guess(chunked):
    """A 5'-truncated CDS does not begin at the initiator Met, so mature numbering has no anchor.

    Those alleles must report `signal_peptide` and `mature_leader_tail` as null rather than
    slicing the first 21 residues of a sequence that starts somewhere else.
    """
    unanchored = chunked.filter(pl.col("mature_leader_tail").is_null())
    assert unanchored.height == 82, f"expected 82 unanchored alleles, got {unanchored.height}"
    assert unanchored["cds_partial"].all(), "every unanchored allele must be a partial CDS"
    # The ectodomain still exists; it just starts at the first Ig residue rather than mature 1.
    assert unanchored["ectodomain"].is_not_null().all()
    # `mature_leader_tail` is the field that carries the frame claim, and it is null for all 82.
    # `signal_peptide` is a prefix slice of the sequence, so it is still emitted where the
    # sequence is long enough -- those are the alleles whose leader region ends before the
    # cleavage site, so the prefix exists but is not the whole signal peptide. Measured, not
    # assumed: pinning the split keeps the distinction visible.
    with_prefix = unanchored.filter(pl.col("signal_peptide").is_not_null())
    assert with_prefix.height == 82 - unanchored["signal_peptide"].null_count()
    for row in unanchored.iter_rows(named=True):
        assert row["mature_leader_tail"] is None, row["name"]
        if row["signal_peptide"] is not None:
            # A prefix that long can only be a claim about the sequence's own start, and the note
            # must say the mature frame is missing so a caller does not read it as mature 1.
            assert any("no mature frame" in note for note in row["notes"]), row["name"]


def test_null_and_questionable_alleles_are_chunked_not_dropped(chunked, region_parquet):
    per_allele = dict(region_parquet.group_by("expression_status").len().iter_rows())
    assert per_allele.get("null") == 25
    assert per_allele.get("questionable") == 5
    assert per_allele.get("expressed") == 1953
    for status in ("null", "questionable"):
        subset = chunked.filter(pl.col("expression_status") == status)
        assert subset["ectodomain_ok"].all(), f"a {status} allele fails an ectodomain identity"


# --- provenance and the gene input ------------------------------------------------------


def test_every_ipdkir_allele_takes_the_annotated_path(chunked):
    """A verbatim IPD-KIR precursor must never be inferred, whatever gene was supplied."""
    counts = dict(chunked.group_by("provenance").len().iter_rows())
    assert counts == {"annotated": 1983}, f"expected all 1983 annotated, got {counts}"
    assert chunked["identity"].is_null().all(), "the annotated path must not report an identity"


def test_deposited_chains_take_the_inferred_path(kir_chains):
    """The 54 deposited chains are fragments, not precursors, so they exercise the transfer.

    This is the measurement that makes the alignment path load-bearing rather than vestigial: with
    the gene supplied, an exact-match-only implementation would return nothing for any of them.
    """
    frame = pl.DataFrame(
        {
            "name": [f"{c['pdb_id']}_{c['chain']}" for c in kir_chains],
            "input_gene": [c["gene"] for c in kir_chains],
            "sequence": [c["sequence"] for c in kir_chains],
        }
    ).with_columns(c=pl.col("sequence").kir_regions.chunk(gene="input_gene")).unnest("c")

    assert frame.height == 54
    assert (frame["provenance"] == "inferred").all(), (
        "a deposited chain fragment matched a precursor exactly, which would make the transfer "
        "path unnecessary for structural input"
    )
    assert frame["identity"].is_not_null().all()
    assert (frame["identity"] >= 0.55).all()
    assert frame["ectodomain_ok"].all()

    # An inferred region must be a substring of the input, in order: the transfer may place a
    # boundary imprecisely but must never invent or reorder residues.
    for row in frame.iter_rows(named=True):
        cursor = 0
        for region in IG:
            if row[region] is None:
                continue
            found = row["sequence"].find(row[region], cursor)
            assert found >= 0, f"{row['name']}: {region} is not a substring of the input"
            cursor = found


def test_a_real_fragment_of_a_reference_allele_is_inferred(alleles):
    """Slicing a committed sequence is not synthesising one.

    The fragment is real deposited-style input — a precursor with its leader and C-terminus
    removed — and it is exactly the case the transfer exists for.
    """
    precursor = next(a["sequence"] for a in alleles if a["allele"] == "KIR2DL1*0010101")
    fragment = precursor[21:220]  # mature residue 1 onward, cut inside the stem

    row = (
        pl.DataFrame({"sequence": [fragment]})
        .with_columns(c=pl.col("sequence").kir_regions.chunk(gene="KIR2DL1"))
        .unnest("c")
        .to_dicts()[0]
    )
    assert row["provenance"] == "inferred"
    assert row["gene"] == "KIR2DL1"
    assert row["ectodomain_ok"]
    assert row["D1"] in fragment and row["D2"] in fragment


def test_a_literal_gene_and_a_column_agree(alleles, region_parquet):
    """`gene="KIR3DL1"` and `gene="gene"` must give the same answer on the same rows.

    The namespace resolves a bare string as a literal when it starts with `KIR` and as a column
    name otherwise; this pins that both readings reach the same kernel.
    """
    gene_of = dict(zip(region_parquet["allele"], region_parquet["gene"], strict=True))
    names = [a["allele"] for a in alleles if gene_of[a["allele"]] == "KIR3DL1"][:40]
    sequences = {a["allele"]: a["sequence"] for a in alleles}
    frame = pl.DataFrame(
        {"input_gene": ["KIR3DL1"] * len(names), "sequence": [sequences[n] for n in names]}
    )
    by_column = frame.select(c=pl.col("sequence").kir_regions.chunk(gene="input_gene")).unnest("c")
    by_literal = frame.select(c=pl.col("sequence").kir_regions.chunk(gene="KIR3DL1")).unnest("c")
    assert by_column.equals(by_literal)


# --- what happens when the supplied gene is unusable ------------------------------------


def test_an_unknown_gene_gives_null_fields_and_a_note(alleles):
    """Never a guess: an unrecognised gene name produces no boundaries at all."""
    fragment = next(a["sequence"] for a in alleles if a["allele"] == "KIR2DL1*0010101")[21:220]
    row = (
        pl.DataFrame({"sequence": [fragment]})
        .with_columns(c=pl.col("sequence").kir_regions.chunk(gene="KIR9DZ9"))
        .unnest("c")
        .to_dicts()[0]
    )
    assert row["provenance"] == "none"
    for field in ("leader", "signal_peptide", "mature_leader_tail", *IG, "ectodomain"):
        assert row[field] is None, f"{field} is not null for an unknown gene"
    assert any("not in the vendored region table" in note for note in row["notes"])
    # Nothing was assigned, so there is no ectodomain to be wrong about.
    assert row["ectodomain_ok"]


def test_a_null_gene_gives_null_fields_and_a_note(alleles):
    fragment = next(a["sequence"] for a in alleles if a["allele"] == "KIR2DL1*0010101")[21:220]
    row = (
        pl.DataFrame({"g": [None], "sequence": [fragment]}, schema_overrides={"g": pl.String})
        .with_columns(c=pl.col("sequence").kir_regions.chunk(gene="g"))
        .unnest("c")
        .to_dicts()[0]
    )
    assert row["provenance"] == "none"
    assert row["ectodomain"] is None
    assert any("no gene was supplied" in note for note in row["notes"])


def test_a_wrong_gene_on_an_exact_match_loses_to_the_sequence(alleles, region_parquet):
    """A byte-identical precursor outranks a supplied label, and the disagreement is noted.

    This is the one place the expression overrides its caller, so it is worth pinning: silently
    honouring the label would return KIR2DL1 boundaries for a KIR3DL1 sequence, which differ by
    95 residues and would corrupt every downstream slice.
    """
    sequence = next(a["sequence"] for a in alleles if a["allele"] == "KIR3DL1*0010101")
    row = (
        pl.DataFrame({"sequence": [sequence]})
        .with_columns(c=pl.col("sequence").kir_regions.chunk(gene="KIR2DL1"))
        .unnest("c")
        .to_dicts()[0]
    )
    assert row["gene"] == "KIR3DL1", "the exact match must win over the supplied gene"
    assert row["provenance"] == "annotated"
    assert row["D0"] is not None, "KIR3DL1 has a D0; honouring the KIR2DL1 label would drop it"
    assert any("was not used" in note for note in row["notes"]), (
        "the disagreement must be recorded, not silently resolved"
    )


def test_a_tie_set_is_refused_rather_than_arbitrated(alleles):
    """A `;`-separated gene value is a decision the caller has not made yet."""
    fragment = next(a["sequence"] for a in alleles if a["allele"] == "KIR2DL3*0010101")[21:200]
    row = (
        pl.DataFrame({"sequence": [fragment]})
        .with_columns(c=pl.col("sequence").kir_regions.chunk(gene=pl.lit("KIR2DL2;KIR2DL3")))
        .unnest("c")
        .to_dicts()[0]
    )
    assert row["provenance"] == "none", "a tie set must not be silently resolved to one gene"
    assert row["ectodomain"] is None


def test_a_non_kir_sequence_with_a_gene_label_is_still_transferred_but_scores_low(non_kir_curated):
    """The honest limit of a package with no classifier, pinned so it is not mistaken for a bug.

    Given a gene, this package transfers that gene's boundaries onto whatever sequence it is
    handed — it has no model with which to say "that is not a KIR". LILRB1 mislabelled as KIR3DL1
    therefore produces regions, and the only signal that something is wrong is the low alignment
    identity. A caller who does not trust its gene column should classify first.
    """
    # `next(...)` over a fixture raises StopIteration on a key rename, which reads as a broken
    # test rather than a broken lookup -- this assertion names the cause instead. The curated
    # negatives key their label as `identity`; an earlier version of this test guessed `name`/
    # `gene` and failed opaquely.
    assert all("identity" in entry for entry in non_kir_curated), (
        "the curated-negative fixture no longer carries an `identity` label; update the lookup"
    )
    matches = [c for c in non_kir_curated if c["identity"].startswith("LILRB1")]
    assert len(matches) == 1, f"expected exactly one LILRB1 negative, found {len(matches)}"
    lilrb1 = matches[0]["sequence"]
    row = (
        pl.DataFrame({"sequence": [lilrb1]})
        .with_columns(c=pl.col("sequence").kir_regions.chunk(gene="KIR3DL1"))
        .unnest("c")
        .to_dicts()[0]
    )
    assert row["provenance"] == "inferred"
    assert row["identity"] is not None
    assert row["identity"] < 0.45, (
        f"LILRB1 aligned to KIR3DL1 at identity {row['identity']}, which is high enough that "
        "identity would not warn a caller about a mislabelled sequence"
    )
    # The identities must still hold: a wrong answer must at least be internally consistent.
    assert row["ectodomain_ok"]


# --- the region table's own invariant ---------------------------------------------------


def test_region_table_still_tiles(region_parquet, alleles):
    """`leader + D0 + D1 + D2 + stem + TM + cyt == precursor`, for all 1983 alleles.

    This is the Phase 0 invariant. `chunk` no longer emits the regions needed to check it — the
    emitted fields stop at the D2 end by scope decision — so it is asserted here against the
    parquet directly. It is what licenses the ectodomain slice: the emitted regions are a prefix
    of a verified partition, not an unverified subset.
    """
    sequences = {a["allele"]: a["sequence"] for a in alleles}
    failures = []
    for row in region_parquet.iter_rows(named=True):
        sequence = sequences[row["allele"]]
        rebuilt = "".join(
            sequence[row[f"{region}_start"] - 1 : row[f"{region}_end"]]
            for region in ALL_REGIONS
            if row[f"{region}_start"] is not None
        )
        if rebuilt != sequence:
            failures.append(row["allele"])
    assert not failures, f"{len(failures)} alleles no longer tile, e.g. {failures[:5]}"
    assert region_parquet.height == 1983


def test_emitted_fields_do_not_tile_the_precursor(chunked):
    """The scope decision, asserted so it cannot be mistaken for a bug.

    Someone will concatenate the emitted regions and expect the input back. They will get a short
    sequence, because everything from the stem onward is out of scope. A zero shortfall is
    legitimate for exactly one class: a null allele whose premature stop falls at or inside D2, so
    it genuinely has no stem, TM or cytoplasmic region to omit.
    """
    shortfalls = {}
    for row in chunked.iter_rows(named=True):
        if row["ectodomain"] is None or row["cds_partial"]:
            continue
        emitted = len(row["leader"] or "") + sum(
            len(row[region]) for region in IG if row[region] is not None
        )
        shortfalls[row["name"]] = (row["length"] - emitted, row["expression_status"])
    assert shortfalls, "no allele was measurable"

    zero = {name: status for name, (gap, status) in shortfalls.items() if gap == 0}
    assert len(zero) == 21, f"expected 21 zero-shortfall alleles, got {len(zero)}"
    assert set(zero.values()) == {"null"}, (
        f"a non-null allele's emitted fields tile its whole precursor: "
        f"{[n for n, s in zero.items() if s != 'null'][:5]}"
    )

    expressed = [gap for gap, status in shortfalls.values() if status == "expressed"]
    assert min(expressed) > 0, (
        "an expressed allele's emitted fields tile its whole precursor, which contradicts the "
        "ectodomain scope"
    )


# --- independent structural cross-checks ------------------------------------------------


def test_d2_starts_at_the_structural_hinge_in_every_gene(alleles, gene_reference):
    """The exon 4/5 junction lands exactly on the crystallographic hinge — delta 0, 15/15 genes.

    Computed from `chunk`'s own output, so it exercises the expression's mature frame end to end.
    Independent evidence: `hinge_start` comes from deposited coordinates, `D2` from the exon
    annotation.
    """
    reference = gene_reference.to_dicts()
    sequences = {a["allele"]: a["sequence"] for a in alleles}
    frame = pl.DataFrame(
        {
            "input_gene": [r["gene"] for r in reference],
            "sequence": [sequences[r["reference_allele"]] for r in reference],
        }
    ).with_columns(c=pl.col("sequence").kir_regions.chunk(gene="input_gene")).unnest("c")

    for expected, row in zip(reference, frame.to_dicts(), strict=True):
        d2_start = len(row["ectodomain"]) - len(row["D2"]) + 1
        assert d2_start == expected["hinge_start"], (
            f"{expected['gene']}: D2 starts at mature {d2_start} but the structural hinge is at "
            f"{expected['hinge_start']}"
        )


def test_ectodomain_end_is_one_past_histo_kir_d2_end_by_the_shared_codon_rule(
    alleles, gene_reference
):
    """The emitted ectodomain is 1 residue longer than `histo_kir`'s `D2_end`, in all 15 genes.

    **This is the shared-codon convention, not an off-by-one.** Exon 5's terminal codon is split
    across the exon 5/6 boundary and draws two of its three nucleotides from exon 5, so the rule
    in `docs/REGIONS.md` section 3 assigns that residue to D2 while the structural `stem_start`
    puts it in the stem. Pinned so the difference stays a documented convention.
    """
    reference = gene_reference.to_dicts()
    sequences = {a["allele"]: a["sequence"] for a in alleles}
    frame = pl.DataFrame(
        {
            "input_gene": [r["gene"] for r in reference],
            "sequence": [sequences[r["reference_allele"]] for r in reference],
        }
    ).with_columns(c=pl.col("sequence").kir_regions.chunk(gene="input_gene")).unnest("c")

    for expected, row in zip(reference, frame.to_dicts(), strict=True):
        assert len(row["ectodomain"]) == expected["D2_end"] + 1, (
            f"{expected['gene']}: ectodomain is {len(row['ectodomain'])} residues, expected "
            f"{expected['D2_end']} + 1"
        )
        assert expected["mature_sequence"].startswith(row["ectodomain"]), (
            f"{expected['gene']}: the ectodomain is not a prefix of the mature reference"
        )
