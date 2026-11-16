"""Vendor the per-allele exon-derived region table into `python/polars_kir_hmm/regions/`.

Same pattern as `vendor_models.py`, and for the same reason: the `chunk` expression must not
re-derive boundaries at runtime. `histo_kir/data/kir_allele_regions.parquet` is the normative
source (`docs/REGIONS.md`); this script reshapes it into the lookup structures the Rust kernel
needs and writes one `regions.json`.

Two structures, because `chunk` has two paths:

  alleles   sha256-prefix of the precursor sequence -> the annotated spans for the allele(s)
            with that exact sequence. The exact-match path; no alignment, no inference.
  genes     per gene: the reference allele's precursor sequence and a per-position region code,
            for the alignment-transfer path on sequences that are not verbatim IPD-KIR entries.

**Duplicate precursor sequences are merged, not deduplicated by picking one.** 199 of the 892
distinct precursors are shared by 2 or more allele names (measured in this session). Every such
group is within a single gene and every group agrees on all 14 region boundaries (also measured,
and asserted below), so the merged entry carries one set of spans and the full list of names.
A caller reading `alleles` sees all of them; `allele` is null when there is more than one, the
same convention `classify` uses for a tied gene call.

What this script *checks* — each check is a hard failure, because a silent violation would make
the tiling invariant untrue downstream:

  * the seven regions tile every allele's own precursor sequence exactly (no overlap, no gap),
  * each region occupies one contiguous block (`_blocks` in {0, 1}),
  * `extracellular` == D0+D1+D2+stem and `intracellular` == cyt, span for span,
  * duplicate-sequence groups agree on gene, architecture and every boundary,
  * every gene has a reference allele present in the sequence set,
  * the BLOSUM62 submatrix hardcoded in `src/align.rs` matches Biopython's `blastp` matrix over
    the 21 characters a cleaned sequence can contain.

Run:  python codegen/vendor_regions.py --kirs-root ..
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parent.parent
DEST = ROOT / "python" / "polars_kir_hmm" / "regions"

#: Region order is the tiling order, N- to C-terminal. It is also the order of the numeric codes
#: the Rust side uses (index + 1; 0 means "no region").
REGIONS = ("leader", "D0", "D1", "D2", "stem", "TM", "cyt")

#: The characters `clean_sequence` keeps: the 20 amino acids and `X`.
ALPHABET = "ACDEFGHIKLMNPQRSTVWY"
CLEAN_CHARS = ALPHABET + "X"


def _sequence_hash(sequence: str) -> str:
    """The same 16-hex-character key `classify.rs::sequence_hash` computes."""
    return hashlib.sha256(sequence.encode()).hexdigest()[:16]


def _check_tiling(row: dict, sequence: str) -> None:
    """The invariant: the seven regions reproduce the precursor character for character."""
    allele = row["allele"]
    if len(sequence) != row["protein_length"]:
        raise SystemExit(
            f"{allele}: protein_length {row['protein_length']} but sequence is {len(sequence)}"
        )

    pieces: list[str] = []
    cursor = 1
    for region in REGIONS:
        start, end = row[f"{region}_start"], row[f"{region}_end"]
        blocks = row[f"{region}_blocks"]
        if start is None:
            if blocks not in (0, None):
                raise SystemExit(f"{allele}: {region} has no span but {blocks} blocks")
            continue
        if blocks != 1:
            raise SystemExit(
                f"{allele}: {region} spans {start}-{end} but reports {blocks} blocks; "
                "a region must be one contiguous run"
            )
        if start != cursor:
            raise SystemExit(
                f"{allele}: {region} starts at {start} but the previous region ended at "
                f"{cursor - 1} -- that is a {'gap' if start > cursor else 'overlap'}"
            )
        pieces.append(sequence[start - 1 : end])
        cursor = end + 1

    if cursor - 1 != len(sequence):
        raise SystemExit(f"{allele}: regions end at {cursor - 1} of {len(sequence)} residues")
    if "".join(pieces) != sequence:
        raise SystemExit(f"{allele}: concatenated regions do not reproduce the sequence")

    # The composites must be exactly the spans they are documented to be (REGIONS.md section 6),
    # not merely the same length.
    present = [r for r in ("D0", "D1", "D2", "stem") if row[f"{r}_start"] is not None]
    if present:
        expected = (row[f"{present[0]}_start"], row[f"{present[-1]}_end"])
        actual = (row["extracellular_start"], row["extracellular_end"])
        if expected != actual:
            raise SystemExit(f"{allele}: extracellular {actual} != D0..stem extent {expected}")
    if (row["intracellular_start"], row["intracellular_end"]) != (
        row["cyt_start"],
        row["cyt_end"],
    ):
        raise SystemExit(f"{allele}: intracellular span differs from the cyt span")


def _owner_codes(row: dict, length: int) -> list[int]:
    """Per-position region code for one allele: 1..7 in `REGIONS` order, 0 for none."""
    codes = [0] * length
    for index, region in enumerate(REGIONS, start=1):
        start, end = row[f"{region}_start"], row[f"{region}_end"]
        if start is None:
            continue
        for position in range(start, end + 1):
            codes[position - 1] = index
    return codes


def _check_blosum62(align_rs: Path) -> int:
    """Cross-check the matrix in `src/align.rs` against Biopython's `blastp` matrix.

    `src/align.rs` is shared with the `loops` expression and its table is generated by
    `codegen/vendor_loops.py` from the live reference aligner, so this is a second, independent
    check rather than the generator — the substitution matrix and the two gap penalties *are* the
    transfer method, and a silent drift in them would move every inferred boundary.

    The table is parsed rather than assumed: its alphabet and dimensions are read from the
    source, so it keeps checking if `vendor_loops.py` regenerates it in a different shape.
    """
    try:
        from Bio import Align
    except ImportError:
        print("  biopython absent -- skipping the BLOSUM62 cross-check", file=sys.stderr)
        return 0

    matrix = Align.PairwiseAligner(scoring="blastp").substitution_matrix
    text = align_rs.read_text()

    alpha = re.search(r'const ALPHA: &\[u8; \d+\] = b"([^"]+)"', text)
    if alpha is None:
        raise SystemExit(f"could not find the ALPHA constant in {align_rs}")
    letters = alpha.group(1)

    table = re.search(r"const BLOSUM62: \[\[i\d+; (\d+)\]; \d+\] = \[(.*?)\n\];", text, re.S)
    if table is None:
        raise SystemExit(f"could not find the BLOSUM62 table in {align_rs}")
    size = int(table.group(1))
    numbers = [int(n) for n in re.findall(r"-?\d+", table.group(2))]
    if size != len(letters) or len(numbers) != size * size:
        raise SystemExit(
            f"{align_rs}: BLOSUM62 is {size}x{size} with {len(numbers)} entries but ALPHA has "
            f"{len(letters)} characters"
        )

    # Only the characters a cleaned sequence can actually contain are checked. The rest of the
    # matrix alphabet (B/J/O/U/Z/*) is unreachable from `clean_sequence`, so a discrepancy there
    # could not affect a transfer and asserting on it would be a check with no consequence.
    checked = 0
    for a in CLEAN_CHARS:
        for b in CLEAN_CHARS:
            actual = numbers[letters.index(a) * size + letters.index(b)]
            expected = int(matrix[a, b])
            if actual != expected:
                raise SystemExit(
                    f"{align_rs}: BLOSUM62[{a}][{b}] is {actual}, Biopython's blastp says "
                    f"{expected}"
                )
            checked += 1
    return checked


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--kirs-root",
        type=Path,
        default=ROOT.parent,
        help="the kirs project root holding histo_kir/ (default: the parent of this package)",
    )
    args = ap.parse_args()

    regions_parquet = args.kirs_root / "histo_kir" / "data" / "kir_allele_regions.parquet"
    gene_csv = args.kirs_root / "histo_kir" / "data" / "kir_gene_reference.csv"
    alleles_json = ROOT / "tests" / "data" / "ipdkir_alleles.json"
    for path in (regions_parquet, gene_csv, alleles_json):
        if not path.exists():
            raise SystemExit(f"missing input: {path}")

    table = pl.read_parquet(regions_parquet)
    gene_reference = pl.read_csv(gene_csv)
    sequences = {
        entry["allele"]: entry["sequence"]
        for entry in json.loads(alleles_json.read_text())["alleles"]
    }

    rows = table.to_dicts()
    missing = {row["allele"] for row in rows} - set(sequences)
    if missing:
        raise SystemExit(f"{len(missing)} alleles in the region table have no sequence: {sorted(missing)[:5]}")

    # --- per-gene block -----------------------------------------------------------------
    signal_lengths = dict(
        zip(gene_reference["gene"], gene_reference["signal_peptide_length"], strict=True)
    )
    reference_alleles = dict(
        zip(gene_reference["gene"], gene_reference["reference_allele"], strict=True)
    )
    by_allele = {row["allele"]: row for row in rows}

    gene_block: dict[str, dict] = {}
    for gene, reference_allele in reference_alleles.items():
        if reference_allele not in by_allele:
            raise SystemExit(f"{gene}: reference allele {reference_allele} is not in the region table")
        row = by_allele[reference_allele]
        sequence = sequences[reference_allele]
        if row["cds_partial"]:
            raise SystemExit(f"{gene}: reference allele {reference_allele} has a partial CDS")
        gene_block[gene] = {
            "reference_allele": reference_allele,
            "architecture": row["gene_architecture"],
            "signal_peptide_length": int(signal_lengths[gene]),
            "sequence": sequence,
            "owner": _owner_codes(row, len(sequence)),
        }

    # --- per-sequence block -------------------------------------------------------------
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        sequence = sequences[row["allele"]]
        _check_tiling(row, sequence)
        grouped.setdefault(_sequence_hash(sequence), []).append(row)

    spans_of = lambda row: [  # noqa: E731 - a local shorthand, used twice
        None if row[f"{r}_start"] is None else [int(row[f"{r}_start"]), int(row[f"{r}_end"])]
        for r in REGIONS
    ]

    # Fields that MUST agree within a sequence group, because the boundaries depend on them and
    # a merged entry that got one wrong would mis-slice every allele in the group.
    STRUCTURAL = ("gene", "gene_architecture", "architecture", "protein_length")

    # Fields that MAY differ within a sequence group. Measured, not assumed: 3 KIR3DL3 groups
    # mix a 5'-partial CDS with complete ones (a truncated CDS can still translate to the same
    # 410-residue protein), and 1 KIR3DL1 group mixes `KIR3DL1*0150222Q` with 33 sequence-
    # identical expressed siblings. These are properties of the *CDS annotation*, not of the
    # protein, so a protein-keyed table cannot carry one value for them. Where a group
    # disagrees the merged entry reports the **cautious** value -- partial if any member is
    # partial, and the least-confident expression status present -- and names the disagreement
    # so a caller is never told a sequence is complete when one of its alleles is not.
    ANNOTATION = (
        "expression_status",
        "cds_partial",
        "cds_partial_5prime",
        "cds_partial_3prime",
    )
    #: Least confident first: the value a mixed group reports.
    STATUS_ORDER = ("null", "questionable", "expressed")

    allele_block: dict[str, dict] = {}
    n_merged = 0
    n_mixed = 0
    mixed_examples: list[str] = []
    for key, group in grouped.items():
        head = group[0]
        mixed_fields: list[str] = []
        if len(group) > 1:
            n_merged += 1
            # Merging is only sound because the structural fields agree. Assert, never trust.
            for other in group[1:]:
                for field in STRUCTURAL:
                    if other[field] != head[field]:
                        raise SystemExit(
                            f"identical sequences {head['allele']}/{other['allele']} disagree "
                            f"on {field}: {head[field]!r} vs {other[field]!r}"
                        )
                if spans_of(other) != spans_of(head):
                    raise SystemExit(
                        f"identical sequences {head['allele']}/{other['allele']} disagree on "
                        "region boundaries -- the merged entry would be wrong for one of them"
                    )
            mixed_fields = [
                field
                for field in ANNOTATION
                if len({row[field] for row in group}) > 1
            ]
            if mixed_fields:
                n_mixed += 1
                if len(mixed_examples) < 4:
                    mixed_examples.append(
                        f"{head['gene']} {sorted(r['allele'] for r in group)[0]}+"
                        f"{len(group) - 1} ({','.join(mixed_fields)})"
                    )

        statuses = {row["expression_status"] for row in group}
        expression_status = next(s for s in STATUS_ORDER if s in statuses)

        allele_block[key] = {
            "alleles": sorted(row["allele"] for row in group),
            "gene": head["gene"],
            "architecture": head["architecture"],
            "gene_architecture": head["gene_architecture"],
            "expression_status": expression_status,
            # `any`, not the head's value: if one allele with this protein sequence has a partial
            # CDS, a caller must not be told the sequence is complete.
            "cds_partial": any(bool(row["cds_partial"]) for row in group),
            "cds_partial_5prime": any(bool(row["cds_partial_5prime"]) for row in group),
            "cds_partial_3prime": any(bool(row["cds_partial_3prime"]) for row in group),
            "protein_length": int(head["protein_length"]),
            "spans": spans_of(head),
            # Present only when it applies, so its absence is not a claim of agreement.
            **({"annotation_mixed": mixed_fields} if mixed_fields else {}),
        }

    n_checked = _check_blosum62(ROOT / "src" / "align.rs")

    payload = {
        "regions": list(REGIONS),
        "genes": gene_block,
        "alleles": allele_block,
        "vendored_from": {
            "region_table": str(regions_parquet),
            "gene_reference": str(gene_csv),
            "sequences": str(alleles_json),
            "n_alleles": len(rows),
            "n_distinct_sequences": len(allele_block),
            "note": (
                "reshaped from kir_allele_regions.parquet by codegen/vendor_regions.py; "
                "spans are 1-based inclusive precursor positions, see docs/REGIONS.md"
            ),
            "checksums": {
                path.name: hashlib.sha256(path.read_bytes()).hexdigest()[:16]
                for path in (regions_parquet, gene_csv, alleles_json)
            },
        },
    }

    DEST.mkdir(parents=True, exist_ok=True)
    out = DEST / "regions.json"
    out.write_text(json.dumps(payload, separators=(",", ":"), sort_keys=True) + "\n")

    print(f"vendored {len(rows)} alleles ({out.stat().st_size / 1e6:.3f} MB) -> {out}")
    print(f"  {len(allele_block)} distinct precursor sequences, {n_merged} shared by >1 allele name")
    if n_mixed:
        print(f"  {n_mixed} shared-sequence group(s) disagree on CDS annotation; the cautious "
              f"value is reported: {'; '.join(mixed_examples)}")
    print(f"  {len(gene_block)} genes; tiling verified for {len(rows)}/{len(rows)} alleles")
    if n_checked:
        print(f"  BLOSUM62 cross-check against Biopython blastp: {n_checked}/441 entries agree")
    return 0


if __name__ == "__main__":
    sys.exit(main())
