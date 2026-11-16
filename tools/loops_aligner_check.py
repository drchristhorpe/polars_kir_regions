"""Re-measure the two aligner claims in LOOPS_REPORT.md against THIS package's build.

Both were first measured against the pre-split build; neither is carried over.

1. **Traceback parity.** The Rust aligner's match-state-first traceback must reproduce Biopython's
   `alignment[0]` — the alignment `histo_kir` actually uses. Measured by comparing the shipped
   expression's element sequences against Biopython-derived windows for every allele and chain.
   Ties matter: several alleles admit more than one optimal alignment, so the count of alleles with
   multiple optima, and of those whose branches give DIFFERENT windows, is reported too.

2. **The end-gap divergence.** `histo_kir`'s aligner sets free end-gap scores and then assigns
   `open_gap_score`, which in Biopython overwrites them, so end gaps are penalized. This measures
   how many element windows free end gaps would change — the figure that justifies reproducing the
   reference's real behaviour rather than its documented intent.
"""
from __future__ import annotations

import json
import sys
import types
from collections import Counter
from pathlib import Path

import polars as pl
import polars_kir_regions  # noqa: F401
from Bio import Align

ROOT = Path(__file__).resolve().parent.parent
KIRS = ROOT.parent

# histo_kir's package __init__ pulls in the HMM stack, which this package does not depend on.
# Bind a package shell so the three Biopython-only modules import from the committed source. This
# is a MEASUREMENT script under tools/, not shipped code -- the wheel imports none of it.
shell = types.ModuleType("histo_kir")
shell.__path__ = [str(KIRS / "histo_kir/src/histo_kir")]
sys.modules.setdefault("histo_kir", shell)

alleles = json.loads((ROOT / "tests/data/ipdkir_alleles.json").read_text())["alleles"]
chains = json.loads((ROOT / "tests/data/structural_chains.json").read_text())["kir_chains"]
table = json.loads((ROOT / "python/polars_kir_regions/loops/loop_windows.json").read_text())
ELEMENTS = ("A'B", "CC'", "EF", "hinge", "BC", "FG", "D0a", "D0b", "CC'ext")


def make_aligner(free_ends: bool) -> Align.PairwiseAligner:
    aligner = Align.PairwiseAligner(scoring="blastp")
    aligner.mode = "global"
    aligner.open_gap_score = -12.0
    aligner.extend_gap_score = -1.0
    if free_ends:
        # Assigned AFTER the general scores, so they survive -- which is precisely what the
        # reference does not do.
        aligner.end_insertion_score = 0.0
        aligner.end_deletion_score = 0.0
    return aligner


PENALIZED = make_aligner(False)   # what histo_kir's code actually does
FREE_ENDS = make_aligner(True)    # what its docstring says


def windows_via_biopython(aligner, sequence: str, gene: str) -> dict[str, str | None]:
    """Transfer this gene's windows onto `sequence` using Biopython, as histo_kir does."""
    entry = table["genes"][gene]
    reference = entry["precursor"]
    signal = entry["signal_peptide_length"]
    alignment = aligner.align(reference, sequence)[0]
    reference_to_query: dict[int, int] = {}
    for (r_start, r_end), (q_start, _) in zip(alignment.aligned[0], alignment.aligned[1]):
        for offset in range(r_end - r_start):
            reference_to_query.setdefault(int(r_start + offset + 1), int(q_start + offset))
    out: dict[str, str | None] = {}
    for element in entry["elements"]:
        indices = [reference_to_query.get(signal + p)
                   for p in range(element["mature_start"], element["mature_end"] + 1)]
        out[element["name"]] = (None if any(i is None for i in indices)
                               else "".join(sequence[i] for i in indices))
    return out


def shipped_windows(sequences: list[str], genes: list[str]) -> list[dict[str, str | None]]:
    frame = pl.DataFrame({"gene": genes, "sequence": sequences}).with_columns(
        loops=pl.col("sequence").kir_regions.loops(gene="gene")
    )
    rows = []
    for row in frame.iter_rows(named=True):
        record = row["loops"]
        rows.append({e: (record[e]["sequence"] if record[e] else None) for e in ELEMENTS})
    return rows


# ---- 1. traceback parity -----------------------------------------------------------------------
report: dict = {}
agree = 0
disagree: list[dict] = []

for label, items, key in (("alleles", alleles, "allele"), ("chains", chains, "file")):
    got = shipped_windows([i["sequence"] for i in items], [i["gene"] for i in items])
    for item, mine in zip(items, got):
        theirs = windows_via_biopython(PENALIZED, item["sequence"], item["gene"])
        for element in ELEMENTS:
            if element not in theirs:
                continue  # architecturally absent
            if mine[element] == theirs[element]:
                agree += 1
            else:
                disagree.append({"id": item[key], "element": element,
                                 "rust": mine[element], "biopython": theirs[element]})

print("TRACEBACK PARITY -- shipped Rust aligner vs Biopython alignment[0]")
print(f"  window comparisons       {agree + len(disagree)}")
print(f"  identical                {agree}")
print(f"  disagreements            {len(disagree)}")
for item in disagree[:8]:
    print(f"    {item}")

multiplicity: Counter = Counter()
tie_sensitive = []
for item in alleles:
    entry = table["genes"][item["gene"]]
    optima = PENALIZED.align(entry["precursor"], item["sequence"])
    count = len(optima)
    multiplicity["one" if count == 1 else "many"] += 1
    if count < 2:
        continue
    seen = set()
    for index in range(min(count, 50)):
        alignment = optima[index]
        reference_to_query: dict[int, int] = {}
        for (r_start, r_end), (q_start, _) in zip(alignment.aligned[0], alignment.aligned[1]):
            for offset in range(r_end - r_start):
                reference_to_query.setdefault(int(r_start + offset + 1), int(q_start + offset))
        signal = entry["signal_peptide_length"]
        shape = []
        for element in entry["elements"]:
            indices = [reference_to_query.get(signal + p)
                       for p in range(element["mature_start"], element["mature_end"] + 1)]
            shape.append(None if any(i is None for i in indices)
                        else "".join(item["sequence"][i] for i in indices))
        seen.add(tuple(shape))
    if len(seen) > 1:
        tie_sensitive.append(item["allele"])

print(f"  alleles with >1 optimal alignment          {multiplicity['many']}/{len(alleles)}")
print(f"  ... whose branches give DIFFERENT windows  {len(tie_sensitive)} {tie_sensitive}")
report["traceback"] = {"comparisons": agree + len(disagree), "identical": agree,
                      "disagreements": len(disagree), "examples": disagree[:20],
                      "alleles_with_multiple_optima": multiplicity["many"],
                      "tie_sensitive_alleles": tie_sensitive}

# ---- 2. the end-gap divergence -----------------------------------------------------------------
changed_alleles = 0
total_alleles = 0
changed_chains = 0
total_chains = 0
examples = []
for label, items, key in (("alleles", alleles, "allele"), ("chains", chains, "file")):
    for item in items:
        a = windows_via_biopython(PENALIZED, item["sequence"], item["gene"])
        b = windows_via_biopython(FREE_ENDS, item["sequence"], item["gene"])
        for element in a:
            if label == "alleles":
                total_alleles += 1
            else:
                total_chains += 1
            if a[element] != b[element]:
                if label == "alleles":
                    changed_alleles += 1
                else:
                    changed_chains += 1
                if len(examples) < 20:
                    examples.append({"id": item[key], "element": element,
                                    "penalized": a[element], "free_ends": b[element]})

print("\nEND-GAP DIVERGENCE -- what free end gaps would change")
print(f"  allele windows changed   {changed_alleles}/{total_alleles}")
print(f"  chain windows changed    {changed_chains}/{total_chains}")
for item in examples[:6]:
    print(f"    {item}")
report["end_gaps"] = {"allele_windows_changed": changed_alleles, "allele_windows": total_alleles,
                     "chain_windows_changed": changed_chains, "chain_windows": total_chains,
                     "examples": examples}

# The defect itself, read off the live object rather than asserted from the docstring.
probe = Align.PairwiseAligner(scoring="blastp")
probe.mode = "global"
probe.end_insertion_score = 0.0
probe.end_deletion_score = 0.0
before = probe.open_left_insertion_score
probe.open_gap_score = -12.0
after = probe.open_left_insertion_score
print(f"\n  open_left_insertion_score after setting end scores to 0: {before}")
print(f"  ... after then assigning open_gap_score = -12.0:          {after}")
report["end_gaps"]["open_left_insertion_before"] = before
report["end_gaps"]["open_left_insertion_after"] = after

(ROOT / "tools/loops_aligner_check.json").write_text(json.dumps(report, indent=1))
print("\nwrote tools/loops_aligner_check.json")
