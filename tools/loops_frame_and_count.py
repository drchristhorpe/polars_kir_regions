"""The two figures that pin the numbering frame and the ectodomain scope, both computed here.

FRAME. The loop windows are mature numbering counted from the signal-peptide CLEAVAGE site. The
region table is precursor numbering. Slicing the 15 gene reference alleles with each candidate
origin decides which is right, and the wrong one is wrong by exactly 2 in every gene -- a shift
that still returns the right NUMBER of plausible residues, so nothing downstream would flag it.
Both directions are computed, because a check that only confirms the right answer would still pass
if the two origins ever converged.

SCOPE. The element count, its breakdown by evidence class, and how many windows end at or before
their gene's mature D2 end -- the figure that justifies emitting the ectodomain only.
"""
from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parent.parent
KIRS = ROOT.parent

table = json.loads((ROOT / "python/polars_kir_regions/loops/loop_windows.json").read_text())
alleles = {a["allele"]: a for a in
           json.loads((ROOT / "tests/data/ipdkir_alleles.json").read_text())["alleles"]}
regions = pl.read_parquet(KIRS / "histo_kir/data/kir_allele_regions.parquet")
region_row = {r["allele"]: r for r in regions.iter_rows(named=True)}
with open(KIRS / "histo_kir/data/kir_gene_reference.csv") as handle:
    gene_reference = {row["gene"]: row for row in csv.DictReader(handle)}

# ---- the frame -------------------------------------------------------------------------------
cleavage_ok = leader_ok = total = 0
per_gene = {}
for gene, entry in sorted(table["genes"].items()):
    allele = entry["reference_allele"]
    precursor = alleles[allele]["sequence"]
    signal = entry["signal_peptide_length"]
    exon_leader = region_row[allele]["leader_end_exon_frame"]
    hits = [0, 0]
    published = [e for e in entry["elements"] if e["kind"] == "published_loop"]
    for element in published:
        start, end = element["mature_start"], element["mature_end"]
        from_cleavage = precursor[signal + start - 1 : signal + end]
        from_leader = precursor[exon_leader + start - 1 : exon_leader + end]
        total += 1
        hits[0] += from_cleavage == element["sequence"]
        hits[1] += from_leader == element["sequence"]
    cleavage_ok += hits[0]
    leader_ok += hits[1]
    per_gene[gene] = {"published": len(published), "from_cleavage": hits[0],
                      "from_exon_leader": hits[1], "signal_peptide_length": signal,
                      "leader_end_exon_frame": exon_leader, "delta": exon_leader - signal}

print("FRAME -- published loop windows reproduced from each candidate origin")
print(f"  windows                        {total}")
print(f"  from the CLEAVAGE site         {cleavage_ok}/{total}")
print(f"  from the exon-derived leader   {leader_ok}/{total}")
deltas = Counter(v["delta"] for v in per_gene.values())
print(f"  leader_end_exon_frame - signal_peptide_length, per gene: {dict(deltas)}")

# ---- the scope -------------------------------------------------------------------------------
by_kind: Counter = Counter()
checked = inside = 0
outside = []
for gene, entry in sorted(table["genes"].items()):
    allele = entry["reference_allele"]
    signal = entry["signal_peptide_length"]
    d2_end_mature = region_row[allele]["D2_end"] - signal
    for element in entry["elements"]:
        by_kind[element["kind"]] += 1
        checked += 1
        if element["mature_end"] <= d2_end_mature:
            inside += 1
        else:
            outside.append({"gene": gene, "element": element["name"],
                            "mature_end": element["mature_end"], "d2_end": d2_end_mature})

print("\nSCOPE -- element windows and whether each ends at or before the mature D2 end")
print(f"  element windows                {checked}")
print(f"  by evidence class              {dict(by_kind)}")
print(f"  ending at or before D2 end     {inside}/{checked}")
print(f"  after the D2 end               {len(outside)}")
for item in outside:
    print(f"    {item}")

by_arch: dict[str, set[int]] = {}
for gene, entry in table["genes"].items():
    by_arch.setdefault(entry["architecture"], set()).add(len(entry["elements"]))
print(f"  elements per gene by architecture: "
      f"{ {k: sorted(v) for k, v in sorted(by_arch.items())} }")

payload = {
    "frame": {"windows": total, "from_cleavage_site": cleavage_ok,
              "from_exon_leader": leader_ok, "per_gene": per_gene,
              "deltas": {str(k): v for k, v in deltas.items()}},
    "scope": {"elements": checked, "by_kind": dict(by_kind), "inside_ectodomain": inside,
              "outside": outside,
              "per_architecture": {k: sorted(v) for k, v in by_arch.items()}},
}
(ROOT / "tools/loops_frame_and_count.json").write_text(json.dumps(payload, indent=1))
print("\nwrote tools/loops_frame_and_count.json")
