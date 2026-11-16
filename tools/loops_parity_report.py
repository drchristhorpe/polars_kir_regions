"""Measure `polars_kir_regions`' loops expression against the histo_kir-derived golden fixture.

Writes tools/loops_parity.json so the numbers in the README and the report come from a run.

The gene is supplied from the corpus in every case -- this package takes the gene as given and has
no classifier, so there is no self-typed comparison to make.
"""
from __future__ import annotations

import gzip
import json
import time
from collections import Counter
from pathlib import Path

import polars as pl
import polars_kir_regions  # noqa: F401  (registers the .kir_regions namespace)

ROOT = Path(__file__).resolve().parent.parent
golden = json.loads(gzip.open(ROOT / "tests/data/loops_golden.json.gz", "rt").read())
alleles = json.loads((ROOT / "tests/data/ipdkir_alleles.json").read_text())["alleles"]
structural = json.loads((ROOT / "tests/data/structural_chains.json").read_text())
ELEMENTS = ("A'B", "CC'", "EF", "hinge", "BC", "FG", "D0a", "D0b", "CC'ext")
FIELDS = ("sequence", "status", "start", "end", "n_observed", "n_expected", "matches_reference")

report: dict = {}


def compare(frame: pl.DataFrame, expected: dict, key: str, label: str) -> dict:
    exprs = [pl.col(key)]
    for element in ELEMENTS:
        for field in FIELDS:
            exprs.append(
                pl.col("loops").struct.field(element).struct.field(field)
                .alias(f"{element}|{field}")
            )
    flat = frame.select(exprs)

    agree = 0
    disagree: list[dict] = []
    absent_expected = 0
    status_counts: Counter = Counter()
    for row in flat.iter_rows(named=True):
        record = expected[row[key]]
        for element in ELEMENTS:
            want = record["elements"].get(element)
            got_status = row[f"{element}|status"]
            if want is None:
                # Not part of this architecture: must be a null struct.
                if got_status is None:
                    agree += 1
                    absent_expected += 1
                else:
                    disagree.append({"id": row[key], "element": element,
                                     "reason": "emitted for an architecture that lacks it",
                                     "got_status": got_status})
                continue
            status_counts[want["status"]] += 1
            mismatch = {}
            for field in FIELDS:
                if row[f"{element}|{field}"] != want[field]:
                    mismatch[field] = [row[f"{element}|{field}"], want[field]]
            if mismatch:
                disagree.append({"id": row[key], "element": element, "fields": mismatch})
            else:
                agree += 1

    total = agree + len(disagree)
    print(f"\n{label}")
    print(f"  element comparisons      {total}")
    print(f"  identical                {agree}")
    print(f"  disagreements            {len(disagree)}")
    print(f"  null (architecture)      {absent_expected}")
    print(f"  golden status counts     {dict(status_counts)}")
    for item in disagree[:8]:
        print(f"    {item}")
    return {"total": total, "identical": agree, "disagreements": len(disagree),
            "architecturally_absent": absent_expected,
            "golden_status_counts": dict(status_counts),
            "examples": disagree[:20]}


# --- the 1983 IPD-KIR alleles ------------------------------------------------------------------
frame = pl.DataFrame({
    "allele": [a["allele"] for a in alleles],
    "gene": [a["gene"] for a in alleles],
    "sequence": [a["sequence"] for a in alleles],
})
start = time.time()
out = frame.with_columns(loops=pl.col("sequence").kir_regions.loops(gene="gene"))
elapsed = time.time() - start
print(f"1983 alleles in {elapsed:.2f}s")
report["alleles"] = compare(out, golden["alleles"], "allele", "IPD-KIR alleles")
report["alleles"]["seconds"] = round(elapsed, 3)

# --- the 54 deposited KIR chains ---------------------------------------------------------------
chains = pl.DataFrame({
    "file": [c["file"] for c in structural["kir_chains"]],
    "gene": [c["gene"] for c in structural["kir_chains"]],
    "sequence": [c["sequence"] for c in structural["kir_chains"]],
})
start = time.time()
chain_out = chains.with_columns(loops=pl.col("sequence").kir_regions.loops(gene="gene"))
report["chains"] = compare(chain_out, golden["chains"], "file", "deposited KIR chains")
report["chains"]["seconds"] = round(time.time() - start, 3)

# --- a literal gene must equal the same gene supplied as a column -------------------------------
one_gene = [c for c in structural["kir_chains"] if c["gene"] == "KIR3DL1"]
literal = pl.DataFrame({"sequence": [c["sequence"] for c in one_gene]}).with_columns(
    loops=pl.col("sequence").kir_regions.loops(gene="KIR3DL1")
)
column = pl.DataFrame({
    "gene": ["KIR3DL1"] * len(one_gene),
    "sequence": [c["sequence"] for c in one_gene],
}).with_columns(loops=pl.col("sequence").kir_regions.loops(gene="gene"))
same = differ = 0
for a, b in zip(literal.iter_rows(named=True), column.iter_rows(named=True)):
    for element in ELEMENTS:
        x, y = a["loops"][element], b["loops"][element]
        xs = x["sequence"] if x else None
        ys = y["sequence"] if y else None
        same += xs == ys
        differ += xs != ys
print(f"\nliteral gene vs gene column, {len(one_gene)} KIR3DL1 chains")
print(f"  identical element sequences {same}; differing {differ}")
report["literal_vs_column"] = {"chains": len(one_gene), "identical": same, "differing": differ}

# --- provenance: annotated for the 15 reference alleles, inferred for transfers ------------------
prov = out.select(
    pl.col("allele"),
    pl.col("loops").struct.field("provenance").alias("provenance"),
)
counts = dict(Counter(prov["provenance"].to_list()))
chain_prov = dict(Counter(
    chain_out.select(pl.col("loops").struct.field("provenance"))["provenance"].to_list()
))
print(f"\nprovenance over 1983 alleles: {counts}")
print(f"provenance over 54 chains:   {chain_prov}")
report["provenance"] = {"alleles": counts, "chains": chain_prov}

(ROOT / "tools/loops_parity.json").write_text(json.dumps(report, indent=1))
print("\nwrote tools/loops_parity.json")
