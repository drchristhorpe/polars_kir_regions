"""Validate the loops expression against the 54 deposited KIR chains, read from coordinates.

Three checks:

1. Against the coordinate files. The expression's element slices are compared with the residues
   read out of the PDB at the file residue numbers the window implies. 12 of the 54 chains are
   deposited in PRECURSOR numbering (offset +21) -- the eight 9f2d chains, 9hml_B and the three
   9lr* chains, per validation/numbering_conventions.csv -- so the offset is applied per chain
   from that table. Ignoring it produces a spurious result.

   Two invariants of this project govern how a structure is read here, and both cost a bug when
   ignored: the framed files carry ROLE-based chain letters (the KIR chain is `K` in every file,
   whatever the depositor called it), so the chain is found by matching the scored sequence rather
   than by parsing the filename; and a file residue number is not a position, so it is used only
   to look up what the mature-frame window predicts, after the offset is applied.

2. Loop-in-domain. Each element must fall inside the region its name implies, using the Phase 0
   per-allele region boundaries: A'B/CC'/EF/CC'ext in D1, BC/FG/hinge in D2, D0a/D0b in D0. The
   hinge is additionally checked for containing the exon-junction (shared) codon.

3. Transfer failures. Any window that did not transfer whole is reported, not hidden.
"""
from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

import gemmi
import polars as pl
import polars_kir_regions  # noqa: F401

ROOT = Path(__file__).resolve().parent.parent
KIRS = ROOT.parent
FRAMED = KIRS / "validation/structures/framed"

ELEMENTS = ("A'B", "CC'", "EF", "hinge", "BC", "FG", "D0a", "D0b", "CC'ext")
IMPLIED = {"A'B": "D1", "CC'": "D1", "EF": "D1", "CC'ext": "D1",
           "BC": "D2", "FG": "D2", "hinge": "D2", "D0a": "D0", "D0b": "D0"}

structural = json.loads((ROOT / "tests/data/structural_chains.json").read_text())
chains = structural["kir_chains"]

conventions = {}
with open(KIRS / "validation/numbering_conventions.csv") as handle:
    for row in csv.DictReader(handle):
        conventions[f"{row['pdb_id']}_{row['kir_chain']}"] = row

regions = pl.read_parquet(KIRS / "histo_kir/data/kir_allele_regions.parquet")
region_row = {r["allele"]: r for r in regions.iter_rows(named=True)}
with open(KIRS / "histo_kir/data/kir_gene_reference.csv") as handle:
    gene_reference = {row["gene"]: row for row in csv.DictReader(handle)}

frame = pl.DataFrame({
    "file": [c["file"] for c in chains],
    "gene": [c["gene"] for c in chains],
    "sequence": [c["sequence"] for c in chains],
})
out = frame.with_columns(loops=pl.col("sequence").kir_regions.loops(gene="gene"))

report: dict = {}

# ---- check 1: element slices against the coordinates ------------------------------------------
agree = 0
disagree: list[dict] = []
offsets_applied: Counter = Counter()

for row in out.iter_rows(named=True):
    stem = row["file"].replace(".pdb", "")
    convention = conventions.get(stem)
    offset = int(convention["modal_offset"]) if convention else 0
    offsets_applied[offset] += 1

    structure = gemmi.read_structure(str(FRAMED / row["file"]))
    structure.setup_entities()
    polymer = None
    for model in structure:
        for chain in model:
            observed = "".join(
                gemmi.find_tabulated_residue(residue.name).one_letter_code.upper()
                for residue in chain
                if gemmi.find_tabulated_residue(residue.name) is not None
                and gemmi.find_tabulated_residue(residue.name).is_amino_acid()
            )
            if observed == row["sequence"]:
                polymer = chain
                break
        break
    if polymer is None:
        disagree.append({"file": row["file"], "reason": "no chain matches the scored sequence"})
        continue

    by_number: dict[int, str] = {}
    for residue in polymer:
        info = gemmi.find_tabulated_residue(residue.name)
        if info is None or not info.is_amino_acid():
            continue
        by_number[residue.seqid.num] = info.one_letter_code.upper()

    for element in ELEMENTS:
        got = row["loops"][element]
        if got is None or got["sequence"] is None:
            continue
        wanted = [by_number.get(m + offset)
                  for m in range(got["mature_start"], got["mature_end"] + 1)]
        if any(residue is None for residue in wanted):
            disagree.append({"file": row["file"], "element": element,
                             "reason": "window not fully present in the coordinates",
                             "expression": got["sequence"]})
            continue
        from_coordinates = "".join(wanted)
        if from_coordinates == got["sequence"]:
            agree += 1
        else:
            disagree.append({"file": row["file"], "element": element,
                             "expression": got["sequence"],
                             "coordinates": from_coordinates, "offset": offset})

print("CHECK 1 -- element slices vs the coordinate files")
print(f"  comparisons              {agree + len(disagree)}")
print(f"  identical                {agree}")
print(f"  disagreements            {len(disagree)}")
print(f"  numbering offsets seen   {dict(offsets_applied)}")
for item in disagree[:10]:
    print(f"    {item}")
report["coordinate_check"] = {"identical": agree, "disagreements": len(disagree),
                             "offsets": dict(offsets_applied), "examples": disagree[:20]}

# ---- check 2: loop-in-domain -------------------------------------------------------------------
windows = polars_kir_regions.loop_windows()
inside = 0
outside: list[dict] = []
hinge_rows: list[dict] = []

for element_row in windows.iter_rows(named=True):
    gene = element_row["gene"]
    allele = gene_reference[gene]["reference_allele"]
    region = region_row[allele]
    signal = region["signal_peptide_length"]
    domain = IMPLIED[element_row["element"]]
    start, end = region[f"{domain}_start"], region[f"{domain}_end"]
    if start is None:
        outside.append({"gene": gene, "element": element_row["element"],
                        "reason": f"{domain} absent from the region table"})
        continue
    span = (start - signal, end - signal)
    window = (element_row["mature_start"], element_row["mature_end"])
    if span[0] <= window[0] and window[1] <= span[1]:
        inside += 1
    else:
        outside.append({"gene": gene, "element": element_row["element"],
                        "window": list(window), "implied": domain, "span": list(span)})

    if element_row["element"] == "hinge":
        shared = json.loads(region["shared_codon_positions"] or "[]")
        positions = [signal + p for p in range(window[0], window[1] + 1)]
        hits = [s["position"] for s in shared if s["position"] in positions]
        hinge_rows.append({"gene": gene,
                          "inside_D2": span[0] <= window[0] and window[1] <= span[1],
                          "has_shared_codon": bool(hits),
                          "first_residue_is_shared": bool(hits) and hits[0] == positions[0]})

print("\nCHECK 2 -- each element inside the region its name implies")
print(f"  elements checked         {inside + len(outside)}")
print(f"  inside                   {inside}")
print(f"  outside                  {len(outside)}")
for item in outside[:10]:
    print(f"    {item}")
n_d2 = sum(1 for h in hinge_rows if h["inside_D2"])
n_shared = sum(1 for h in hinge_rows if h["has_shared_codon"])
n_first = sum(1 for h in hinge_rows if h["first_residue_is_shared"])
print(f"  hinge wholly inside D2   {n_d2}/{len(hinge_rows)}")
print(f"  hinge has a shared codon {n_shared}/{len(hinge_rows)} "
      f"(first residue IS that codon in {n_first})")
report["domain_check"] = {"inside": inside, "outside": len(outside), "examples": outside[:20],
                         "hinge_inside_D2": n_d2, "hinge_has_shared_codon": n_shared,
                         "hinge_first_residue_is_shared": n_first, "n_genes": len(hinge_rows)}

# ---- check 3: transfer failures ----------------------------------------------------------------
failures = []
for row in out.iter_rows(named=True):
    for element in ELEMENTS:
        got = row["loops"][element]
        if got is not None and got["status"] != "complete":
            failures.append({"file": row["file"], "element": element, "status": got["status"],
                            "n_observed": got["n_observed"], "n_expected": got["n_expected"]})
print("\nCHECK 3 -- windows that did not transfer whole (reported, not hidden)")
print(f"  count {len(failures)}")
for item in failures:
    print(f"    {item}")
report["transfer_failures"] = failures

(ROOT / "tools/loops_structural.json").write_text(json.dumps(report, indent=1))
print("\nwrote tools/loops_structural.json")
