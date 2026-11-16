"""Vendor the per-gene loop windows into `python/polars_kir_hmm/models/loop_windows.json`.

The windows themselves come from `histo_kir/data/kir_loop_windows.csv` (Boyington & Sun, Mol
Immunol 38:1007-1021, 2001, Fig. 5, transferred to each gene by alignment in Phase 0). The two
Phase 0 amendments -- the KIR3DL1 D0 contact patch (`D0a`/`D0b`) and the single-residue CC'
extension (`CC'ext`) -- are not in that CSV: they live as constants in `histo_kir.reference` and are
transferred onto the other three-domain genes by `histo_kir.domains`. This script reads all three
sources through `histo_kir` itself rather than restating any of them, so a retabulation upstream
cannot silently diverge from what the plugin ships.

**Nothing here re-derives a Phase 0 value.** In particular the signal-peptide lengths are read, not
inferred: KIR2DL4's is 23, not the 21 UniProt reports by similarity, and re-inferring it by aligning
full-length KIR sequences matches D1 against D1 for the three-domain genes and returns nonsense.

**What the file records, and why it is the precursor and not the mature sequence.** Each gene entry
carries its reference allele's full **precursor** plus `signal_peptide_length`. The loop windows are
mature numbering counted from the signal-peptide *cleavage site*, so a precursor position is
`signal_peptide_length + mature_position`. Storing the precursor lets the Rust align a query to it
directly, which is what makes a 5'-partial allele and a mature-numbered structural fragment both
work without a branch -- see `src/loops.rs`.

The script checks, before writing, that every recorded window sequence is the slice its own
coordinates name in its own gene's reference precursor. That is the frame self-check: a table
vendored against the exon-derived leader length instead of the cleavage site would fail it on all 81
published windows, because the two origins differ by exactly 2 in all 15 genes.

Run:  python codegen/vendor_loops.py --kirs-root ..
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEST = ROOT / "python" / "polars_kir_hmm" / "models" / "loop_windows.json"

#: Canonical element order, N- to C-terminal in the KIR2D frame. Must match
#: `loops::ELEMENT_ORDER` in the Rust; `src/loops_expr.rs` names the struct fields from it.
ELEMENT_ORDER = ("A'B", "CC'", "EF", "hinge", "BC", "FG", "D0a", "D0b", "CC'ext")


def _import_histo_kir(kirs_root: Path):
    """Import histo_kir's reference/numbering/domains from a source checkout.

    Bound as a package shell rather than imported normally: `histo_kir/__init__.py` pulls in
    `histo_kir_hmm`, `histo_hmm` and `histo_b2m_hmm` for the chain-typing side, none of which this
    script needs. The three modules it does need depend only on Biopython.
    """
    source = kirs_root / "histo_kir" / "src" / "histo_kir"
    if not source.is_dir():
        raise SystemExit(f"no histo_kir source at {source}")
    shell = types.ModuleType("histo_kir")
    shell.__path__ = [str(source)]
    sys.modules["histo_kir"] = shell
    from histo_kir import domains, reference  # noqa: E402

    return reference, domains


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--kirs-root",
        type=Path,
        default=ROOT.parent,
        help="the kirs tree holding histo_kir/ (default: the parent of this package)",
    )
    parser.add_argument("--output", type=Path, default=DEST)
    args = parser.parse_args()

    reference, domains = _import_histo_kir(args.kirs_root)
    gene_reference = reference.load_gene_reference()
    loop_windows = reference.load_loop_windows()

    # The reference allele's precursor. Read from the same allele corpus the tests use, so the
    # vendored precursor and the fixture agree by construction.
    corpus = json.loads(
        (ROOT / "tests" / "data" / "ipdkir_alleles.json").read_text()
    )["alleles"]
    precursors = {entry["allele"]: entry["sequence"] for entry in corpus}

    table: dict = {
        "_provenance": {
            "generated_by": "codegen/vendor_loops.py",
            "published_windows": "histo_kir/data/kir_loop_windows.csv -- Boyington & Sun, Mol "
            "Immunol 38:1007-1021 (2001), Fig. 5, transferred per gene by alignment in Phase 0",
            "d0_patch": "histo_kir.reference.D0_PATCHES_KIR3DL1 (KIR3DL1 mature 9-13 and 27-34), "
            "transferred within the D0D1D2 architecture by histo_kir.domains.d0_patches; "
            "undetermined for D0D2 and absent for D1D2",
            "ccprime_extension": "histo_kir.reference.CCPRIME_EXTENSION_REF_POS = 41 in the "
            "KIR2DL2 frame, transferred by histo_kir.domains.ccprime_extension; D0D1D2 only",
            "gene_reference": "histo_kir/data/kir_gene_reference.csv",
            "precursors": "tests/data/ipdkir_alleles.json (IPD-KIR 2.15.0 translations)",
            "numbering": "element windows are MATURE numbering counted from the signal-peptide "
            "CLEAVAGE site; a precursor position is signal_peptide_length + mature_position. The "
            "exon-derived leader is a different origin, 2 residues longer in all 15 genes.",
            "aligner": {
                "matrix": "BLOSUM62 (Biopython scoring='blastp')",
                "gap_open": -12.0,
                "gap_extend": -1.0,
                "end_gaps": "penalized as internal -- the reference's open_gap_score assignment "
                "overwrites the end-gap scores it sets earlier",
                "traceback": "match state preferred, then ref-consuming, then query-consuming gap",
            },
        },
        "element_order": list(ELEMENT_ORDER),
        "genes": {},
    }

    for gene in sorted(gene_reference):
        entry = gene_reference[gene]
        precursor = precursors.get(entry.reference_allele)
        if precursor is None:
            raise SystemExit(
                f"{gene}: reference allele {entry.reference_allele} is not in the allele corpus"
            )
        if precursor[entry.signal_peptide_length :] != entry.mature_sequence:
            raise SystemExit(
                f"{gene}: precursor[{entry.signal_peptide_length}:] does not equal the recorded "
                "mature sequence -- the signal-peptide length and the precursor disagree"
            )

        elements: list[dict] = []
        windows = loop_windows[gene]
        for name in reference.LOOP_ORDER:
            window = windows.get(name)
            if window is None:
                continue
            elements.append(
                {
                    "name": name,
                    "kind": "published_loop",
                    "domain": window.domain,
                    "helix": window.hla_helix,
                    "mature_start": window.mature_start,
                    "mature_end": window.mature_end,
                    "sequence": window.sequence,
                }
            )

        patches = domains.d0_patches(gene)
        for name, (start, end) in (patches or {}).items():
            elements.append(
                {
                    "name": name,
                    "kind": "d0_patch",
                    "domain": "D0",
                    "helix": "",
                    "mature_start": start,
                    "mature_end": end,
                    "sequence": entry.mature_sequence[start - 1 : end],
                }
            )

        extension = domains.ccprime_extension(gene)
        if extension is not None:
            elements.append(
                {
                    "name": reference.CCPRIME_EXTENSION,
                    "kind": "ccprime_extension",
                    "domain": "D1",
                    "helix": "alpha1",
                    "mature_start": extension,
                    "mature_end": extension,
                    "sequence": entry.mature_sequence[extension - 1 : extension],
                }
            )

        order = {name: index for index, name in enumerate(ELEMENT_ORDER)}
        unknown = [e["name"] for e in elements if e["name"] not in order]
        if unknown:
            raise SystemExit(f"{gene}: element(s) {unknown} are not in ELEMENT_ORDER")
        elements.sort(key=lambda e: order[e["name"]])

        table["genes"][gene] = {
            "architecture": entry.architecture,
            "reference_allele": entry.reference_allele,
            "signal_peptide_length": entry.signal_peptide_length,
            "mature_length": entry.mature_length,
            "precursor": precursor,
            "elements": elements,
        }

    # The frame self-check. This is the assertion that catches a 2-residue shift.
    checked = 0
    for gene, block in table["genes"].items():
        signal = block["signal_peptide_length"]
        precursor = block["precursor"]
        for element in block["elements"]:
            start = signal + element["mature_start"] - 1
            end = signal + element["mature_end"]
            got = precursor[start:end]
            if got != element["sequence"]:
                raise SystemExit(
                    f"{gene} {element['name']}: precursor[{start}:{end}] is {got!r} but the table "
                    f"records {element['sequence']!r}. The windows are mature numbering from the "
                    "signal-peptide CLEAVAGE site; a 2-residue shift means the exon-derived "
                    "leader length was used as the origin."
                )
            checked += 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(table, indent=1, sort_keys=False))
    digest = hashlib.sha256(args.output.read_bytes()).hexdigest()

    by_kind: dict[str, int] = {}
    for block in table["genes"].values():
        for element in block["elements"]:
            by_kind[element["kind"]] = by_kind.get(element["kind"], 0) + 1

    print(f"wrote {args.output.relative_to(ROOT)}")
    print(f"  genes                 {len(table['genes'])}")
    print(f"  element windows        {checked} ({by_kind})")
    print(f"  frame self-check       {checked}/{checked} windows match their precursor slice")
    print(f"  sha256                 {digest}")


if __name__ == "__main__":
    main()
