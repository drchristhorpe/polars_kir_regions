# The `loops` expression — measured agreement

Every number here comes from a run against **this package** in the session that wrote this file.
Figures measured earlier against the pre-split `polars_kir_hmm` build were re-run here rather than
carried over; the commands are given so each can be reproduced.

## What it emits

```python
import polars as pl
import polars_kir_regions  # registers the .kir_regions namespace

out = df.with_columns(loops=pl.col("sequence").kir_regions.loops(gene="gene"))
out.select(
    pl.col("loops").struct.field("FG").struct.field("sequence").alias("FG"),
    pl.col("loops").struct.field("hinge").struct.field("sequence").alias("hinge"),
)
```

Nine named struct fields — the six published loops of Boyington & Sun (*Mol Immunol* 38:1007–1021,
2001, Fig. 5) as `A'B`, `CC'`, `EF`, `hinge`, `BC`, `FG`, plus the KIR3D-only `D0a`, `D0b` and
`CC'ext` — each a struct of `element`, `kind`, `domain`, `hla_helix`, `mature_start`, `mature_end`,
`start`, `end`, `sequence`, `status`, `n_observed`, `n_expected`, `matches_reference`. Top level
also carries `gene`, `architecture`, `provenance`, `identity`, `n_mapped`, `elements` (the same nine
as a list) and `notes`.

**The gene is an input.** It comes from the caller's dataframe — a column name, a literal, or an
expression — and this package has no classifier, no trained models and no tie arbitration. That is
why it is standalone: the gene-call path was the only coupling the loop slicing ever had to the HMM.
An unknown gene, a null gene or a `;`-separated tie set yields null fields plus a note, never a
guess.

## The numbering frame

This is the whole difficulty, and it is decided by measurement rather than assumption.

The loop windows are **mature numbering counted from the signal-peptide cleavage site**. The Phase 0
region table is precursor numbering. Converting requires `signal_peptide_length`, not
`leader_end_exon_frame`.

| origin | published windows reproduced exactly |
|---|---|
| signal-peptide cleavage site (`signal_peptide_length`) | **81 / 81** |
| exon-derived leader end (`leader_end_exon_frame`) | **0 / 81** |

`leader_end_exon_frame - signal_peptide_length` is **exactly 2 in all 15 genes**. Every window under
the wrong origin is shifted by 2 and still returns the right *number* of plausible residues, so
nothing downstream would flag it. `tests/test_loops.py::test_frame_is_the_cleavage_site` asserts
both directions — that the right origin reproduces every window *and* the wrong one reproduces none
— because a test that only confirms the right answer would still pass if the two origins converged.

```bash
python tools/loops_frame_and_count.py
```

## Agreement with `histo_kir`

Compared field by field against `tests/data/loops_golden.json.gz`, recorded from
`histo_kir.numbering.align_to_reference`. Seven fields per element: `sequence`, `status`, `start`,
`end`, `n_observed`, `n_expected`, `matches_reference`.

| corpus | element comparisons | identical | disagreements |
|---|---|---|---|
| 1983 IPD-KIR 2.15.0 alleles | 17847 | **17847** | **0** |
| 54 deposited KIR chains | 486 | **486** | **0** |

Golden status breakdown, alleles: 13855 complete, 113 absent, 9 partial, plus 3870
architecturally-absent null structs. Chains: 373 complete, 3 absent, 2 partial, plus 108 null
structs. The 1983-allele column slices in **2.0 s**.

A literal gene and the same gene supplied as a column give identical results: **180 / 180** element
sequences over the 20 KIR3DL1 chains, 0 differing.

```bash
python tools/loops_parity_report.py     # writes tools/loops_parity.json
```

## Against the deposited coordinates

Each element slice compared with the residues actually present in the PDB at the file residue
numbers the window implies.

| | |
|---|---|
| comparisons | 373 |
| identical | **373** |
| disagreements | **0** |
| numbering offsets applied | 42 chains at 0, **12 chains at +21** |

The 12 precursor-numbered chains are the eight `9f2d` chains, `9hml_B` and the three `9lr*` chains,
per `validation/numbering_conventions.csv`. Ignoring that offset produces a spurious result.

Two project invariants govern how a structure is read here, and each cost a bug when ignored: the
framed files carry **role-based** chain letters (the KIR chain is `K` in every file, whatever the
depositor called it), so the chain is located by matching the scored sequence rather than by parsing
the filename; and a file residue number is not a position, so it is used only to look up what the
mature-frame window predicts, after the offset.

## Loop-in-domain

Each element checked against the Phase 0 per-allele region boundaries converted to the mature frame:
`A'B`/`CC'`/`EF`/`CC'ext` in D1, `BC`/`FG`/`hinge` in D2, `D0a`/`D0b` in D0.

| | |
|---|---|
| elements checked | 93 |
| inside the region their name implies | **93** |
| outside | **0** |
| hinge wholly inside D2 | **15 / 15** genes |
| hinge contains an exon-junction (shared) codon | **15 / 15** genes |
| … and that codon is the window's **first** residue | **15 / 15** genes |

The hinge does not straddle a region under the Phase 0 partition.

## Ectodomain scope

The emitted regions stop at the D2 end, so it matters that no loop lies beyond it.

| | |
|---|---|
| gene-level element windows | **93** |
| `published_loop` | 81 |
| `d0_patch` | 8 |
| `ccprime_extension` | 4 |
| ending at or before the gene's mature D2 end | **93 / 93** |
| after the D2 end | **0** |

Elements per gene by architecture: D1D2 → 6, D0D2 → 3, D0D1D2 → 9. The ectodomain scope costs no
loop. `tests/test_loops.py::test_every_element_inside_ectodomain` recomputes all three figures from
the shipped table rather than trusting this table.

## Transfer failures, reported not hidden

Five windows across the 54 chains did not transfer whole. Each emits a **null** sequence with
`status` `partial` or `absent` — never a shortened slice.

| chain | element | status | observed / expected | why |
|---|---|---|---|---|
| `8tui_A` | hinge | absent | 0 / 5 | a lone 98-residue KIR2DL3 D1 domain; its D2 loops genuinely do not exist |
| `8tui_A` | BC | absent | 0 / 6 | as above |
| `8tui_A` | FG | absent | 0 / 9 | as above |
| `9bl6_G` | `D0b` | partial | 5 / 8 | disordered stretch |
| `9bla_G` | `CC'` | partial | 2 / 4 | disordered stretch |

## Why alignment transfer, not a fixed offset

277 of the 1983 alleles differ in length from their gene reference, so a fixed offset is wrong for
those. Two further findings settled the design:

- **Align to the reference PRECURSOR, do not strip the signal peptide off the query.** Stripping
  `signal_peptide_length` characters loses 41 element windows across 40 alleles, and **40 / 40** of
  those alleles carry `cds_partial_5prime` — their translation begins *inside* the signal peptide,
  so stripping removes real mature residues. `KIR2DL2*011` is the test case.
- **Aligning a full precursor against a *mature* reference is unsound.** On the 39-residue null
  allele `KIR2DL1*0450101N` it lands signal-peptide residues on the `EF` window — a
  plausible-looking wrong slice. The precursor-to-precursor path returns nothing there, correctly.

One code path serves full precursors and free-standing mature structural fragments, with no branch:
a deposited ectodomain aligns to the mature part of the reference precursor unaided.

## Provenance

`provenance` distinguishes boundaries read from an annotation from boundaries transferred onto a
sequence:

| value | meaning | count |
|---|---|---|
| `annotated` | the cleaned sequence **is** the gene reference's precursor | 242 of 1983 alleles |
| `inferred` | boundaries transferred by alignment | 1741 of 1983 alleles; **all 54** chains |

242 rather than 15 because IPD-KIR alleles can share a precursor translation — several alleles per
gene are byte-identical to their gene's reference at the protein level while differing in DNA.

## The aligner

`src/align.rs` transcribes what `histo_kir.numbering.align_to_reference` **actually does**, not what
its docstring says. Three details are deliberate:

- BLOSUM62 (Biopython `scoring="blastp"`), gap open −12, extend −1.
- **End gaps are penalized exactly as internal gaps are.** The reference assigns
  `end_insertion_score = 0.0` and then assigns `open_gap_score = -12.0`, which in Biopython
  overwrites all four terminal open scores. Read off the live object: `open_left_insertion_score` is
  `0.0` after the end scores are set and `-12.0` after the general assignment — so the effective
  behaviour contradicts the reference's own docstring. Free end gaps would change **28 of 13977**
  allele windows and **0 of 378** chain windows, all on heavily-diverged or truncated alleles
  (`KIR2DS4*003*`, and null alleles such as `KIR2DL1*114N` and `KIR2DL3*008N`). Parity with the
  reference is the contract, so the reference's real behaviour is what is reproduced; the defect is
  documented rather than silently corrected. If it is fixed upstream, re-vendor and re-measure.
- **Match-state-first traceback**, which reproduces Biopython's `alignment[0]` on **14355 / 14355**
  window comparisons across all 2037 sequences (1983 alleles + 54 chains), 0 disagreements.
  **185 of 1983** alleles have more than one optimal alignment, but only **one** —
  `KIR3DL3*080N`, a null allele — has a tie whose branches give *different* windows, and match-first
  agrees with Biopython there too.

```bash
python tools/loops_aligner_check.py     # writes tools/loops_aligner_check.json
```

This package carries its own copy of the aligner, and its own `sequence.rs`, so that neither it nor
`polars_kir_hmm` imports the other.

## Structural support is uneven

Only **7 of the 15** genes have any deposited chain in the corpus: KIR3DL1 (20 chains), KIR2DL2
(13), KIR2DL1 (11), KIR2DS2 (4), KIR2DL3 (3), KIR2DL4 (2), KIR2DS4 (1). The windows for the other
eight genes are alignment **predictions**, not observations. `loop_windows()` marks them
`predicted_only`; any interface-coverage statistic across all 15 genes should exclude or flag them.

## Per-gene summary figure

![loops verification]({{artifact:art_17a20d1b-ff8a-4871-9e80-bfd4688e0327}})

Regenerate with `python tools/loops_verification_figure.py`. Panel **a** puts all nine element
windows on the mature axis with the D0/D1/D2 extents behind them — the three D0D2 genes visibly lack
the D1 loops, and `D0a`/`D0b`/`CC'ext` appear only on the four three-domain genes. Panel **b** is
the frame result per gene. Panel **c** shows the uneven structural support.

## Tests

`tests/test_loops.py` — 21 tests, real committed fixtures only. The suite pins the frame (both
directions), full-corpus and per-chain parity against the golden, the vendored table's own frame,
the 93-element scope recount, literal-gene vs gene-column equivalence, unknown/null/tie gene
handling, the architecture rules (D1 loops absent from D0D2; `D0a`/`D0b`/`CC'ext` only for the four
three-domain genes), null-not-truncated on `8tui_A`, the 5'-partial recovery on `KIR2DL2*011`,
provenance, and two anti-coupling guards — that the Python module imports neither reference package
and the built extension carries neither symbol, and that no loop module reintroduces
`use crate::classify`.

```bash
python -m pytest tests/test_loops.py -q
```
