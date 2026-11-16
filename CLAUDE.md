# polars_kir_regions — working notes

Two Polars expressions over a KIR sequence column: `chunk` (ectodomain regions) and `loops`
(HLA-contacting loops). Both take the gene as an **input**. Verification lives in
`CHUNK_REPORT.md` and `LOOPS_REPORT.md`; the region specification is `../docs/REGIONS.md`.

This package was briefly developed inside `polars_kir_hmm` and then split out. If you are looking
for its history, that is where it is.

## Key invariants

**1. The gene is an input, never inferred.** No classifier, no models, no `type_chain`. That is
the reason this package exists apart from `polars_kir_hmm`: gene calling was the only part of the
problem that needed a trained model. A null, empty or unrecognised gene gives a null row with a
note. Do not add a fallback that guesses — a caller who does not know the gene should classify
first and pass the answer in.

**2. No runtime dependency on the HMM stack, and it is asserted.**
`tests/test_independence.py` imports this package in a fresh interpreter and fails if
`polars_kir_hmm`, `histo_kir_hmm` or `histo_kir` appear in `sys.modules`; it then runs both
expressions with those imports actively blocked, which also catches a lazy import inside a
function. `histo_kir` is a **dev** dependency: `codegen/` and the parity tests read it, the wheel
does not. `src/sequence.rs` exists for this reason — `clean_sequence` is copied rather than
imported.

**3. `chunk` emits the ectodomain only, and the emitted fields do NOT tile the precursor.**
Emitted: `leader`, `signal_peptide`, `mature_leader_tail`, `D0`, `D1`, `D2`, `ectodomain`. Stem,
TM and cytoplasmic are deliberately absent — median shortfall 127 residues, measured. Do not
"complete" the struct by adding them back without checking with the user; the scope reduction was
a decision, and it dropped the weakest evidence in the region table (TM and cytoplasmic are
structurally unobservable, 0 residues in all 54 deposited units).

**4. `ectodomain` excludes the extracellular stem, and the docs must say so.** A reader who knows
the stem is extracellular has to be able to see that we left it out on purpose. Never write
"ectodomain" in a field doc without that qualification.

**5. Two identities, asserted per row and surfaced as `ectodomain_ok`.**

```
ectodomain == mature_leader_tail + D0 + D1 + D2                    # 1983/1983
ectodomain == cleaned[signal_peptide_length : D2_end]              # 1901/1901
```

The second is the one that catches a frame error. The 82 alleles where it is not checkable are
exactly the 5'-partial CDS entries, which have no mature anchor.

**6. `leader` is 2 residues longer than `signal_peptide`, and both are emitted.** 23/21 in
fourteen genes, 25/23 in KIR2DL4. Mature numbering counts from the **cleavage site**, so
mature-frame work uses `signal_peptide`, never `leader_end_exon_frame`. The two residues between
them are emitted as `mature_leader_tail` so the offset is in the data rather than in a document —
that is what makes `D0+D1+D2` being 2 short of the ectodomain visible instead of surprising.

**7. Absent regions are null, never `""`.** An empty string claims the region exists and is zero
residues long. With the stem out of scope KIR3DL3 has no missing region.

**8. On an exact sequence match, the sequence wins over the supplied gene.** A byte-identical
precursor is stronger evidence than a label, and honouring the label would return boundaries that
do not fit the sequence. The disagreement goes in `notes`. This is the one place the expression
overrides its caller; `test_a_wrong_gene_on_an_exact_match_loses_to_the_sequence` pins it.

**9. A tie set is refused, not arbitrated.** A `;`-separated gene value (how `histo_kir`
serialises a tie) gives a null row with a note. Measured on the emitted `ectodomain` over all
1983 alleles: **125 alleles** share an ectodomain with an allele of a different gene — 5 distinct
sequences across **two** pairs, KIR2DL5A/KIR2DL5B (71 alleles, 4 sequences) and KIR2DL3/KIR2DS2
(54 alleles, 1 sequence). KIR2DL2/KIR2DL3 share none. So a tie can be genuinely unresolvable;
picking the first entry would be a wrong answer that looks right.

(An earlier draft of this file said "127 alleles across three named pairs", counting a
KIR2DL2/KIR2DL3 collision that does not exist under the ectodomain-only span. 125/2 is the
figure for what `chunk` emits.)

**10. The vendored table is keyed on the PROTEIN sequence, so CDS annotation can disagree within a
group.** 892 distinct sequences carry 1983 allele names. Gene, architecture and all boundaries
must agree within a group — asserted at vendoring time. CDS partiality and expression status
describe the *CDS* and 4 groups disagree, so the merge reports the **cautious** value and names the
disagreement in `notes`. Measured consequence: 130 rows report partial where the table marks 105,
and **0 alleles are told their CDS is complete when it is not**. Keep the asymmetry in that
direction.

**11. `src/align.rs` reproduces the reference aligner AS IT BEHAVES, not as documented.**
`histo_kir.numbering._aligner()` sets the end-gap scores to 0.0 and *then* sets `open_gap_score`,
which in Biopython aliases and overwrites them — so end gaps are penalised exactly like internal
ones despite the docstring. The traceback prefers the match state on a score tie, which is
load-bearing: preferring the gap state leaves every score identical while changing many mappings.
Do not "fix" either to match the docstring; that would break parity with the validated results.

**12. Region boundaries index the CLEANED sequence.** Uppercased, then filtered to the 20 amino
acids plus `X`. A stray gap character shifts every downstream boundary, so cleaning is not
cosmetic here.

## Key invariants — `loops`

> **Provenance note.** I overwrote this file while it held the `loops` track's own invariants, and
> there was no copy to restore from. The items below are reconstructed from `LOOPS_REPORT.md` and
> `src/loops*.rs`, which survived intact — so the *substance* is that track's measured work, but the
> wording is mine and it may be missing items that were only ever recorded here. `loops`
> maintainer: please check this section against your intent and add anything I lost.

**13. The loop windows are mature numbering from the CLEAVAGE SITE.** Measured both directions:
**81/81** published windows reproduce from `signal_peptide_length`, **0/81** from
`leader_end_exon_frame`. The two differ by exactly 2 in all 15 genes, and a window under the wrong
origin still returns the right *number* of plausible residues — so nothing downstream would flag
it. `test_frame_is_the_cleavage_site` asserts both directions deliberately: a test that only
confirmed the right answer would still pass if the two origins converged.

**14. Nine elements, and the three non-published ones are labelled as such.** Six published loops
(Boyington & Sun 2001, Fig. 5) as `A'B`, `CC'`, `EF`, `hinge`, `BC`, `FG`, plus the KIR3D-only
`D0a`, `D0b` and `CC'ext`. The `kind` field carries `published_loop` / `d0_patch` /
`ccprime_extension` — do not flatten that distinction, it is the difference between a published
window and an extrapolated one. Per gene by architecture: D1D2 → 6, D0D2 → 3, D0D1D2 → 9.

**15. Every element lies inside the ectodomain — 93/93, 0 after the D2 end.** So the ectodomain
scope costs no loop. `test_every_element_inside_ectodomain` recomputes this from the shipped table
rather than trusting the report.

**16. A window that does not transfer whole emits a NULL sequence, never a shortened slice**, with
`status` `partial` or `absent` and `n_observed`/`n_expected` recording the shortfall. Five windows
across the 54 deposited chains are in this state; three of them are a lone D1 domain whose D2 loops
genuinely do not exist.

**17. Free end gaps would change 28 of 13977 allele windows** and 0 of 378 chain windows, all on
heavily-diverged or truncated alleles. Parity with the reference is the contract, so invariant 11
applies here too — the defect is documented, not silently corrected. If it is fixed upstream,
re-vendor and re-measure.

**18. Structural support is uneven and must be stated.** Only 7 of the 15 genes have any deposited
chain in the corpus; the rest are transferred from the gene reference alone. Do not present a
transferred window for an unrepresented gene as structurally observed.

## Build gotchas

**A PyO3 cdylib cannot be linked by bare `cargo build` on macOS.** Use `maturin build` /
`maturin develop`. `cargo check` is fine and is the fast way to type-check.

**`polars-arrow` must stay `default-features = false`.** The IPC compression features pull in
`lz4` and `zstd`, whose crate archives carry `.git*` paths that a write-restricted sandbox cannot
unpack — so the minimal feature set is load-bearing for the build, not merely tidy.

**A literal gene arrives as a length-1 column.** Polars does not broadcast a literal for a plugin
the way it does for a built-in expression, so the row index must be clamped:
`gene_ca.get(if gene_len == 1 { 0 } else { row })`. Reading `get(row)` directly panics past row 0
— `chunk(gene="KIR3DL1")` worked on a one-row frame and crashed on any larger one. Both
expressions now also reject a gene column whose height is neither 1 nor `n`.

**Run the tests against the INSTALLED wheel, never with `PYTHONPATH=python` from the package
root.** The source tree has no compiled `.so` beside it, so putting `python/` first on the path
shadows the installed package and every test that touches an expression dies with
`FileNotFoundError: no dynamic library found at path .../python/polars_kir_regions` — measured
26 failed / 16 errors, which reads as breakage and is not. Working invocations:

```bash
maturin develop --release && pytest tests/ -q      # editable install, the usual loop
# or, against a built wheel:
pip install --no-deps --target /tmp/site dist/*.whl && PYTHONPATH=/tmp/site pytest tests/ -q
```

**The struct emits its own `gene`,** so `unnest` beside an input column of that name raises
`DuplicateError`. That is a caller-side collision, documented in the method docstrings and the
README, not something to fix by renaming the field: the emitted `gene` is not always the one that
was passed (see invariant 8).

## Re-vendoring after a Phase 0 change

```bash
python codegen/vendor_regions.py --kirs-root ..
python codegen/vendor_loops.py --kirs-root ..
maturin build --release --out dist
pytest tests/ -q
```

Both scripts hard-fail rather than warn: they re-verify tiling, contiguity, shared-group agreement
and the substitution matrix against Biopython before writing. If a script exits non-zero, the
vendored JSON is unchanged and the reason is on stdout.

## Scope

Two expressions. `chunk` covers the exon-derived ectodomain regions; `loops` covers the
HLA-contacting windows. Both read vendored `histo_kir` Phase 0 output rather than deriving
anything, so this package is a consumer of that work and not a second implementation of it. Do not
add a third expression, or a gene-calling fallback, without checking with the user.
