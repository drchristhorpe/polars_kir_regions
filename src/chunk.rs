//! Slicing the pMHC-interacting ectodomain out of a KIR sequence.
//!
//! **Scope: the ectodomain only.** The emitted fields are the leader, the signal peptide, the
//! three Ig domains D0/D1/D2 and their concatenation. The stem, the transmembrane helix and the
//! cytoplasmic tail are deliberately **not** emitted, by scope decision -- the question this
//! package serves is KIR/pMHC recognition, and those three regions do not participate in it.
//! They remain in `histo_kir/data/kir_allele_regions.parquet`, which is unchanged; this is a
//! decision about what the expression surfaces, not about the reference data.
//!
//! Dropping them also drops the weakest evidence in the region table. TM and cytoplasmic are
//! structurally unobservable -- 0 residues in all 54 deposited KIR units -- so they rested on
//! prediction-level annotation alone. Exon 7 spans all three topological compartments, which
//! made the exon-derived `cyt` 7-9 residues shorter than UniProt's cytoplasmic compartment, a
//! caveat that no longer has to be carried. And the unresolved KIR3DL3 stem conflict (histo_kir
//! places a 7-residue stem at mature 295-301, the exon parse finds no exon 6 in 238/238 KIR3DL3
//! alleles) is now out of scope -- it could never be settled structurally anyway, because
//! KIR3DL3 protein is not expressed in healthy individuals.
//!
//! **The emitted fields do NOT tile the precursor, and concatenating them gives a short
//! sequence.** They stop at the D2 end, so everything from the stem onward is absent by
//! construction. Anyone reaching for a full-length reassembly wants
//! `kir_allele_regions.parquet`, which does tile (verified 1983/1983 in Phase 0).
//!
//! Two paths, and the `provenance` field always says which one produced a row:
//!
//! * **`annotated`** -- the cleaned sequence is byte-identical to a known IPD-KIR precursor, so
//!   the boundaries come straight from `kir_allele_regions.parquet`. Nothing is inferred.
//! * **`inferred`** -- it is not, so the boundaries are transferred from the given gene's
//!   reference allele by gapped alignment (`align.rs`, the method `histo_kir.numbering` uses).
//!   Materially less certain: 277 of the 1983 alleles differ in length from their gene's
//!   reference, so a divergent query is routine rather than pathological. This path is
//!   load-bearing for structural input: measured, 0 of the 144 deposited chain fragments
//!   exact-matches a precursor, while all 1983 IPD-KIR alleles do.
//! * **`none`** -- the gene value was null, empty, or names no gene in the vendored table, so
//!   there is nothing to transfer from and every region is null. A sequence with no gene is not
//!   chunked into plausible-looking pieces.
//!
//! ## The two invariants, and the 2-residue subtlety between them
//!
//! `D0 + D1 + D2` is **not** the mature ectodomain, and the difference is 2 residues. The exon
//! 1+2 product (`leader`) is consistently 2 residues longer than the cleavage-based signal
//! peptide -- 23 versus 21 in 14 genes, 25 versus 23 in KIR2DL4 -- so mature residues 1 and 2
//! are encoded by the *leader* exon while mature residue 3 is the first residue of the D0 (or
//! D1) exon region. Measured over the region table: the first Ig region starts at the cleavage
//! site + 3 in **1899 of 1983** alleles. The remaining 84 are **all** 5'-partial CDS entries
//! (verified 84/84): their sequence does not begin at the initiator Met, so the leader is
//! truncated and the first Ig region lands at an arbitrary offset -- measured spread -22..+2, so
//! the offset is not a second convention but the absence of one. Those alleles have no mature
//! anchor and report `signal_peptide`/`mature_leader_tail` as null rather than a guessed frame.
//!
//! So the two residues are extracellular, are part of the mature protein, and belong to no Ig
//! exon. They are emitted as `mature_leader_tail`, which makes the offset explicit in the output
//! rather than implicit in a document -- the same reason `leader` and `signal_peptide` are both
//! emitted. That buys one exact, universal identity:
//!
//! ```text
//! ectodomain == mature_leader_tail + D0 + D1 + D2      -- 1983/1983 alleles
//! ```
//!
//! and, wherever the cleavage site is present in the sequence,
//!
//! ```text
//! ectodomain == cleaned_input[signal_peptide_length .. D2_end]   -- 1901/1901 alleles
//! ```
//!
//! i.e. the ectodomain is a contiguous slice of the input beginning at mature residue 1, with no
//! gap and no overlap. Both are re-checked per row and reported as `ectodomain_ok`; that column
//! is what catches an off-by-one or a frame error. The 82 alleles where the second identity is
//! not checkable are exactly the 5'-partial CDS entries whose sequence does not begin at the
//! initiator Met, so the cleavage site is not in it and mature numbering has no anchor -- there
//! `signal_peptide` and `mature_leader_tail` are null and the ectodomain begins at the first Ig
//! residue instead.
//!
//! **`ectodomain` here is D0+D1+D2 and therefore excludes the stem, which is genuinely
//! extracellular.** That is the scope decision above, not an error, and the field documentation
//! says so. A caller wanting the full extracellular extent must go to the region table.
//!
//! **Absent regions are null, never the empty string.** D0 is null for the eight D1D2 genes and
//! D1 for the three D0D2 genes. An empty string would claim the region exists and is zero
//! residues long, which is a different and false statement. With the stem out of scope,
//! **KIR3DL3 no longer has a missing region** -- its D2 runs to the ectodomain end and it
//! behaves like every other three-domain gene. Verified: 0 nulls across D0/D1/D2/leader for all
//! 220 expressed complete-CDS KIR3DL3 alleles, matching KIR3DL1, KIR3DL2 and KIR3DS1 exactly.
//!
//! **The gene is an input, not an inference.** The caller's dataframe supplies it, so this
//! module contains no gene calling and no tie handling. That is deliberate and it is what makes
//! this crate standalone: the gene call was the only thing here that needed a trained model. A
//! caller that does not know the gene should type the sequence first -- with `polars_kir_hmm`'s
//! `classify`, or any other method -- and pass the answer in. A caller that passes a tie set has
//! made a decision this module will not second-guess; see `loops_expr.rs`, which accepts a
//! `;`-separated value for exactly that case.

use sha2::{Digest, Sha256};

use crate::align;
use crate::regions::{AlleleEntry, GeneEntry, RegionTable, REGION_NAMES};
use crate::sequence::clean_sequence;

/// How a row's boundaries were obtained. A caller must be able to tell these apart, so the
/// values are a small closed vocabulary rather than free text.
pub const PROVENANCE_ANNOTATED: &str = "annotated";
pub const PROVENANCE_INFERRED: &str = "inferred";
pub const PROVENANCE_NONE: &str = "none";

/// One sequence's ectodomain regions.
pub struct Chunks {
    /// Region slices in `REGION_NAMES` order. Slots 4, 5 and 6 (stem, TM, cyt) are computed on
    /// the way through -- the vendored table and the transfer both work over all seven -- but
    /// are **not** emitted, per the ectodomain scope in the module docs.
    pub regions: [Option<String>; 7],
    /// The cleavage-based mature signal peptide: the first `signal_peptide_length` residues.
    /// **Two residues shorter than `leader`**, which is the exon 1+2 product. Both are emitted
    /// because mature numbering counts from the cleavage site while the exon frame does not;
    /// using the wrong one shifts every mature position by 2 (`docs/REGIONS.md` section 4).
    /// Null when the sequence does not contain the whole signal peptide (5'-partial CDS).
    pub signal_peptide: Option<String>,
    /// The mature residues encoded by the leader exon: `leader` minus the signal peptide, so
    /// normally mature residues 1-2. Extracellular, part of the mature protein, and in no Ig
    /// exon -- which is why `D0+D1+D2` alone is 2 residues short of the mature ectodomain.
    /// Null when the cleavage site is not in the sequence.
    pub mature_leader_tail: Option<String>,
    /// `mature_leader_tail + D0 + D1 + D2`: the mature ectodomain from residue 1 to the D2 end.
    ///
    /// **Excludes the extracellular stem, by scope decision** -- see the module docs. Also
    /// excludes the 5-7 exon-7 residues N-terminal to the TM helix, which are extracellular too.
    pub ectodomain: Option<String>,
    /// Both ectodomain identities checked on this row: `ectodomain == mature_leader_tail + D0 +
    /// D1 + D2`, and `ectodomain == cleaned_input[signal_peptide_length .. D2_end]` where the
    /// cleavage site is present. A `false` is a bug in this package, not a property of the input.
    pub ectodomain_ok: bool,
    /// The gene these boundaries came from -- echoed back from the input column so a caller can
    /// join on it. Null only when the input gene was null or unknown, in which case every region
    /// is null too.
    pub gene: Option<String>,
    /// The matching allele name, or null when the sequence matches more than one allele name
    /// (they share a precursor) or when nothing matched exactly.
    pub allele: Option<String>,
    /// Every allele name with this exact precursor sequence. Empty on the inferred path.
    pub alleles: Vec<String>,
    /// This sequence's own architecture on the annotated path; the gene's on the inferred path,
    /// where an individual allele's exon usage is not observable.
    pub architecture: Option<String>,
    pub gene_architecture: Option<String>,
    pub expression_status: Option<String>,
    /// True when the matched allele's CDS is annotated partial. Null-free: false on the
    /// inferred path, where there is no annotation to read.
    pub cds_partial: bool,
    pub cds_partial_5prime: bool,
    pub cds_partial_3prime: bool,
    pub provenance: PROVENANCE,
    /// Alignment identity to the gene reference on the inferred path; null on the others.
    pub identity: Option<f64>,
    /// Number of query residues the alignment mapped, on the inferred path.
    pub n_mapped: Option<u32>,
    pub notes: Vec<String>,
    /// Length of the cleaned sequence the regions tile.
    pub length: u32,
}

#[allow(clippy::upper_case_acronyms)]
pub type PROVENANCE = &'static str;

/// `_sequence_hash`, the same 16 hex characters `classify.rs` computes and the vendoring script
/// keys the table on.
fn sequence_hash(sequence: &[u8]) -> String {
    let digest = Sha256::digest(sequence);
    let mut out = String::with_capacity(16);
    for byte in &digest[..8] {
        out.push_str(&format!("{byte:02x}"));
    }
    out
}

/// Index of a region in `REGION_NAMES`.
const LEADER: usize = 0;
const IG_REGIONS: [usize; 3] = [1, 2, 3]; // D0, D1, D2

/// Build the mature ectodomain and check both identities from the module docs.
///
/// Returns `(mature_leader_tail, ectodomain, ok)`. `ok` is the conjunction of:
///
/// 1. `ectodomain == mature_leader_tail + D0 + D1 + D2` -- always checkable.
/// 2. `ectodomain == cleaned[signal_peptide_length .. D2_end]` -- checkable only when the
///    cleavage site is inside the sequence, i.e. not on a 5'-partial CDS.
///
/// Identity 2 is the one that catches a frame error, so when it is not checkable that is
/// recorded by `mature_leader_tail` being null rather than by silently reporting `ok`.
fn build_ectodomain(
    regions: &[Option<String>; 7],
    cleaned: &str,
    signal_peptide_length: Option<usize>,
) -> (Option<String>, Option<String>, bool) {
    let ig: String = IG_REGIONS
        .iter()
        .filter_map(|&index| regions[index].as_deref())
        .collect();
    if ig.is_empty() {
        // No Ig region at all: there is no ectodomain to build, and saying so is not a failure.
        return (None, None, true);
    }

    // The leader's mature tail: the leader residues past the cleavage site. Requires both a
    // leader and a signal-peptide length that the leader actually reaches.
    let tail = match (regions[LEADER].as_deref(), signal_peptide_length) {
        (Some(leader), Some(length)) if leader.len() > length => Some(leader[length..].to_string()),
        _ => None,
    };

    let ectodomain = format!("{}{}", tail.as_deref().unwrap_or(""), ig);

    // Identity 1, by construction above; re-checked so a future edit that breaks it is caught.
    let mut ok = ectodomain == format!("{}{}", tail.as_deref().unwrap_or(""), ig);

    // Identity 2: the ectodomain must be the contiguous slice of the input that starts at mature
    // residue 1. Only checkable when the cleavage site is in the sequence.
    if let (Some(length), true) = (signal_peptide_length, tail.is_some()) {
        let end = length + ectodomain.len();
        ok = ok && end <= cleaned.len() && cleaned[length..end] == ectodomain;
    }

    (tail, Some(ectodomain), ok)
}

/// Slice the annotated spans out of the cleaned sequence.
///
/// The spans are 1-based inclusive precursor positions over the allele's own sequence, and the
/// sequence matched byte-for-byte, so they are in range by construction. A span that is not is
/// treated as absent rather than panicking or silently clamping to a shorter slice.
fn slice_annotated(entry: &AlleleEntry, cleaned: &str) -> [Option<String>; 7] {
    let mut out: [Option<String>; 7] = Default::default();
    for (index, span) in entry.spans.iter().enumerate().take(REGION_NAMES.len()) {
        if let Some((start, end)) = span {
            let (from, to) = (*start as usize - 1, *end as usize);
            if from < to && to <= cleaned.len() {
                out[index] = Some(cleaned[from..to].to_string());
            }
        }
    }
    out
}

/// Transfer one gene reference's region codes onto a query through an alignment.
///
/// **Why carry-forward rather than per-position lookup.** A query residue the alignment leaves
/// unpaired has no reference position and therefore no region code of its own. Dropping it would
/// break the tiling invariant -- the pieces would no longer reassemble into the input -- so it
/// inherits the region of the last mapped residue before it, and a leading unpaired run inherits
/// the region of the first mapped residue after it. Every residue lands in exactly one region and
/// the regions stay contiguous, which is what makes the invariant hold on this path too.
///
/// The cost of that choice is that a boundary inside an unmapped run is placed at the run's edge
/// rather than measured. That is inherent to transferring an annotation across an indel, and is
/// what `provenance = "inferred"` exists to tell the caller.
fn transfer(
    entry: &GeneEntry,
    encoded_reference: &[u8],
    cleaned: &str,
    encoded_query: &[u8],
) -> (
    [Option<String>; 7],
    f64,
    u32,
    Vec<(u32, u32)>,
) {
    let alignment = align::align(encoded_reference, encoded_query);

    let mut pieces: [Vec<u8>; 7] = Default::default();
    let bytes = cleaned.as_bytes();

    // The region of the first mapped query residue, for a leading unpaired run.
    let leading = alignment
        .query_to_reference
        .iter()
        .flatten()
        .next()
        .map(|&position| entry.owner[position as usize - 1])
        .filter(|&code| code > 0);

    let mut current = leading;
    for (index, byte) in bytes.iter().enumerate() {
        if let Some(position) = alignment.query_to_reference[index] {
            let code = entry.owner[position as usize - 1];
            if code > 0 {
                current = Some(code);
            }
        }
        if let Some(code) = current {
            pieces[code as usize - 1].push(*byte);
        }
    }

    let mut out: [Option<String>; 7] = Default::default();
    for (slot, piece) in out.iter_mut().zip(pieces) {
        if !piece.is_empty() {
            // The bytes came out of a `&str` in order, so this cannot fail.
            *slot = String::from_utf8(piece).ok();
        }
    }

    (
        out,
        alignment.identity,
        alignment.n_mapped as u32,
        alignment.reference_gaps,
    )
}

/// Chunk one sequence, given its gene.
///
/// `gene` is whatever the caller's dataframe said. It is trusted for the transfer but validated
/// against the vendored table first, so an unknown value produces a null row with a note rather
/// than a panic or a silent fallback to some other gene's boundaries.
///
/// Note the asymmetry between the paths: on the annotated path the gene is *read from the matched
/// allele*, not from the argument, because the sequence itself identifies the allele and its
/// annotation is authoritative. If the caller's gene disagrees with the match, the match wins and
/// a note records the disagreement -- a byte-identical precursor is stronger evidence than a
/// label, and silently honouring the label would return boundaries that do not fit the sequence.
pub fn chunk_one(sequence: &str, gene: Option<&str>, table: &RegionTable) -> Chunks {
    let cleaned_bytes = clean_sequence(sequence);
    // `clean_sequence` keeps only ASCII amino-acid letters, so this is valid UTF-8.
    let cleaned = String::from_utf8(cleaned_bytes.clone()).unwrap_or_default();
    let length = cleaned.len() as u32;

    let mut notes: Vec<String> = Vec::new();

    // --- the annotated path -------------------------------------------------------------
    if let Some(entry) = table.exact(&sequence_hash(&cleaned_bytes)) {
        let regions = slice_annotated(entry, &cleaned);
        let signal_length = table
            .gene(&entry.gene)
            .map(|gene| gene.signal_peptide_length as usize);
        let (mature_leader_tail, ectodomain, ectodomain_ok) =
            build_ectodomain(&regions, &cleaned, signal_length);

        if !ectodomain_ok {
            notes.push(
                "the annotated spans for this allele do not reassemble into its ectodomain; \
                 this is a vendored-table bug, not a property of the input"
                    .to_string(),
            );
        }
        if signal_length.is_some() && mature_leader_tail.is_none() {
            notes.push(
                "the sequence does not reach the signal-peptide cleavage site, so it has no \
                 mature frame: `signal_peptide` and `mature_leader_tail` are null and \
                 `ectodomain` begins at the first Ig residue rather than at mature residue 1"
                    .to_string(),
            );
        }
        if let Some(given) = gene {
            if !given.is_empty() && given != entry.gene {
                notes.push(format!(
                    "the sequence is a byte-identical match to a {} precursor, so the boundaries \
                     are that allele's own annotation; the supplied gene {:?} was not used",
                    entry.gene, given
                ));
            }
        }
        if entry.alleles.len() > 1 {
            notes.push(format!(
                "this exact precursor sequence is shared by {} allele names, so `allele` is null \
                 and `alleles` lists them",
                entry.alleles.len()
            ));
        }
        if entry.cds_partial {
            notes.push(
                "this allele's CDS is annotated partial, so regions exist only where the \
                 annotation supports them"
                    .to_string(),
            );
        }
        if !entry.annotation_mixed.is_empty() {
            // The reported value is the cautious one, so a caller is never told a sequence is
            // complete or expressed when one of its allele names is not.
            notes.push(format!(
                "the allele names sharing this exact protein sequence disagree on {}; these \
                 describe the CDS rather than the protein, so the cautious value is reported \
                 and `alleles` lists every name",
                entry.annotation_mixed.join("/")
            ));
        }

        let signal_peptide = signal_length.and_then(|n| {
            (cleaned.len() >= n).then(|| cleaned[..n].to_string())
        });

        return Chunks {
            ectodomain,
            ectodomain_ok,
            mature_leader_tail,
            regions,
            signal_peptide,
            gene: Some(entry.gene.clone()),
            allele: (entry.alleles.len() == 1).then(|| entry.alleles[0].clone()),
            alleles: entry.alleles.clone(),
            architecture: Some(entry.architecture.clone()),
            gene_architecture: Some(entry.gene_architecture.clone()),
            expression_status: Some(entry.expression_status.clone()),
            cds_partial: entry.cds_partial,
            cds_partial_5prime: entry.cds_partial_5prime,
            cds_partial_3prime: entry.cds_partial_3prime,
            provenance: PROVENANCE_ANNOTATED,
            identity: None,
            n_mapped: None,
            notes,
            length,
        };
    }

    // --- the inferred path --------------------------------------------------------------
    // No exact match, so the gene must come from the caller. There is nothing to infer it from
    // here by design: inferring it is what a classifier is for, and this crate does not have one.
    let named = gene.filter(|value| !value.is_empty());
    let entry: Option<&GeneEntry> = named.and_then(|value| table.gene(value));

    let Some(reference) = entry else {
        notes.push(match named {
            None => "no gene was supplied for a sequence that matches no known allele, so no \
                     boundaries were transferred; pass a gene column to chunk(gene=...)"
                .to_string(),
            Some(value) => format!(
                "gene {value:?} is not in the vendored region table, so no boundaries were \
                 transferred; the table covers the 15 IPD-KIR genes"
            ),
        });
        return Chunks {
            regions: Default::default(),
            signal_peptide: None,
            mature_leader_tail: None,
            ectodomain: None,
            // Nothing was assigned, so there is no ectodomain to be wrong about.
            ectodomain_ok: true,
            gene: None,
            allele: None,
            alleles: Vec::new(),
            architecture: None,
            gene_architecture: None,
            expression_status: None,
            cds_partial: false,
            cds_partial_5prime: false,
            cds_partial_3prime: false,
            provenance: PROVENANCE_NONE,
            identity: None,
            n_mapped: None,
            notes,
            length,
        };
    };

    // The reference is encoded once at table load, not per row, so the encoding cost is paid once
    // for the whole column rather than once per sequence.
    let encoded_reference = &table.encoded[named.expect("gene is Some when entry resolved")];
    let encoded_query = align::encode(cleaned.as_bytes());
    let (regions, identity, n_mapped, reference_gaps) =
        transfer(reference, encoded_reference, &cleaned, &encoded_query);
    let signal_length = Some(reference.signal_peptide_length as usize);
    let (mature_leader_tail, ectodomain, ectodomain_ok) =
        build_ectodomain(&regions, &cleaned, signal_length);

    let signal_peptide = signal_length
        .filter(|_| mature_leader_tail.is_some())
        .and_then(|n| (cleaned.len() >= n).then(|| cleaned[..n].to_string()));

    if mature_leader_tail.is_none() {
        notes.push(
            "the alignment does not cover the signal-peptide cleavage site, so this sequence has \
             no mature frame: `signal_peptide` and `mature_leader_tail` are null and \
             `ectodomain` begins at the first Ig residue rather than at mature residue 1"
                .to_string(),
        );
    }
    if !reference_gaps.is_empty() {
        notes.push(format!(
            "the alignment to the {} reference has {} internal gap(s); a boundary falling inside \
             one is placed at the gap edge rather than measured",
            reference.reference_allele,
            reference_gaps.len()
        ));
    }
    if !ectodomain_ok {
        notes.push(
            "the transferred spans do not reassemble into the reported ectodomain; this is a bug \
             in this package, not a property of the input"
                .to_string(),
        );
    }

    Chunks {
        ectodomain,
        ectodomain_ok,
        mature_leader_tail,
        regions,
        signal_peptide,
        gene: named.map(str::to_string),
        allele: None,
        alleles: Vec::new(),
        // An individual allele's exon usage is not observable without an exact match, so the
        // gene's architecture is the honest answer for both fields on this path.
        architecture: Some(reference.architecture.clone()),
        gene_architecture: Some(reference.architecture.clone()),
        expression_status: None,
        cds_partial: false,
        cds_partial_5prime: false,
        cds_partial_3prime: false,
        provenance: PROVENANCE_INFERRED,
        identity: Some(identity),
        n_mapped: Some(n_mapped),
        notes,
        length,
    }
}
