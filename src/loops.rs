//! The six HLA-contacting loops, plus the three KIR3D-only contact elements.
//!
//! Emits, per sequence, the nine named elements Phase 0 recorded: the six published loops of
//! Boyington & Sun (2001) Fig. 5 — `A'B`, `CC'`, `EF`, `hinge`, `BC`, `FG` — and, for the four
//! three-domain genes only, the two D0 contact patches `D0a`/`D0b` and the single-residue
//! `CC'ext`. All nine lie inside D0+D1+D2, so the ectodomain scope of this package costs no loop:
//! the vendored table holds 93 gene-level element windows (81 published loops across the 15 genes,
//! 8 D0 patches and 4 `CC'ext` positions across the four three-domain genes) and 93/93 end at or
//! before their gene's mature D2 end. `tests/test_loops.py::test_every_element_inside_ectodomain`
//! recomputes both figures from the shipped table rather than trusting this comment.
//!
//! **The numbering frame is the whole difficulty, so it is stated once here.** The windows in
//! `kir_loop_windows.csv` are MATURE numbering counted from the signal-peptide **cleavage site**.
//! A precursor position is therefore `signal_peptide_length + mature_position`. The exon-derived
//! `leader` region is a *different* origin — it runs 2 residues past the cleavage site in all 15
//! genes (23 vs 21; 25 vs 23 for KIR2DL4) — so using `leader_end_exon_frame` as the origin shifts
//! every window by exactly 2. Measured on the 15 gene reference alleles: the cleavage-site origin
//! reproduces the recorded window sequence **81/81**, the exon-leader origin **0/81**. The failure
//! is silent — a 2-shifted window still returns the right number of plausible-looking residues —
//! which is why `tests/test_loops.py` asserts the 81/81 rather than trusting this comment.
//!
//! **Transfer is by gapped alignment, never by a fixed offset.** 277 of the 1983 IPD-KIR alleles
//! differ in length from their gene's reference allele, so an offset is wrong for a seventh of the
//! corpus. Each query is aligned to its gene's **reference precursor** (`align.rs`, the method
//! `histo_kir.numbering` uses) and each window position is read through that alignment.
//!
//! Aligning to the reference *precursor* rather than stripping `signal_peptide_length` residues
//! off the query is deliberate, and it is not cosmetic. A query is not guaranteed to begin at the
//! initiator Met: 105 IPD-KIR alleles are 5'-partial, and for those, cutting `signal_peptide_length`
//! characters off the front removes real mature residues. Measured: query-stripping loses 41
//! element windows across 40 alleles that precursor alignment recovers correctly, and every one of
//! those 40 is flagged `cds_partial_5prime` in the region table — a 40/40 correlation. The
//! precursor path also needs no branch for a mature-numbered structural fragment: free-standing
//! ectodomains align to the mature part of the reference precursor on their own. Verified on the 54
//! deposited chains: 378/378 element windows identical to `histo_kir`'s.
//!
//! **A window that does not transfer completely is null, never truncated.** A loop that aligns
//! partially is not a loop that has moved; reporting `min`/`max` over the aligned subset would
//! present a wrong slice as a measurement. `status` distinguishes `complete` / `partial` /
//! `absent` so a caller can tell "this allele lacks the window" from "this window was not
//! observed".
//!
//! **The gene is a parameter, never a decision made here.** `loops_for` takes the gene name from
//! its caller, which takes it from the caller's dataframe. This module has no classifier and no
//! notion of a "best" gene: given a gene it transfers that gene's windows, and given no usable gene
//! it returns nothing. There are consequently no tie fields — see `loops_expr.rs` for why supplying
//! the gene removes the ambiguity that made them necessary.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock};

use serde::Deserialize;

use crate::align;

/// Canonical element order, N- to C-terminal in the KIR2D frame with the D0 elements first for the
/// three-domain genes. Matches `histo_kir.reference.LOOP_ORDER` extended with the Phase 0
/// amendments.
pub const ELEMENT_ORDER: [&str; 9] = [
    "A'B", "CC'", "EF", "hinge", "BC", "FG", "D0a", "D0b", "CC'ext",
];

// ---------------------------------------------------------------------------------------------
// The vendored window table
// ---------------------------------------------------------------------------------------------

#[derive(Deserialize)]
pub struct ElementWindow {
    pub name: String,
    /// `published_loop`, `d0_patch` or `ccprime_extension` — the evidence class of the window.
    pub kind: String,
    pub domain: String,
    pub helix: String,
    /// Inclusive, MATURE numbering from the signal-peptide cleavage site.
    pub mature_start: u32,
    pub mature_end: u32,
    /// The window as it reads in this gene's reference allele. A frame self-check, not an output.
    pub sequence: String,
}

#[derive(Deserialize)]
pub struct GeneWindows {
    pub architecture: String,
    /// The allele the windows were measured on. Retained as provenance for the transfer even
    /// though the transfer itself indexes the encoded sequence rather than the name.
    #[allow(dead_code)]
    pub reference_allele: String,
    pub signal_peptide_length: u32,
    pub precursor: String,
    pub elements: Vec<ElementWindow>,
}

#[derive(Deserialize)]
struct RawTable {
    genes: HashMap<String, GeneWindows>,
}

/// The window table, with each gene's reference precursor pre-encoded for alignment.
pub struct LoopTable {
    pub genes: HashMap<String, GeneWindows>,
    encoded: HashMap<String, Vec<u8>>,
}

impl LoopTable {
    pub fn gene(&self, gene: &str) -> Option<&GeneWindows> {
        self.genes.get(gene)
    }
}

fn read_table(path: &Path) -> Result<LoopTable, String> {
    let text = std::fs::read_to_string(path).map_err(|e| {
        // Phrased without the words "from <package>": the independence test greps shipped source
        // line by line for an import of the reference packages, and a prose "from histo_kir/..."
        // reads as one. The source path is still named, just the other way round.
        format!(
            "cannot read the loop-window table at {}: {e}. Vendor it by running \
             codegen/vendor_loops.py, which reads kir_loop_windows.csv in the reference tree.",
            path.display()
        )
    })?;
    let raw: RawTable = serde_json::from_str(&text)
        .map_err(|e| format!("malformed loop-window table at {}: {e}", path.display()))?;

    // Frame self-check at load: every recorded window sequence must be the slice its own
    // coordinates name in its own gene's reference precursor. This is what catches a table
    // vendored in the wrong frame -- the 2-residue shift -- before any query is scored.
    for (gene, entry) in &raw.genes {
        let sig = entry.signal_peptide_length as usize;
        let bytes = entry.precursor.as_bytes();
        for element in &entry.elements {
            let from = sig + element.mature_start as usize - 1;
            let to = sig + element.mature_end as usize;
            let slice = bytes.get(from..to).ok_or_else(|| {
                format!("{gene} {}: window runs past the reference precursor", element.name)
            })?;
            if slice != element.sequence.as_bytes() {
                return Err(format!(
                    "{gene} {}: the vendored table is in the wrong numbering frame -- mature \
                     {}-{} of the reference precursor reads {:?} but the table records {:?}. \
                     The windows are mature numbering from the signal-peptide CLEAVAGE site; a \
                     2-residue shift means the exon-derived leader length was used as the origin.",
                    element.name,
                    element.mature_start,
                    element.mature_end,
                    String::from_utf8_lossy(slice),
                    element.sequence,
                ));
            }
        }
    }

    let encoded = raw
        .genes
        .iter()
        .map(|(gene, entry)| (gene.clone(), align::encode(entry.precursor.as_bytes())))
        .collect();
    Ok(LoopTable {
        genes: raw.genes,
        encoded,
    })
}

type Cache = Mutex<HashMap<PathBuf, Arc<LoopTable>>>;

fn cache() -> &'static Cache {
    static CACHE: OnceLock<Cache> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Load the table for a model directory, once per directory per process.
pub fn load(model_dir: &Path) -> Result<Arc<LoopTable>, String> {
    let path = model_dir.join("loop_windows.json");
    let mut guard = cache().lock().map_err(|_| "loop-table cache poisoned".to_string())?;
    if let Some(found) = guard.get(&path) {
        return Ok(Arc::clone(found));
    }
    let table = Arc::new(read_table(&path)?);
    guard.insert(path, Arc::clone(&table));
    Ok(table)
}

// ---------------------------------------------------------------------------------------------
// Per-sequence transfer
// ---------------------------------------------------------------------------------------------

/// One element as observed on one query sequence.
pub struct ObservedElement {
    pub name: &'static str,
    pub kind: String,
    pub domain: String,
    pub helix: String,
    pub mature_start: u32,
    pub mature_end: u32,
    /// 1-based indices into the cleaned query, or `None` when the window did not transfer whole.
    pub start: Option<u32>,
    pub end: Option<u32>,
    pub sequence: Option<String>,
    /// `complete`, `partial` or `absent`.
    pub status: &'static str,
    pub n_observed: u32,
    pub n_expected: u32,
    /// Whether the sliced sequence equals the gene reference's. An observation, not a check:
    /// a `false` is a divergent allotype and must stay visible.
    pub matches_reference: bool,
}

/// The loop record for one sequence.
pub struct LoopCall {
    /// The gene whose windows were transferred — echoed back from the caller's input, null when no
    /// usable gene was supplied. This is never a gene this module chose.
    pub gene: Option<String>,
    pub architecture: Option<String>,
    pub identity: Option<f64>,
    pub n_mapped: Option<u32>,
    pub elements: Vec<ObservedElement>,
    pub notes: Vec<String>,
}

/// Transfer one gene's windows onto one cleaned query.
fn transfer(entry: &GeneWindows, encoded_reference: &[u8], cleaned: &str, encoded_query: &[u8])
    -> (Vec<ObservedElement>, f64, u32)
{
    let alignment = align::align(encoded_reference, encoded_query);
    let reference_to_query = alignment.reference_to_query(encoded_reference.len());
    let sig = entry.signal_peptide_length as i64;
    let bytes = cleaned.as_bytes();

    let mut out = Vec::with_capacity(entry.elements.len());
    for element in &entry.elements {
        let n_expected = element.mature_end - element.mature_start + 1;
        let mut indices = Vec::with_capacity(n_expected as usize);
        for mature in element.mature_start..=element.mature_end {
            let reference_position = (sig + mature as i64) as usize;
            let found = reference_to_query
                .get(reference_position)
                .copied()
                .unwrap_or(-1);
            if found >= 0 {
                indices.push(found as usize);
            }
        }
        let n_observed = indices.len() as u32;
        let name = ELEMENT_ORDER
            .iter()
            .find(|candidate| **candidate == element.name)
            .copied()
            .unwrap_or("?");

        // A window is emitted only when every one of its positions transferred. Anything less is
        // reported as partial with a null slice -- never as a shortened sequence.
        let (start, end, sequence, status) = if n_observed == n_expected {
            let text: String = indices.iter().map(|&i| bytes[i] as char).collect();
            (
                Some(indices[0] as u32 + 1),
                Some(indices[indices.len() - 1] as u32 + 1),
                Some(text),
                "complete",
            )
        } else if n_observed > 0 {
            (None, None, None, "partial")
        } else {
            (None, None, None, "absent")
        };

        let matches_reference = sequence.as_deref() == Some(element.sequence.as_str());
        out.push(ObservedElement {
            name,
            kind: element.kind.clone(),
            domain: element.domain.clone(),
            helix: element.helix.clone(),
            mature_start: element.mature_start,
            mature_end: element.mature_end,
            start,
            end,
            sequence,
            status,
            n_observed,
            n_expected,
            matches_reference,
        });
    }
    (out, alignment.identity, alignment.n_mapped as u32)
}

/// Slice the nine elements out of one sequence, for one named gene.
///
/// `gene` comes from the caller, which takes it from the caller's dataframe. An unrecognised name
/// returns an empty record with an explanatory note rather than a guess: a window sliced from the
/// wrong gene's coordinates is a plausible-looking wrong answer, which is worse than a null.
pub fn loops_for(cleaned: &str, gene: &str, table: &LoopTable) -> LoopCall {
    let mut notes: Vec<String> = Vec::new();

    let Some(entry) = table.genes.get(gene) else {
        notes.push(format!(
            "no loop windows are recorded for gene {gene:?}, so nothing was transferred"
        ));
        return LoopCall {
            gene: None,
            architecture: None,
            identity: None,
            n_mapped: None,
            elements: Vec::new(),
            notes,
        };
    };

    let encoded_query = align::encode(cleaned.as_bytes());
    let (elements, identity, n_mapped) =
        transfer(entry, &table.encoded[gene], cleaned, &encoded_query);

    if entry.architecture == "D0D2" {
        notes.push(
            "D0D2 architecture: the three D1 loops do not exist and the D0 contact windows are \
             undetermined -- the D0 domain differs in length from KIR3D's and no D0D2-pMHC \
             complex has been solved, so KIR3DL1's D0 patch is not transferred here"
                .to_string(),
        );
    }

    LoopCall {
        gene: Some(gene.to_string()),
        architecture: Some(entry.architecture.clone()),
        identity: Some(identity),
        n_mapped: Some(n_mapped),
        elements,
        notes,
    }
}
