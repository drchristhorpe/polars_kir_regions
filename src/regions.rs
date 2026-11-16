//! The vendored per-allele region table, and the two lookups `chunk` needs.
//!
//! Loaded from `regions/regions.json`, written by `codegen/vendor_regions.py` out of
//! `histo_kir/data/kir_allele_regions.parquet`. Vendored at build time for the same reason the
//! models are: the boundaries are a measured, verified data product (`docs/REGIONS.md`), and
//! re-deriving them at runtime would make this package a second implementation of the exon
//! arithmetic rather than a consumer of it.
//!
//! Cached per directory exactly as `models::load` is, and for the same reason -- an expression
//! is constructed once and evaluated over every row group, so the table must not be re-parsed
//! per batch.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock};

use serde::Deserialize;

/// The seven regions, N- to C-terminal. This is the tiling order and the order of every
/// `spans`/`owner` array in the vendored table.
pub const REGION_NAMES: [&str; 7] = ["leader", "D0", "D1", "D2", "stem", "TM", "cyt"];

/// One region's inclusive 1-based span in precursor numbering, or absence.
pub type Span = Option<(u32, u32)>;

/// The annotated regions of one precursor sequence, keyed in the table by sequence hash.
#[derive(Deserialize)]
pub struct AlleleEntry {
    /// Every allele name with this exact precursor sequence, sorted. 199 of the 892 distinct
    /// precursors are shared by more than one name; all such groups agree on gene and on every
    /// boundary (asserted by the vendoring script), so one entry serves them all.
    pub alleles: Vec<String>,
    pub gene: String,
    /// This allele's own translated CDS architecture. Differs from `gene_architecture` for the
    /// six null KIR3D alleles whose premature stop truncates translation inside D0.
    pub architecture: String,
    /// The gene-level property: the consensus over that gene's expressed, complete-CDS alleles.
    pub gene_architecture: String,
    pub expression_status: String,
    pub cds_partial: bool,
    pub cds_partial_5prime: bool,
    pub cds_partial_3prime: bool,
    /// Precursor length from the region table. Not read by `chunk` (the cleaned sequence is the
    /// authority on its own length) but retained so the vendored entry is a faithful record.
    #[allow(dead_code)]
    pub protein_length: u32,
    /// Per-region spans in `REGION_NAMES` order; `None` where the region is absent.
    pub spans: Vec<Span>,
    /// CDS-annotation fields on which the allele names sharing this protein sequence disagree.
    ///
    /// Absent when they agree, which is the usual case. Non-empty for 4 of the 199 shared-
    /// sequence groups: 3 KIR3DL3 groups mix a 5'-partial CDS with complete ones (a truncated
    /// CDS can translate to the same protein), and 1 KIR3DL1 group mixes a questionable allele
    /// with 33 sequence-identical expressed siblings. These describe the *CDS*, not the protein,
    /// so a protein-keyed entry cannot carry one value for them; the entry reports the cautious
    /// value and names the disagreement here.
    #[serde(default)]
    pub annotation_mixed: Vec<String>,
}

/// One gene's reference allele, for the alignment-transfer path.
#[derive(Deserialize)]
pub struct GeneEntry {
    /// The allele the boundaries are transferred *from*. Retained as provenance for the inferred
    /// path even though the transfer itself indexes `owner`/`sequence`.
    #[allow(dead_code)]
    pub reference_allele: String,
    pub architecture: String,
    /// Cleavage-based signal peptide length, from `kir_gene_reference.csv`. **Not** the exon 1+2
    /// end, which is 2 residues longer in all 15 genes; mature numbering counts from the
    /// cleavage site (`docs/REGIONS.md` section 4).
    pub signal_peptide_length: u32,
    /// The reference allele's precursor sequence.
    pub sequence: String,
    /// Per-position region code over `sequence`: 1..7 indexing `REGION_NAMES`, 0 for none.
    pub owner: Vec<u8>,
}

#[derive(Deserialize)]
struct RawTable {
    genes: HashMap<String, GeneEntry>,
    alleles: HashMap<String, AlleleEntry>,
}

/// The vendored table, plus the encoded reference sequences the aligner works on.
pub struct RegionTable {
    pub genes: HashMap<String, GeneEntry>,
    pub alleles: HashMap<String, AlleleEntry>,
    /// Gene -> its reference sequence encoded for `align`, so the encoding is done once at load
    /// rather than once per row.
    pub encoded: HashMap<String, Vec<u8>>,
}

impl RegionTable {
    /// The annotated entry for an exact precursor-sequence match, by sequence hash.
    pub fn exact(&self, hash: &str) -> Option<&AlleleEntry> {
        self.alleles.get(hash)
    }

    pub fn gene(&self, gene: &str) -> Option<&GeneEntry> {
        self.genes.get(gene)
    }
}

fn load_uncached(dir: &Path) -> Result<RegionTable, String> {
    let path = dir.join("regions.json");
    let text = std::fs::read_to_string(&path).map_err(|e| {
        format!(
            "cannot read the region table at {}: {e}. If you are working from a source \
             checkout, run:\n  python codegen/vendor_regions.py --kirs-root ..",
            path.display()
        )
    })?;
    let raw: RawTable = serde_json::from_str(&text)
        .map_err(|e| format!("malformed region table at {}: {e}", path.display()))?;

    if raw.genes.is_empty() {
        return Err(format!("the region table at {} lists no genes", path.display()));
    }

    // Every gene entry must be self-consistent: one region code per residue, and no code outside
    // the seven regions. A violation here would silently mis-slice every inferred sequence of
    // that gene, so it is a load error rather than something to clamp at use time.
    for (gene, entry) in &raw.genes {
        if entry.owner.len() != entry.sequence.len() {
            return Err(format!(
                "{gene}: reference sequence is {} residues but the owner map has {} entries",
                entry.sequence.len(),
                entry.owner.len()
            ));
        }
        if let Some(bad) = entry.owner.iter().find(|&&code| code as usize > REGION_NAMES.len()) {
            return Err(format!("{gene}: owner map holds out-of-range region code {bad}"));
        }
        if entry.signal_peptide_length == 0 {
            return Err(format!("{gene}: signal_peptide_length is 0"));
        }
    }

    let encoded = raw
        .genes
        .iter()
        .map(|(gene, entry)| (gene.clone(), crate::align::encode(entry.sequence.as_bytes())))
        .collect();

    Ok(RegionTable {
        genes: raw.genes,
        alleles: raw.alleles,
        encoded,
    })
}

type Cache = Mutex<HashMap<PathBuf, Arc<RegionTable>>>;

fn cache() -> &'static Cache {
    static CACHE: OnceLock<Cache> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Load a region directory, or return the already-loaded copy. Keyed on the canonical path.
pub fn load(dir: &Path) -> Result<Arc<RegionTable>, String> {
    let key = dir.canonicalize().unwrap_or_else(|_| dir.to_path_buf());

    if let Some(hit) = cache().lock().expect("region cache mutex poisoned").get(&key) {
        return Ok(Arc::clone(hit));
    }

    // Parsed outside the lock, as in `models::load`: two threads racing a cold directory
    // duplicate the work once rather than making every reader queue behind the first.
    let table = Arc::new(load_uncached(dir)?);

    let mut guard = cache().lock().expect("region cache mutex poisoned");
    Ok(Arc::clone(guard.entry(key).or_insert(table)))
}
