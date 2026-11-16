//! The `chunk` Polars expression.
//!
//! `loops` lives in `loops_expr.rs`, so the two tracks do not have to edit the same file.
//!
//! **The gene is a required input.** It comes from a second input column (or a literal). This
//! package has no classifier, so there is nothing here to infer a gene with -- which is exactly
//! why it can stand alone: gene calling was the only thing in the original implementation that
//! needed a trained model.

use std::path::Path;

use polars::prelude::*;
use pyo3_polars::derive::polars_expr;
use serde::Deserialize;

use crate::chunk::{self, Chunks};
use crate::regions;

#[derive(Deserialize)]
pub struct ChunkKwargs {
    /// Resolved to a concrete directory by Python, which also checks `regions.json` is there.
    region_dir: String,
}

/// The struct `chunk` returns.
///
/// **Ectodomain scope.** The emitted fields are the leader, the signal peptide, the leader's
/// mature tail, the three Ig domains and their concatenation. Stem, TM and cytoplasmic are
/// deliberately absent (see `chunk.rs`), so **these fields do not tile the precursor** and
/// concatenating them yields a sequence that stops at the D2 end.
fn chunk_output(input_fields: &[Field]) -> PolarsResult<Field> {
    let string = |name: &'static str| Field::new(PlSmallStr::from_static(name), DataType::String);
    let boolean = |name: &'static str| Field::new(PlSmallStr::from_static(name), DataType::Boolean);
    let dt = DataType::Struct(vec![
        string("leader"),
        string("signal_peptide"),
        string("mature_leader_tail"),
        string("D0"),
        string("D1"),
        string("D2"),
        string("ectodomain"),
        string("gene"),
        string("allele"),
        Field::new(
            PlSmallStr::from_static("alleles"),
            DataType::List(Box::new(DataType::String)),
        ),
        string("architecture"),
        string("gene_architecture"),
        string("expression_status"),
        boolean("cds_partial"),
        boolean("cds_partial_5prime"),
        boolean("cds_partial_3prime"),
        string("provenance"),
        Field::new(PlSmallStr::from_static("identity"), DataType::Float64),
        Field::new(PlSmallStr::from_static("n_mapped"), DataType::UInt32),
        Field::new(PlSmallStr::from_static("length"), DataType::UInt32),
        boolean("ectodomain_ok"),
        Field::new(
            PlSmallStr::from_static("notes"),
            DataType::List(Box::new(DataType::String)),
        ),
    ]);
    Ok(Field::new(input_fields[0].name().clone(), dt))
}

#[polars_expr(output_type_func=chunk_output)]
fn chunk_expr(inputs: &[Series], kwargs: ChunkKwargs) -> PolarsResult<Series> {
    let ca = inputs[0].str()?;
    let table =
        regions::load(Path::new(&kwargs.region_dir)).map_err(|e| polars_err!(ComputeError: "{e}"))?;
    let n = ca.len();

    // A missing gene column is a usage error, not a per-row null: a caller who forgot it would
    // otherwise get a frame of silently-null rows and read that as "no regions found".
    if inputs.len() < 2 {
        polars_bail!(
            ComputeError:
            "chunk() needs a gene: pass a column name or a literal, e.g. \
             pl.col('sequence').kir_regions.chunk(gene='gene'). This package has no classifier -- \
             the gene comes from your dataframe."
        );
    }
    let gene_ca = inputs[1].str()?;
    // A literal gene arrives as a length-1 column while the sequence column has n rows, so the
    // index has to be clamped: reading `gene_ca.get(row)` directly panics past row 0. Polars does
    // not broadcast a literal for a plugin the way it does for a builtin expression.
    let gene_len = gene_ca.len();
    if gene_len != 1 && gene_len != n {
        polars_bail!(
            ComputeError:
            "the gene column has {gene_len} rows but the sequence column has {n}; pass a column \
             of the same height or a single literal"
        );
    }
    let gene_at = |row: usize| gene_ca.get(if gene_len == 1 { 0 } else { row });

    // Rows are not parallelised: the annotated path is a hash lookup, and the inferred path costs
    // one O(n*m) alignment per row. Neither is the 15-model scan that made a fan-out worthwhile in
    // the HMM package, and `rayon` is not a dependency of this crate as a result.
    let results: Vec<Option<Chunks>> = (0..n)
        .map(|row| {
            ca.get(row)
                .map(|sequence| chunk::chunk_one(sequence, gene_at(row), &table))
        })
        .collect();

    let region_col = |index: usize, name: &'static str| {
        StringChunked::from_iter_options(
            PlSmallStr::from_static(name),
            results
                .iter()
                .map(|r| r.as_ref().and_then(|c| c.regions[index].as_deref())),
        )
        .into_series()
    };
    let str_col = |name: &'static str, get: fn(&Chunks) -> Option<&str>| {
        StringChunked::from_iter_options(
            PlSmallStr::from_static(name),
            results.iter().map(|r| r.as_ref().and_then(get)),
        )
        .into_series()
    };
    let bool_col = |name: &'static str, get: fn(&Chunks) -> bool| {
        BooleanChunked::from_iter_options(
            PlSmallStr::from_static(name),
            results.iter().map(|r| r.as_ref().map(get)),
        )
        .into_series()
    };

    let identity = Float64Chunked::from_iter_options(
        PlSmallStr::from_static("identity"),
        results.iter().map(|r| r.as_ref().and_then(|c| c.identity)),
    )
    .into_series();
    let n_mapped = UInt32Chunked::from_iter_options(
        PlSmallStr::from_static("n_mapped"),
        results.iter().map(|r| r.as_ref().and_then(|c| c.n_mapped)),
    )
    .into_series();
    let length = UInt32Chunked::from_iter_options(
        PlSmallStr::from_static("length"),
        results.iter().map(|r| r.as_ref().map(|c| c.length)),
    )
    .into_series();

    // Both ectodomain identities, checked per row inside `chunk_one` and surfaced here so a caller
    // can assert on the column rather than re-deriving the check.
    let ectodomain_ok = bool_col("ectodomain_ok", |c| c.ectodomain_ok);

    let alleles =
        string_list(&results, |c| c.alleles.iter().map(|s| s.as_str()).collect(), "alleles")?;
    let notes = string_list(&results, |c| c.notes.iter().map(|s| s.as_str()).collect(), "notes")?;

    let fields = vec![
        region_col(0, "leader"),
        str_col("signal_peptide", |c| c.signal_peptide.as_deref()),
        str_col("mature_leader_tail", |c| c.mature_leader_tail.as_deref()),
        region_col(1, "D0"),
        region_col(2, "D1"),
        region_col(3, "D2"),
        str_col("ectodomain", |c| c.ectodomain.as_deref()),
        str_col("gene", |c| c.gene.as_deref()),
        str_col("allele", |c| c.allele.as_deref()),
        alleles,
        str_col("architecture", |c| c.architecture.as_deref()),
        str_col("gene_architecture", |c| c.gene_architecture.as_deref()),
        str_col("expression_status", |c| c.expression_status.as_deref()),
        bool_col("cds_partial", |c| c.cds_partial),
        bool_col("cds_partial_5prime", |c| c.cds_partial_5prime),
        bool_col("cds_partial_3prime", |c| c.cds_partial_3prime),
        str_col("provenance", |c| Some(c.provenance)),
        identity,
        n_mapped,
        length,
        ectodomain_ok,
        notes,
    ];

    let out = StructChunked::from_series(ca.name().clone(), n, fields.iter())?;
    Ok(out.into_series())
}

/// Build a `List(String)` column from a per-row string vector.
fn string_list(
    results: &[Option<Chunks>],
    extract: fn(&Chunks) -> Vec<&str>,
    name: &'static str,
) -> PolarsResult<Series> {
    let rows: Vec<Option<Series>> = results
        .iter()
        .map(|r| {
            r.as_ref().map(|c| {
                StringChunked::from_iter_values(PlSmallStr::from_static(name), extract(c).into_iter())
                    .into_series()
            })
        })
        .collect();
    let ca: ListChunked = rows.into_iter().collect();
    let series = ca.into_series().with_name(PlSmallStr::from_static(name));
    series.cast(&DataType::List(Box::new(DataType::String)))
}
