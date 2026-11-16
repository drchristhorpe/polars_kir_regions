//! `polars_kir_regions._internal` -- the native half of the plugin.
//!
//! Polars discovers the expressions in `expressions.rs` by loading this shared library directly;
//! the `#[pymodule]` below exists so that Python can import the module and hand Polars its path.
//!
//! **Standalone by design.** This crate has no HMM, no trained models and no classifier. The gene
//! is supplied by the caller's dataframe, so there is nothing to infer -- and that is the whole
//! reason this is a separate package rather than two more expressions inside `polars_kir_hmm`.
//! The only thing the two share is a build-time data source (`histo_kir`'s Phase 0 outputs) and a
//! house layout; at runtime neither imports the other.

mod align;
mod chunk;
mod expressions;
mod loops;
mod loops_expr;
mod regions;
mod sequence;

use pyo3::prelude::*;

#[pymodule]
fn _internal(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
