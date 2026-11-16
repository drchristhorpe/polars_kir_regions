"""Polars expressions slicing KIR ectodomain regions and HLA-contacting loops.

Registers a ``.kir_regions`` namespace with two expressions, :meth:`KIRRegionsNameSpace.chunk`
and :meth:`KIRRegionsNameSpace.loops`.

**The gene comes from your dataframe.** Both expressions take it as an input; neither infers it.
That is what makes this package standalone -- gene calling is the only part of the problem that
needs a trained model, and it lives in ``polars_kir_hmm``. If you do not know the gene, type the
sequence there (or anywhere else) and pass the answer in::

    import polars as pl
    import polars_kir_regions  # noqa: F401  -- registers the .kir_regions namespace

    df = pl.DataFrame({"gene": ["KIR3DL1"], "sequence": [seq]})
    df.with_columns(r=pl.col("sequence").kir_regions.chunk(gene="gene")).unnest("r")

Nothing in this package imports ``polars_kir_hmm``, ``histo_kir_hmm`` or ``histo_kir`` at
runtime. The region table and loop windows are vendored at build time from ``histo_kir``'s
Phase 0 outputs; the wheel carries the JSON.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl
from polars.plugins import register_plugin_function

if TYPE_CHECKING:
    from polars._typing import IntoExpr

__all__ = [
    "KIRRegionsNameSpace",
    "chunk",
    "genes",
    "loop_dir",
    "loop_windows",
    "loops",
    "region_dir",
    "__version__",
]
__version__ = "0.1.0"

_PLUGIN_PATH = Path(__file__).parent
_REGION_DIR = _PLUGIN_PATH / "regions"
_LOOP_DIR = _PLUGIN_PATH / "loops"


def region_dir() -> Path:
    """Path to the bundled region table, vendored from ``histo_kir``'s Phase 0 output."""
    return _REGION_DIR


def loop_dir() -> Path:
    """Path to the bundled loop windows, vendored from ``histo_kir``'s loop definitions."""
    return _LOOP_DIR


def _resolve(directory: str | Path | None, default: Path, filename: str, argument: str) -> str:
    """Resolve a data directory, failing here rather than inside the query engine."""
    if directory is None:
        if not (default / filename).exists():
            raise RuntimeError(
                f"the bundled {filename} is missing from {default}. If you are working from a "
                "source checkout, run:\n"
                "  python codegen/vendor_regions.py --kirs-root ..\n"
                "  python codegen/vendor_loops.py --kirs-root .."
            )
        return str(default)

    path = Path(directory)
    if not path.is_dir():
        raise ValueError(f"{argument} {str(directory)!r} is not a directory")
    if not (path / filename).exists():
        raise ValueError(f"no {filename} in {str(directory)!r}")
    return str(path)


def _gene_expr(gene: IntoExpr) -> pl.Expr:
    """Accept a column name, a literal gene, or an expression.

    A bare string is ambiguous -- ``"KIR3DL1"`` is a gene, ``"gene"`` is a column -- so it is
    resolved as a column name when the frame has such a column and as a literal otherwise. That
    resolution happens in the query engine, not here, so it works on a LazyFrame too: the
    ambiguity is settled by trying the column and falling back to the literal.
    """
    if isinstance(gene, pl.Expr):
        return gene
    if isinstance(gene, str):
        # A gene name always contains "KIR"; a column name identifying genes almost never does.
        # Prefer the literal reading only when the string is unmistakably a gene.
        if gene.upper().startswith("KIR"):
            return pl.lit(gene, dtype=pl.String)
        return pl.col(gene)
    return pl.lit(gene, dtype=pl.String)


@pl.api.register_expr_namespace("kir_regions")
class KIRRegionsNameSpace:
    """The ``.kir_regions`` expression namespace."""

    def __init__(self, expr: pl.Expr) -> None:
        self._expr = expr

    def chunk(
        self,
        gene: IntoExpr,
        region_dir: str | Path | None = None,
    ) -> pl.Expr:
        """Slice the pMHC-interacting ectodomain out of a KIR sequence column.

        ``gene`` is required: a column name (``gene="gene"``), a literal gene
        (``gene="KIR3DL1"``), or any expression yielding a String column. A bare string starting
        with ``KIR`` is read as a literal, anything else as a column name.

        Returns a struct column. **Scope: the ectodomain only** -- the leader, the signal peptide,
        the leader's mature tail, the three Ig domains and their concatenation. The stem,
        transmembrane helix and cytoplasmic tail are deliberately not emitted; the question this
        serves is KIR/pMHC recognition, and those regions do not participate in it. They remain in
        ``histo_kir/data/kir_allele_regions.parquet``, which is unchanged.

        ========================== ================== =========================================
        field                      dtype              meaning
        ========================== ================== =========================================
        ``leader``                 ``String``         exon 1+2 product, **2 residues longer**
                                                      than ``signal_peptide``
        ``signal_peptide``         ``String``         the cleaved signal peptide (21 aa; 23 for
                                                      KIR2DL4). Null on a 5'-partial CDS
        ``mature_leader_tail``     ``String``         mature residues encoded by the leader exon
                                                      -- normally mature 1-2. Null when the
                                                      cleavage site is not in the sequence
        ``D0``                     ``String``         exon 3. **Null** for the 8 D1D2 genes
        ``D1``                     ``String``         exon 4. **Null** for the 3 D0D2 genes
        ``D2``                     ``String``         exon 5
        ``ectodomain``             ``String``         ``mature_leader_tail + D0 + D1 + D2``:
                                                      mature residue 1 to the D2 end.
                                                      **Excludes the extracellular stem, by
                                                      scope decision** -- see below
        ``gene``                   ``String``         the gene used, echoed back
        ``allele``                 ``String``         matched allele; null when >1 name shares
                                                      the sequence, or nothing matched
        ``alleles``                ``List(String)``   every allele name with this exact sequence
        ``architecture``           ``String``         this allele's own translated CDS
        ``gene_architecture``      ``String``         the gene-level property
        ``expression_status``      ``String``         expressed / null / questionable
        ``cds_partial``            ``Boolean``        annotation partial marker
        ``cds_partial_5prime``     ``Boolean``
        ``cds_partial_3prime``     ``Boolean``
        ``provenance``             ``String``         ``annotated`` / ``inferred`` / ``none``
        ``identity``               ``Float64``        alignment identity, inferred path only
        ``n_mapped``               ``UInt32``         query residues the alignment mapped
        ``length``                 ``UInt32``         length of the cleaned sequence
        ``ectodomain_ok``          ``Boolean``        both ectodomain identities held on this row
        ``notes``                  ``List(String)``   partial CDS, alignment gaps, disagreements
        ========================== ================== =========================================

        Five things about the output are contract, not implementation detail:

        * **The emitted fields do NOT tile the precursor.** They stop at the D2 end, so
          concatenating them gives a sequence short by the stem, TM and cytoplasmic regions --
          median 127 residues. For a full-length partition read
          ``kir_allele_regions.parquet``, which does tile (verified 1983/1983 alleles).
        * **``ectodomain`` is D0+D1+D2 (plus the leader's mature tail) and therefore excludes the
          stem**, which is genuinely extracellular. That is a scope decision, not an error. It
          also excludes the 5-7 exon-7 residues N-terminal to the TM helix. For the full
          extracellular extent use ``histo_kir``'s ``ectodomain_end``.
        * **``leader`` and ``signal_peptide`` differ by 2 residues** -- 23 versus 21 in 14 genes,
          25 versus 23 in KIR2DL4. Mature numbering counts from the *cleavage site*, so anything
          working in the mature frame (the loop windows, for instance) must use
          ``signal_peptide``. Using ``leader`` shifts every mature position by 2. The two
          residues between them are emitted as ``mature_leader_tail`` so the offset is visible in
          the data rather than implicit in a document.
        * **Absent regions are null, never the empty string.** An empty string would claim the
          region exists and is zero residues long, which is false. Note that with the stem out of
          scope, KIR3DL3 has no missing region -- its D2 runs to the ectodomain end.
        * **An unknown or null gene gives a null row**, with ``provenance = "none"`` and a note.
          Nothing is guessed.

        ``provenance`` tells a read boundary from an inferred one:

        * ``annotated`` -- the cleaned sequence is byte-identical to a known IPD-KIR precursor, so
          the boundaries are that allele's own annotation. The gene is taken from the match rather
          than from your column, and a disagreement is noted: a byte-identical precursor is
          stronger evidence than a label.
        * ``inferred`` -- it is not, so the boundaries are transferred from your gene's reference
          allele by gapped alignment. Materially less certain: 277 of the 1983 alleles differ in
          length from their gene's reference.
        * ``none`` -- no usable gene, so every region is null.

        Regions index the **cleaned** sequence -- uppercased and stripped of everything that is
        not one of the 20 amino acids or ``X``. Null inputs give null structs.

        .. note::
           The struct emits its own ``gene``, so ``unnest`` beside an input column of that name
           raises ``DuplicateError``. Either name the input column something else, or drop it::

               df.with_columns(r=chunk_expr).drop("gene").unnest("r")

           This is a name collision in your frame, not an error in the expression: the emitted
           ``gene`` is not always the one you passed -- on an exact match it is the matched
           allele's, which is the whole point of the annotated path.
        """
        return register_plugin_function(
            plugin_path=_PLUGIN_PATH,
            function_name="chunk_expr",
            args=[self._expr, _gene_expr(gene)],
            kwargs={"region_dir": _resolve(region_dir, _REGION_DIR, "regions.json", "region_dir")},
            is_elementwise=True,
        )

    def loops(
        self,
        gene: IntoExpr,
        loop_dir: str | Path | None = None,
    ) -> pl.Expr:
        """Slice the HLA-contacting loops out of a KIR sequence column.

        ``gene`` is required, exactly as for :meth:`chunk`.

        Returns a struct with one named field per element plus ``elements`` as a list, so
        ``pl.col("loops").struct.field("FG").struct.field("sequence")`` works directly. Windows
        are in **mature numbering counted from the signal-peptide cleavage site** -- the offset
        that makes this correct is the same one ``chunk`` exposes as ``mature_leader_tail``.

        Every window is transferred from the gene reference by alignment, so ``provenance`` is
        ``inferred`` for any row that produced a transfer and null where none did. There is no
        exact-match shortcut: a window is defined in the reference's mature frame rather than
        stored per allele.

        A ``;``-separated gene value (how ``histo_kir`` serialises a tie set) yields a null row
        with a note. This package does not arbitrate ties -- choose a gene, or slice once per
        candidate.

        .. note::
           As with :meth:`chunk`, the struct emits its own ``gene``, so ``unnest`` beside an input
           column of that name raises ``DuplicateError``. Rename or drop the input column.
        """
        return register_plugin_function(
            plugin_path=_PLUGIN_PATH,
            function_name="loops_expr",
            args=[self._expr, _gene_expr(gene)],
            kwargs={"loop_dir": _resolve(loop_dir, _LOOP_DIR, "loop_windows.json", "loop_dir")},
            is_elementwise=True,
        )


def genes(region_dir: str | Path | None = None) -> pl.DataFrame:
    """The 15 genes in the bundled region table, with their architecture and reference allele."""
    import json

    directory = Path(_resolve(region_dir, _REGION_DIR, "regions.json", "region_dir"))
    payload = json.loads((directory / "regions.json").read_text())
    rows = [
        {
            "gene": gene,
            "architecture": entry["architecture"],
            "reference_allele": entry["reference_allele"],
            "signal_peptide_length": entry["signal_peptide_length"],
            "precursor_length": len(entry["sequence"]),
        }
        for gene, entry in sorted(payload["genes"].items())
    ]
    return pl.DataFrame(rows)


def loop_windows(loop_dir: str | Path | None = None) -> pl.DataFrame:
    """The vendored per-gene loop windows, as a DataFrame.

    One row per gene per element: ``gene``, ``architecture``, ``reference_allele``, ``element``,
    ``kind``, ``domain``, ``hla_helix``, ``mature_start``, ``mature_end``, ``length``, ``sequence``
    and ``predicted_only``.

    ``kind`` separates the evidence classes, which are not equivalent and should not be pooled:
    ``published_loop`` is Boyington & Sun (2001) Fig. 5 unchanged, ``d0_patch`` is the KIR3DL1 D0
    contact patch, and ``ccprime_extension`` the single-residue CC' extension. The latter two are
    recorded only for the four three-domain genes.

    ``predicted_only`` is True for the eight genes with no deposited structure anywhere, whose
    windows are alignment predictions rather than observations. Any interface-coverage statistic
    computed across all 15 genes should exclude or flag those rows.
    """
    import json

    directory = Path(_resolve(loop_dir, _LOOP_DIR, "loop_windows.json", "loop_dir"))
    payload = json.loads((directory / "loop_windows.json").read_text())

    # The genes with at least one deposited chain, from the structural corpus this package is
    # validated against. Hard-coded rather than derived because the wheel does not ship the
    # structure files; `tools/loops_structural_report.py` re-derives and checks it.
    with_structure = {
        "KIR2DL1", "KIR2DL2", "KIR2DL3", "KIR2DL4", "KIR2DS2", "KIR2DS4", "KIR3DL1",
    }
    rows = [
        {
            "gene": gene,
            "architecture": entry["architecture"],
            "reference_allele": entry["reference_allele"],
            "element": element["name"],
            "kind": element["kind"],
            "domain": element["domain"],
            "hla_helix": element["helix"],
            "mature_start": element["mature_start"],
            "mature_end": element["mature_end"],
            "length": element["mature_end"] - element["mature_start"] + 1,
            "sequence": element["sequence"],
            "predicted_only": gene not in with_structure,
        }
        for gene, entry in sorted(payload["genes"].items())
        for element in entry["elements"]
    ]
    return pl.DataFrame(
        rows,
        schema={
            "gene": pl.String,
            "architecture": pl.String,
            "reference_allele": pl.String,
            "element": pl.String,
            "kind": pl.String,
            "domain": pl.String,
            "hla_helix": pl.String,
            "mature_start": pl.UInt32,
            "mature_end": pl.UInt32,
            "length": pl.UInt32,
            "sequence": pl.String,
            "predicted_only": pl.Boolean,
        },
    )


def chunk(expr: IntoExpr, /, gene: IntoExpr, **kwargs) -> pl.Expr:
    """Function form of :meth:`KIRRegionsNameSpace.chunk`, for when you prefer it."""
    e = pl.col(expr) if isinstance(expr, str) else expr
    return e.kir_regions.chunk(gene, **kwargs)


def loops(expr: IntoExpr, /, gene: IntoExpr, **kwargs) -> pl.Expr:
    """Function form of :meth:`KIRRegionsNameSpace.loops`, for when you prefer it."""
    e = pl.col(expr) if isinstance(expr, str) else expr
    return e.kir_regions.loops(gene, **kwargs)
