"""Per-gene summary figure for the `loops` expression.

Three panels, each drawn from a measurement made in this run -- nothing is hardcoded:

  a. the nine element windows per gene on the mature axis, with the D0/D1/D2 extents behind them;
     shows at a glance that every element sits inside the ectodomain, that the three D0D2 genes
     lack the D1 loops, and that D0a/D0b/CC'ext appear only for the four three-domain genes.
  b. the frame result per gene: windows reproduced from the cleavage site vs from the exon leader.
  c. how much deposited structural support each gene actually has, since 8 of 15 have none and
     their windows are predictions rather than observations.
"""
from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import polars as pl  # noqa: E402

import polars_kir_regions  # noqa: E402,F401
from polars_kir_regions import loop_windows  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
KIRS = ROOT.parent


def apply_figure_style(sizes=(8, 7, 6)) -> None:
    """Three-step role-mapped size ladder, minimal frame, no grid."""
    base, mid, small = sizes
    mpl.rcParams.update({
        "figure.dpi": 110,
        "savefig.dpi": 300,
        "font.size": base,
        "axes.titlesize": base,
        "axes.labelsize": base,
        "xtick.labelsize": mid,
        "ytick.labelsize": mid,
        "legend.fontsize": small,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "axes.titlepad": 7.0,
        "legend.frameon": False,
    })


def panel_letter(ax, letter: str) -> None:
    ax.text(-0.055, 1.045, letter, transform=ax.transAxes, fontsize=11,
            fontweight="bold", va="bottom", ha="right")


apply_figure_style(sizes=(8, 7, 6))

# --- data, all measured here ------------------------------------------------------------------
windows = loop_windows()
table = json.loads((ROOT / "python/polars_kir_regions/loops/loop_windows.json").read_text())
alleles = {a["allele"]: a for a in
           json.loads((ROOT / "tests/data/ipdkir_alleles.json").read_text())["alleles"]}
chains = json.loads((ROOT / "tests/data/structural_chains.json").read_text())["kir_chains"]
regions = pl.read_parquet(KIRS / "histo_kir/data/kir_allele_regions.parquet")
region_row = {r["allele"]: r for r in regions.iter_rows(named=True)}

genes = sorted(table["genes"])
per_gene_chains = Counter(c["gene"] for c in chains)

kind_colour = {"published_loop": "#2a6f97", "d0_patch": "#c05746", "ccprime_extension": "#7d6b91"}
domain_colour = {"D0": "#dde5ec", "D1": "#e8e1ef", "D2": "#e2ece4"}

fig = plt.figure(figsize=(13.0, 8.0))
# Explicit margins rather than bbox_inches="tight": the geometry checked below must be the
# geometry saved, and "tight" re-lays-out the figure after the renderer has been queried.
grid = fig.add_gridspec(2, 2, width_ratios=[1.62, 1.0], height_ratios=[1.0, 1.0],
                        hspace=0.52, wspace=0.30,
                        left=0.075, right=0.985, top=0.945, bottom=0.165)
ax_a = fig.add_subplot(grid[:, 0])
ax_b = fig.add_subplot(grid[0, 1])
ax_c = fig.add_subplot(grid[1, 1])

# --- panel a: the windows on the mature axis ----------------------------------------------------
for y, gene in enumerate(genes):
    entry = table["genes"][gene]
    signal = entry["signal_peptide_length"]
    row = region_row[entry["reference_allele"]]
    for domain in ("D0", "D1", "D2"):
        start, end = row[f"{domain}_start"], row[f"{domain}_end"]
        if start is None:
            continue
        ax_a.barh(y, (end - signal) - (start - signal) + 1, left=start - signal,
                  height=0.68, color=domain_colour[domain], zorder=1)
        if gene == "KIR3DL1":
            # Label the extents once, above the topmost 3D row, clear of the element marks.
            ax_a.text((start + end) / 2 - signal, y - 0.52, domain, ha="center", va="top",
                      fontsize=6.5, color="#3a3a3a")
    for element in entry["elements"]:
        ax_a.barh(y, element["mature_end"] - element["mature_start"] + 1,
                  left=element["mature_start"], height=0.42,
                  color=kind_colour[element["kind"]], zorder=3)

ax_a.set_yticks(range(len(genes)))
ax_a.set_yticklabels(
    [f"{g}  ({table['genes'][g]['architecture']})" for g in genes], fontsize=6.5
)
ax_a.invert_yaxis()
ax_a.set_xlabel("mature residue (1 = signal-peptide cleavage site)")
ax_a.set_title("All nine contact elements lie inside D0+D1+D2", loc="left")
handles = [
    mpl.patches.Patch(color=kind_colour["published_loop"], label="published loop (6)"),
    mpl.patches.Patch(color=kind_colour["d0_patch"], label="D0 contact patch (D0a, D0b)"),
    mpl.patches.Patch(color=kind_colour["ccprime_extension"], label="CC' extension"),
    mpl.patches.Patch(color=domain_colour["D0"], label="D0 extent"),
    mpl.patches.Patch(color=domain_colour["D1"], label="D1 extent"),
    mpl.patches.Patch(color=domain_colour["D2"], label="D2 extent"),
]
ax_a.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.105),
            ncol=3, frameon=False, fontsize=6.5)
panel_letter(ax_a, "a")

# --- panel b: the frame -------------------------------------------------------------------------
cleavage, leader, possible = [], [], []
for gene in genes:
    entry = table["genes"][gene]
    allele = entry["reference_allele"]
    precursor = alleles[allele]["sequence"]
    signal = entry["signal_peptide_length"]
    exon_leader = region_row[allele]["leader_end_exon_frame"]
    published = [e for e in entry["elements"] if e["kind"] == "published_loop"]
    cleavage.append(sum(
        precursor[signal + e["mature_start"] - 1 : signal + e["mature_end"]] == e["sequence"]
        for e in published
    ))
    leader.append(sum(
        precursor[exon_leader + e["mature_start"] - 1 : exon_leader + e["mature_end"]]
        == e["sequence"] for e in published
    ))
    possible.append(len(published))

ys = range(len(genes))
ax_b.barh([y - 0.19 for y in ys], cleavage, height=0.36, color="#2a6f97",
          label="from the cleavage site")
ax_b.barh([y + 0.19 for y in ys], leader, height=0.36, color="#c9ada7",
          label="from the exon-leader end")
ax_b.set_yticks(list(ys))
ax_b.set_yticklabels(genes, fontsize=6)
ax_b.invert_yaxis()
ax_b.set_xlabel("published windows reproduced exactly (of 6, or 3 for D0D2)")
ax_b.set_title(f"The frame is the cleavage site: {sum(cleavage)}/{sum(possible)} "
               f"vs {sum(leader)}/{sum(possible)}", loc="left")
ax_b.legend(loc="upper center", bbox_to_anchor=(0.5, -0.20), ncol=2, frameon=False,
            fontsize=6.5)
ax_b.margins(x=0.06)
panel_letter(ax_b, "b")

# --- panel c: structural support -----------------------------------------------------------------
support_genes = [g for g in genes if per_gene_chains[g]]
frame = pl.DataFrame({
    "file": [c["file"] for c in chains],
    "gene": [c["gene"] for c in chains],
    "sequence": [c["sequence"] for c in chains],
}).with_columns(loops=pl.col("sequence").kir_regions.loops(gene="gene"))

complete, expected = [], []
for gene in support_genes:
    rows = frame.filter(pl.col("gene") == gene)
    n_complete = n_expected = 0
    for row in rows.iter_rows(named=True):
        for element in row["loops"]["elements"] or []:
            n_expected += 1
            n_complete += element["status"] == "complete"
    complete.append(n_complete)
    expected.append(n_expected)

ys_c = range(len(support_genes))
ax_c.barh(list(ys_c), expected, height=0.6, color="#e6e6e6", label="windows expected")
ax_c.barh(list(ys_c), complete, height=0.6, color="#2a6f97", label="observed complete")
for y, (done, total, gene) in enumerate(zip(complete, expected, support_genes)):
    n = per_gene_chains[gene]
    ax_c.text(total + 0.6, y, f"{done}/{total}  ({n} chain{'s' if n != 1 else ''})",
              va="center", fontsize=6, color="#4a4a4a")
ax_c.set_yticks(list(ys_c))
ax_c.set_yticklabels(support_genes, fontsize=6)
ax_c.invert_yaxis()
ax_c.set_ylim(len(support_genes) - 0.35, -0.65)  # headroom so value labels clear the axis
ax_c.set_xlabel("element windows across the deposited chains of that gene")
ax_c.set_title(f"Structural support is uneven: {len(support_genes)} of {len(genes)} genes have any",
               loc="left")
ax_c.legend(loc="upper center", bbox_to_anchor=(0.5, -0.20), ncol=2, frameon=False,
            fontsize=6.5)
ax_c.set_xlim(0, max(expected) * 1.55)
panel_letter(ax_c, "c")

fig.savefig(ROOT / "tools/loops_verification.png", dpi=300)

# --- render-then-verify: geometric check ------------------------------------------------------
# Draw first: axis labels are auto-positioned during draw, and querying their extent before it
# returns a stale box that reports overlaps which are not in the saved image.
fig.canvas.draw()
renderer = fig.canvas.get_renderer()
texts = [(t, t.get_window_extent(renderer)) for t in fig.findobj(mpl.text.Text)
         if t.get_text().strip() and t.get_visible()]
overlaps = [(a.get_text(), b.get_text())
            for i, (a, ba) in enumerate(texts) for b, bb in texts[i + 1:] if ba.overlaps(bb)]

print(f"genes with any deposited chain: {len(support_genes)} of {len(genes)}")
print(f"cleavage-site total {sum(cleavage)} | exon-leader total {sum(leader)} "
      f"| of {sum(possible)}")
print(f"complete transfers {sum(complete)} of {sum(expected)} expected windows")
print(f"chains per gene: {dict(sorted(per_gene_chains.items()))}")
print(f"text overlaps: {len(overlaps)}")
for pair in overlaps[:12]:
    print("   ", pair)
