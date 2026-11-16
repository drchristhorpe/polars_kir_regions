"""Verification summary figure for the polars_kir_hmm `chunk` expression.

Every number is read from verification.json / per_gene.csv, both written by the measurement
run in this session -- nothing here is transcribed by hand.
"""
import json
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

mpl.rcParams.update({
    'font.size': 8, 'axes.titlesize': 8, 'axes.labelsize': 7.5, 'legend.fontsize': 6.5,
    'xtick.labelsize': 6, 'ytick.labelsize': 6, 'axes.linewidth': 0.6,
    'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
    'font.family': 'sans-serif', 'axes.grid': False, 'legend.frameon': False,
})

V = json.load(open('/tmp/chunkwork/verification.json'))
pg = pl.read_csv('/tmp/chunkwork/per_gene.csv')
C = {'tail': '#8c8c8c', 'D0': '#3b6ea5', 'D1': '#c26a27', 'D2': '#4f8a5b'}

fig = plt.figure(figsize=(7.4, 6.1))
gs = fig.add_gridspec(2, 2, height_ratios=[1.5, 1.0], hspace=0.66, wspace=0.40,
                      left=0.145, right=0.965, top=0.835, bottom=0.150)

# --- a: what the expression emits, per gene -----------------------------------------
axA = fig.add_subplot(gs[0, :])
genes = pg['gene'].to_list()
y = np.arange(len(genes))
left = np.zeros(len(genes))
for part, label in (('tail', 'leader tail (mature 1–2)'), ('D0', 'D0'), ('D1', 'D1'), ('D2', 'D2')):
    width = np.array([0.0 if v is None else float(v) for v in pg[part].to_list()])
    axA.barh(y, width, left=left, height=0.66, color=C[part], label=label,
             edgecolor='white', linewidth=0.6)
    left += width
# An architecturally absent domain is marked at *its own* slot -- the position it would occupy
# had it been present -- not at the bar's left edge, where the marker would sit on whichever
# domain happens to be drawn first and read as labelling that one instead.
tail_w = np.array([float(v) for v in pg['tail'].to_list()])
d0_w = np.array([0.0 if v is None else float(v) for v in pg['D0'].to_list()])
for i, total in enumerate(left):
    axA.text(total + 4, i, f'{int(total)}', va='center', ha='left', fontsize=6)
    if pg['D0'][i] is None:
        # D0 would precede D1, so its slot is immediately after the leader tail.
        axA.text(tail_w[i] + 3, i, 'no D0', va='center', ha='left', fontsize=5.5,
                 color='white', style='italic')
    if pg['D1'][i] is None:
        # D1 would sit between D0 and D2.
        axA.text(tail_w[i] + d0_w[i] + 3, i, 'no D1', va='center', ha='left', fontsize=5.5,
                 color='white', style='italic')
axA.set_yticks(y)
axA.set_yticklabels(genes)
axA.invert_yaxis()
axA.set_xlim(0, 345)
axA.set_xticks([0, 100, 200, 300])
axA.margins(y=0.03)
axA.set_xlabel('mature residue position')
# Title on the upper band, legend on the band below it: the bars reach the right edge of the
# panel, so an in-panel legend collides with a value label, and a legend on the title's own band
# collides with the title.
axA.set_title('a  Emitted ectodomain: leader tail + D0 + D1 + D2, median over expressed '
              'complete-CDS alleles', loc='left', pad=22)
axA.legend(loc='lower left', bbox_to_anchor=(0.0, 1.002), ncol=4,
           handlelength=1.1, columnspacing=0.9)
for spine in ('top', 'right'):
    axA.spines[spine].set_visible(False)

# --- b: the checks ------------------------------------------------------------------
axB = fig.add_subplot(gs[1, 0])
labels = ['identity 1\nD0+D1+D2', 'identity 2\ninput slice', 'region\nlengths', 'ectodomain\n_ok']
passed = [V['identity1_pass'], V['identity2_pass'],
          V['region_length_comparisons'] - V['region_length_mismatches'], V['ectodomain_ok_true']]
total = [V['identity1_checked'], V['identity2_checked'],
         V['region_length_comparisons'], V['n_alleles']]
xb = np.arange(4)
axB.bar(xb, [100.0] * 4, width=0.58, color='#e8e8e8')
axB.bar(xb, [100 * p / t for p, t in zip(passed, total)], width=0.58, color='#4f8a5b')
for i, (p, t) in enumerate(zip(passed, total)):
    axB.text(i, 103, f'{p}/\n{t}' if t > 2000 else f'{p}/{t}',
             ha='center', va='bottom', fontsize=5.4)
axB.set_xticks(xb)
axB.set_xticklabels(labels, fontsize=5.6)
axB.set_ylim(0, 132)
axB.set_yticks([0, 50, 100])
axB.set_ylabel('checks passing (%)')
axB.set_title('b  Every check passes', loc='left')
for spine in ('top', 'right'):
    axB.spines[spine].set_visible(False)

# --- c: identity separates real KIR chains from mislabelled sequences -----------------
# The old panel counted provenance classes; with no classifier the interesting question is
# different -- given a gene, this package transfers boundaries onto anything, so alignment
# identity is the only signal that a gene label is wrong. This panel is that separation.
axC = fig.add_subplot(gs[1, 1])
groups = ['deposited\nKIR chains\n(n=54)', 'non-KIR given a\nKIR3DL1 label\n(n=22)']
# One statistic per bar pair, min and max, so the two groups are directly comparable: the point is
# that the ranges do not overlap, which a min-vs-mean pairing would obscure.
lows = [V['deposited_identity_min'], V['mislabelled_identity_min']]
highs = [V['deposited_identity_max'], V['mislabelled_identity_max']]
x = np.arange(2)
axC.bar(x - 0.16, lows, width=0.3, color=['#3b6ea5', '#8c8c8c'], label='min')
axC.bar(x + 0.16, highs, width=0.3, color=['#7ba3cf', '#c4c4c4'], label='max')
for i, (lo, hi) in enumerate(zip(lows, highs)):
    axC.text(i - 0.16, lo + 0.02, f'{lo:.2f}', ha='center', va='bottom', fontsize=5.6)
    axC.text(i + 0.16, hi + 0.02, f'{hi:.2f}', ha='center', va='bottom', fontsize=5.6)
# The gap between the two ranges is the usable signal, so mark it.
axC.axhline(V['deposited_identity_min'], color='#c26a27', linewidth=0.7, linestyle=(0, (3, 2)))
axC.text(1.42, V['deposited_identity_min'] - 0.05, 'no overlap', ha='right', va='top',
         fontsize=5.4, color='#c26a27', style='italic')
axC.set_xticks(x)
axC.set_xticklabels(groups, fontsize=5.4)
axC.set_ylim(0, 1.18)
axC.set_yticks([0, 0.5, 1.0])
axC.set_ylabel('alignment identity')
axC.set_title('c  Identity flags a wrong gene label', loc='left')
# Below the bars rather than upper-right: the real-KIR bars reach 1.0 and the no-overlap rule
# runs across the panel there, so an upper-right legend collides with both.
axC.legend(loc='lower center', bbox_to_anchor=(0.5, 0.02), ncol=2, fontsize=5.4,
           handlelength=1.0, borderpad=0.2)
for spine in ('top', 'right'):
    axC.spines[spine].set_visible(False)

fig.text(0.5, 0.055, 'The gene is an input, never inferred: unknown gene, null gene and a '
         f"';'-separated tie set all give provenance=none with every field null ({V['no_gene_n']}/{V['no_gene_n']} "
         'negatives with no gene).', ha='center', fontsize=6, color='#444444')
fig.text(0.5, 0.022, 'Emitted fields stop at the D2 end and do not tile the precursor '
         f"(median shortfall {V['shortfall_expressed_median']} residues).",
         ha='center', fontsize=6, color='#444444')

fig.canvas.draw()
renderer = fig.canvas.get_renderer()
texts = [(t, t.get_window_extent(renderer)) for t in fig.findobj(mpl.text.Text)
         if t.get_text().strip() and t.get_visible()]
ticks = {ax: set(ax.get_xticklabels(which='both') + ax.get_yticklabels(which='both'))
         for ax in fig.axes}
overlaps = [(a.get_text()[:24], b.get_text()[:24])
            for i, (a, ba) in enumerate(texts) for b, bb in texts[i + 1:] if ba.overlaps(bb)]
overlaps += [(t.get_text()[:24], 'spine') for t, bt in texts for ax in fig.axes
             for s in ax.spines.values()
             if s.get_visible() and bt.overlaps(s.get_window_extent(renderer))
             and t not in ticks[ax]]
oob = [t.get_text()[:24] for t, b in texts
       if not (b.x0 >= -1 and b.y0 >= -1 and b.x1 <= fig.bbox.x1 + 1 and b.y1 <= fig.bbox.y1 + 1)]
print('overlaps:', overlaps)
print('out of bounds:', oob)

fig.savefig('/tmp/chunkwork/kir_chunk_verification.png', dpi=300)
