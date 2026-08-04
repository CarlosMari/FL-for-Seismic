"""Fig 4: Variance compression under biased client sampling.

Shows per-seed mIoU for uniform vs biased sampling on two losses
(Unified Focal, Recall Loss). Biased sampling compresses n=3 std ~4x
while costing peak performance on seed 7.
"""
import os
import matplotlib.pyplot as plt
import matplotlib
import statistics as s
matplotlib.rcParams['font.size'] = 10
matplotlib.rcParams['axes.labelsize'] = 11
matplotlib.rcParams['legend.fontsize'] = 9
matplotlib.rcParams['pdf.fonttype'] = 42

# mIoU per seed for each (loss, sampling) cell
data = {
    'Unified Focal\nuniform':   {42: 0.5850, 123: 0.5554, 7: 0.5954},
    'Unified Focal\nbiased':    {42: 0.5587, 123: 0.5679, 7: 0.5651},
    'Recall Loss\nuniform':     {42: 0.5613, 123: 0.5635, 7: 0.5927},
    'Recall Loss\nbiased':      {42: 0.5748, 123: 0.5707, 7: 0.5663},
}

seed_colors = {42: '#1f77b4', 123: '#ff7f0e', 7: '#2ca02c'}
seed_markers = {42: 'o', 123: 's', 7: '^'}

fig, ax = plt.subplots(figsize=(7.5, 4.5))

groups = list(data.keys())
xs = list(range(len(groups)))

for i, g in enumerate(groups):
    vals = list(data[g].values())
    mn, sd = s.mean(vals), s.stdev(vals)
    # mean + sd band
    ax.errorbar(i, mn, yerr=sd, fmt='_', color='black', capsize=8, capthick=2,
                linewidth=2, zorder=2, label='n=3 mean $\\pm$ std' if i == 0 else None)
    # per-seed points
    for seed, val in data[g].items():
        ax.scatter(i, val, color=seed_colors[seed], marker=seed_markers[seed],
                   s=90, edgecolor='black', linewidth=0.5, zorder=3,
                   label=f'seed {seed}' if i == 0 else None)
    # annotate std
    ax.annotate(f'$\\sigma$={sd:.3f}', (i, mn - sd - 0.004),
                ha='center', va='top', fontsize=8.5, color='dimgray')

# shading per loss family
ax.axvspan(-0.5, 1.5, alpha=0.06, color='blue', zorder=0)
ax.axvspan(1.5, 3.5, alpha=0.06, color='orange', zorder=0)

ax.set_xticks(xs)
ax.set_xticklabels(groups)
ax.set_ylabel('Best mIoU (20 rounds)')
ax.set_xlim(-0.5, 3.5)
ax.set_ylim(0.535, 0.605)
ax.grid(True, alpha=0.3, axis='y')
ax.set_title('Biased client sampling compresses seed variance $\\sim$4$\\times$')

# Deduplicate legend
handles, labels_leg = ax.get_legend_handles_labels()
seen = set()
clean = [(h, l) for h, l in zip(handles, labels_leg) if not (l in seen or seen.add(l))]
ax.legend([h for h, _ in clean], [l for _, l in clean], loc='lower right', framealpha=0.95)

plt.tight_layout()
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fig4_variance')
plt.savefig(f'{out}.png', dpi=200)
plt.savefig(f'{out}.pdf')
print(f'wrote {out}.{{png,pdf}}')
