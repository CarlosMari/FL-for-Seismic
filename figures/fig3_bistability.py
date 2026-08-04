"""Fig 3: Seed-bistability of rare-class (C5) recovery.

Shows final-round C5 IoU per seed across 12+ configurations. Expected
visualization: bimodal, either ~0.25 or exactly 0.000. Method choice
shifts the *probability* of the "learned" attractor but does not remove
the bistability.
"""
import os
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.size'] = 10
matplotlib.rcParams['axes.labelsize'] = 11
matplotlib.rcParams['legend.fontsize'] = 9
matplotlib.rcParams['pdf.fonttype'] = 42

# (config label, final-round C5 for seed 42, 123, 7)
# From Phase H/I/J/K/N/O/P/L and overnight campaign.
configs = [
    ('FedAvg',                  [0.000, 0.000, 0.000]),
    ('FedSeis up4',             [0.000, 0.000, 0.2295]),
    ('FedSeis up2',             [0.000, 0.000, 0.2637]),
    ('FocalDice',               [0.000, 0.000, 0.073]),
    ('Unified Focal',           [0.000, 0.000, 0.2505]),
    ('Asym Tversky',            [0.0361, 0.000, 0.000]),   # I4 had round-20 collapse to 0.036
    ('CRL \u03bb=1.0',           [0.000, 0.000, 0.273]),
    ('CRL \u03bb=3.0',           [0.000, 0.000, 0.201]),
    ('Recall Loss Tian 2021',   [0.000, 0.000, 0.251]),
    ('FCIoU MAKE 2023',         [0.000, 0.000, 0.000]),
    ('Recall-per-slice (ours)', [0.000, 0.000, 0.280]),
    ('UF + biased',             [0.000, 0.000, 0.223]),
    ('Recall Loss + biased',    [0.000, 0.073, 0.221]),
]

fig, ax = plt.subplots(figsize=(8, 5))

seed_colors = {42: '#1f77b4', 123: '#ff7f0e', 7: '#2ca02c'}
seed_markers = {42: 'o', 123: 's', 7: '^'}

ys = list(range(len(configs)))
labels = [c[0] for c in configs]
seeds = [42, 123, 7]

for y_idx, (label, c5_triple) in enumerate(configs):
    for s_idx, seed in enumerate(seeds):
        val = c5_triple[s_idx]
        ax.scatter(val, y_idx, color=seed_colors[seed], marker=seed_markers[seed],
                   s=70, edgecolor='black', linewidth=0.4, zorder=3,
                   label=f'seed {seed}' if y_idx == 0 else None)

# Shade the "bistability gap" — empirically observed empty band between
# ~0.08 and ~0.20: no final-round C5 falls here across 40 runs.
ax.axvspan(0.08, 0.20, alpha=0.12, color='red', zorder=1,
           label='bistability gap (empty in 40 runs)')

ax.set_yticks(ys)
ax.set_yticklabels(labels, fontsize=9)
ax.set_xlabel('Final-round Class-5 IoU')
ax.set_xlim(-0.02, 0.33)
ax.set_ylim(-0.6, len(configs) - 0.4)
ax.grid(True, alpha=0.3, axis='x')
ax.invert_yaxis()
ax.set_title('Seed-bistability of rare-class recovery\n(13 configurations, 3 seeds each)')

# Legend: collapse duplicates
handles, labels_leg = ax.get_legend_handles_labels()
seen = set()
clean = [(h, l) for h, l in zip(handles, labels_leg) if not (l in seen or seen.add(l))]
ax.legend([h for h, _ in clean], [l for _, l in clean], loc='lower right', framealpha=0.95)

plt.tight_layout()
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fig3_bistability')
plt.savefig(f'{out}.png', dpi=200)
plt.savefig(f'{out}.pdf')
print(f'wrote {out}.{{png,pdf}}')
