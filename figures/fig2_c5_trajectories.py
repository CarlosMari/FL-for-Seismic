"""Fig 2: per-round Class-5 IoU trajectories for headline configurations.

Shows that Recall-per-slice (P3 s7) is the only configuration with
sustained C5 across multiple final rounds.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, 'utils'))
from extract_c5_trajectory import parse_log
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.size'] = 10
matplotlib.rcParams['axes.labelsize'] = 11
matplotlib.rcParams['legend.fontsize'] = 9
matplotlib.rcParams['pdf.fonttype'] = 42  # TrueType for embedding

ROOT = os.path.dirname(HERE)

configs = [
    # (display name, log path, color, linestyle, linewidth)
    ('Recall-per-slice (ours, P3 s7)', f'{ROOT}/logs_phase_p/P3_recall_slice_s7.log', '#d62728', '-', 2.5),
    ('Recall Loss Tian 2021 (N3 s7)',  f'{ROOT}/logs_phase_no/N3_recall_s7.log',      '#ff7f0e', '-', 1.8),
    ('CRL \u03bb=1.0 (I9 s7)',          f'{ROOT}/logs_loss_sweep/I9_crossrecall_s7.log', '#2ca02c', '-', 1.8),
    ('Unified Focal (I3 s7)',          f'{ROOT}/logs_loss_sweep/I3_unifiedfocal_s7.log', '#1f77b4', '-', 1.8),
    ('FedAvg baseline (D2 s7)',        f'{ROOT}/logs_overnight/D2_fedavg_seed7.log',  '#7f7f7f', '--', 1.4),
]

fig, ax = plt.subplots(figsize=(8.5, 4.4))
for name, path, color, ls, lw in configs:
    try:
        traj = parse_log(path)
    except FileNotFoundError:
        print(f'[warn] missing: {path}')
        continue
    rnds = [t['round'] for t in traj]
    c5 = [t['per_class'][5] for t in traj]
    ax.plot(rnds, c5, color=color, linestyle=ls, linewidth=lw, label=name, marker='o', markersize=3)

ax.set_xlabel('Round')
ax.set_ylabel('Class-5 IoU')
ax.set_title('Per-round Class-5 IoU: headline configurations (seed 7)')
ax.set_xticks(list(range(1, 21, 2)) + [20])
ax.set_xlim(0.5, 20.5)
ax.grid(True, alpha=0.3)
ax.legend(loc='upper left', framealpha=0.9)
ax.set_ylim(-0.02, 0.35)
ax.axhline(0, color='black', linewidth=0.5)

plt.tight_layout()
out = os.path.join(HERE, 'fig2_c5_trajectories')
plt.savefig(f'{out}.png', dpi=200)
plt.savefig(f'{out}.pdf')
print(f'wrote {out}.{{png,pdf}}')
