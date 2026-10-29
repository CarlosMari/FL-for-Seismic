"""F8 (revised) - "The head update gives you away": who holds each facies.

A presence matrix the interpreter reads at a glance: facies (rows, stratigraphic
order) x operator (columns). A filled facies-coloured cell = that operator HOLDS
the class; an empty cell with an X = they LACK it. The pattern of gaps is the
fingerprint — it is exactly what a model update's head row reveals (a class a
site holds moves the row; one it lacks stays silent).
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from fedseismic.privacy.log import load_labels
from talk_figs import style as S

RUN = "results/runs/seismic_f3/fedavg_noniid_a0.1_lNA_dna_n3_batchnorm/seed_0"


def main():
    hist, _ = load_labels(RUN)          # {client: pi over train pixels}
    names = S.F3_NAMES
    ops = ["Operator A", "Operator B", "Operator C"]
    present = np.array([[hist[i][k] > 0 for i in range(3)] for k in range(6)])

    fig, ax = plt.subplots(figsize=(7.2, 6.6))
    for k in range(6):                   # facies rows (stratigraphic, top->bottom)
        for i in range(3):               # operator columns
            x, y = i, 5 - k
            if present[k, i]:
                ax.add_patch(Rectangle((x - 0.42, y - 0.42), 0.84, 0.84,
                                       facecolor=S.FACIES_COLORS[k], edgecolor="white", lw=1.5))
            else:
                ax.add_patch(Rectangle((x - 0.42, y - 0.42), 0.84, 0.84,
                                       facecolor="white", edgecolor="#C23B22", lw=2.0))
                ax.plot([x - 0.26, x + 0.26], [y - 0.26, y + 0.26], color="#C23B22", lw=2.4)
                ax.plot([x - 0.26, x + 0.26], [y + 0.26, y - 0.26], color="#C23B22", lw=2.4)

    ax.set_xlim(-0.75, 2.75); ax.set_ylim(-1.55, 5.75)
    ax.set_xticks(range(3)); ax.set_xticklabels(ops, fontsize=15, color=S.NAVY)
    ax.set_yticks(range(6)); ax.set_yticklabels(list(reversed(names)), fontsize=13, color=S.NAVY)
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.xaxis.tick_top()

    # key
    ax.add_patch(Rectangle((0.05, -1.32), 0.34, 0.34, facecolor=S.FACIES_COLORS[2], edgecolor="white"))
    ax.text(0.5, -1.15, "holds the class", va="center", fontsize=12, color=S.NAVY)
    ax.add_patch(Rectangle((1.55, -1.32), 0.34, 0.34, facecolor="white", edgecolor="#C23B22", lw=2))
    ax.plot([1.62, 1.82], [-1.25, -1.05], color="#C23B22", lw=2)
    ax.plot([1.62, 1.82], [-1.05, -1.25], color="#C23B22", lw=2)
    ax.text(2.0, -1.15, "lacks it", va="center", fontsize=12, color=S.NAVY)

    S.save(fig, "f8_head_fingerprint", (7.2, 6.6))
    print("f8 head-fingerprint written")


if __name__ == "__main__":
    main()
