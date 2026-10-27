"""F7 (redesigned) - "Silence is a signal".

Six facies drawn as seismic-style wiggle traces over training rounds. A facies
the site HOLDS shows signal (the row moves); a facies it LACKS is a dead-flat
trace -> the silence itself reveals the absence. Clean, sparse, seismic-flavoured.
"""
import numpy as np
import matplotlib.pyplot as plt
from talk_figs import style as S


def wiggle(ax, x, y, color, amp=1.0, baseline=0.0, fill=True):
    ax.plot(x, y + baseline, color=color, lw=2.2, solid_capstyle="round")
    if fill:
        ax.fill_between(x, baseline, y + baseline, where=(y >= 0),
                        color=color, alpha=0.30, linewidth=0)


def main(seed=7):
    rng = np.random.default_rng(seed)
    x = np.linspace(0, 10, 220)
    names = S.F3_NAMES
    absent = 5  # Zechstein absent at this site
    fig, ax = plt.subplots(figsize=(11, 6.2))

    for k in range(6):
        base = 5.8 - k * 1.12
        if k == absent:
            y = np.zeros_like(x)              # dead flat = silent
            ax.plot(x, np.full_like(x, base), color="#c62828", lw=3.4, solid_capstyle="round")
        else:
            # signal: a decaying-noise wiggle (the row is 'active')
            env = np.exp(-0.10 * x) * (0.7 + 0.3 * np.sin(0.8 * x + k))
            y = env * np.sin(2.6 * x + 1.3 * k) + 0.12 * rng.normal(size=len(x))
            wiggle(ax, x, y, S.FACIES_COLORS[k], baseline=base)
        ax.text(-0.35, base, names[k], ha="right", va="center",
                fontsize=15, weight="bold",
                color="#c62828" if k == absent else S.FACIES_COLORS[k])

    # highlight the silent row
    ax.text(9.85, 5.8 - absent * 1.12 + 0.5, "absent here  →  silent",
            ha="right", color="#c62828", fontsize=15, weight="bold")

    ax.set_xlim(-0.35, 10); ax.set_ylim(-0.7, 6.6)
    ax.set_xlabel("training rounds", fontsize=14)
    ax.set_yticks([])
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    S.save(fig, "f7_silence", (11, 6.2))
    print("f7 redesigned")


if __name__ == "__main__":
    main()
