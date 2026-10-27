"""CONTACT_SHEET - all final talk figures on one page for quick review."""
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from talk_figs import style as S

FIGS = [
    ("F1  operators + facies mix", "f1_sections_n3.png"),
    ("F2  section triptych", "f2_triptych_n3.png"),
    ("F3  the blackout (BN)", "f3_strip.png"),
    ("F4  training movie (grid)", "f4_rounds_grid.png"),
    ("F5  fingerprint (BN logs)", "f5_sections_bnlogs.png"),
    ("F6  who has the Zechstein", "f6_sections_n3.png"),
    ("F7  silence is a signal", "f7_silence.png"),
    ("F8  sealed totals", "f8_sealed_sum.png"),
    ("F9  what stays home", "f9_unet_keep_home.png"),
]


def main():
    rows, cols = 3, 3
    fig, axes = plt.subplots(rows, cols, figsize=(16, 11))
    for ax in axes.ravel():
        ax.axis("off")
    for (title, fname), ax in zip(FIGS, axes.ravel()):
        p = os.path.join(S.OUT, fname)
        if os.path.exists(p):
            img = mpimg.imread(p)
            ax.imshow(img)
            ax.set_title(title, fontsize=13, color=S.NAVY, pad=6)
        else:
            ax.text(0.5, 0.5, f"{fname}\n(missing)", ha="center", va="center",
                    fontsize=12, color="#999999")
            ax.set_title(title, fontsize=13, color="#999999", pad=6)
    fig.suptitle("Keep It Local — talk figures", fontsize=15, color=S.NAVY, y=0.995)
    fig.tight_layout()
    S.save(fig, "CONTACT_SHEET", (16, 11))
    print("contact sheet written")


if __name__ == "__main__":
    main()
