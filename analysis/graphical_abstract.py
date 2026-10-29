"""Graphical abstract (1062 x 2656 px, readable at 5 x 13 cm): tone and agency arc by profession, pooled over models.
Output: ../graphical_abstract/Graphical_abstract.pdf and .tif
"""
import os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.abspath(os.path.join(HERE, "..", "graphical_abstract"))
os.makedirs(OUT, exist_ok=True)
u = pd.read_csv(os.path.join(HERE, "results", "analysis_units_225.csv"))
P = ["Engineer", "Teacher", "Nurse", "Doctor", "Janitor"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE = "#2a78d6", "#eb6834"

def ci(x):
    rng = np.random.default_rng(42)
    b = [rng.choice(x, len(x)).mean() for _ in range(2000)]
    return x.mean(), np.percentile(b, 2.5), np.percentile(b, 97.5)

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.edgecolor": INK2, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2})
fig = plt.figure(figsize=(8, 3.2))
fig.text(0.02, 0.95, "Positive tone hides\nunequal agency", fontsize=15, fontweight="bold", color=INK, va="top", linespacing=1.15)
fig.text(0.02, 0.70, "3 LLMs, 75 personas,\n4,500 first-day-at-work\nstories", fontsize=10.5, color=INK2, va="top", linespacing=1.35)
fig.text(0.02, 0.18, "Sentiment audits see\npositive stories for all.\nNarrative agency shows\njanitors without a\ndevelopmental arc.",
         fontsize=10, color=INK, va="bottom", linespacing=1.35)

panels = [("sentiment_score", "Tone: uniformly positive", (-1, 1), "Mean tone (−1 to 1)", [-1, 0, 1]),
          ("agency_arc", "Agency arc: unequal", (0, 0.2), "Mean agency arc (Act III − Act I)", [0, 0.1, 0.2])]
for k, (col, title, lim, xlab, ticks) in enumerate(panels):
    ax = fig.add_axes([0.40 + k * 0.31, 0.22, 0.25, 0.58])
    for i, p in enumerate(P):
        m, lo, hi = ci(u.loc[u.profession == p, col].values)
        c = ORANGE if p == "Janitor" else BLUE
        ax.plot([lo, hi], [i, i], color=c, lw=2, solid_capstyle="round")
        ax.plot(m, i, "o", ms=8, color=c, mec="white", mew=1.5, zorder=3)
    if col == "sentiment_score":
        ax.axvline(0, color=INK2, lw=1, ls=(0, (2, 2)))
        ax.text(0.05, -0.45, "neutral", fontsize=9, color=INK2, va="top")
    ax.set_yticks(range(len(P))); ax.set_yticklabels(P if k == 0 else [])
    for t, p in zip(ax.get_yticklabels(), P):
        t.set_color(ORANGE if p == "Janitor" else INK); t.set_fontweight("bold" if p == "Janitor" else "normal")
    ax.set_ylim(len(P) - 0.4, -0.6); ax.set_xlim(*lim); ax.set_xticks(ticks)
    ax.set_xlabel(xlab, fontsize=9.5)
    ax.set_title(title, fontsize=11.5, color=INK, loc="left", fontweight="bold")
    ax.grid(axis="x", color=GRID, lw=0.8); ax.set_axisbelow(True)
    for s in ["top", "right", "left"]: ax.spines[s].set_visible(False)
    ax.tick_params(axis="y", length=0)
fig.text(0.40, 0.03, "Means with 95% bootstrap intervals over 45 persona–model units; three models pooled.", fontsize=7.5, color=INK2)
fig.savefig(os.path.join(OUT, "Graphical_abstract.pdf"))
fig.savefig(os.path.join(OUT, "Graphical_abstract.tif"), dpi=332, pil_kwargs={"compression": "tiff_lzw"})
print("written", OUT)
