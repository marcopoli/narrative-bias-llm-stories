"""Figures 1-4 of the HCIS article (PNG at 1,200 dpi, PDF and EPS in ../figures).

Fig. 1 audit framework; Fig. 2 janitor gaps across the eight story models and the shape of the stories;
Fig. 3 occupational status and human warmth norms against the rated portrayal (Experiment 4);
Fig. 4 mitigation strategies (Experiment 5). Inputs as in final_tables.py. Run with the system python3.
"""
import json, os, sqlite3
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
FIG = os.path.abspath(os.path.join(HERE, "..", "figures"))
# validated categorical slots 1-3 (all-pairs pass, light surface) plus neutral ink and gray
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID, GRAY = "#0b0b0b", "#52514e", "#e4e3df", "#9b9a96"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
                     "legend.frameon": False, "savefig.facecolor": "white"})
N = json.load(open(os.path.join(RES, "final_numbers.json")))

def save(fig, name):
    for ext, kw in [("png", {"dpi": 1200}), ("pdf", {}), ("eps", {})]:
        fig.savefig(os.path.join(FIG, f"{name}.{ext}"), bbox_inches="tight", **kw)
    plt.close(fig)

# ------------------------------------------------------------------ Fig. 1 framework
# drawn close to the printed size (text width about 5.2 in), in inch coordinates; check_layout() stops the script
# if a text leaves its box or touches another text or an arrow
W, H = 5.7, 3.46
TITLE, BODY, LH = 7.4, 6.8, 0.128           # font sizes (pt) and line height (in)
fig = plt.figure(figsize=(W, H)); ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")
TEXTS, BOXES, ARROWS = [], [], []

def rbox(x, y, w, h, ec, fc="white", lw=1.0):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.06", fc=fc, ec=ec, lw=lw))

def block(x, y, w, h, title, lines, color, sub=None):
    rbox(x, y, w, h, color); BOXES.append((x, y, w, h))
    t = [ax.text(x + 0.09, y + h - 0.1, title, fontsize=TITLE, weight="bold", color=INK, va="top")]
    yy = y + h - 0.1 - 0.17
    if sub:
        t.append(ax.text(x + 0.09, yy, sub, fontsize=BODY, color=color if color != GRAY else MUTED, style="italic", va="top"))
        yy -= LH + 0.03
    for k, line in enumerate(lines):
        t.append(ax.text(x + 0.09, yy - k * LH, line, fontsize=BODY, color=MUTED, va="top"))
    TEXTS.extend((tt, (x, y, w, h)) for tt in t)

def arr(x0, y0, x1, y1, color=MUTED, head=True, ls="-"):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="-|>" if head else "-", color=color, lw=1.0, ls=ls,
                shrinkA=0, shrinkB=0, mutation_scale=9))
    ARROWS.append((min(x0, x1), min(y0, y1), abs(x1 - x0), abs(y1 - y0)))

LX, SW, GAP = 0.04, 1.5, 0.27               # left column x, side column width, gap for arrows
MX = LX + SW + GAP; MW = W - 2 * LX - 2 * SW - 2 * GAP; RX = MX + MW + GAP
TOP, BOT = H - 0.04, 0.5; HB = (TOP - BOT - GAP) / 2
block(LX, BOT + HB + GAP, SW, HB, "Personas and prompts", ["5 professions", "(13 in Exp. 4)", "3 gender labels", "5 nationalities",
      "original prompt and", "variants (Exp. 2, 5)"], GRAY)
block(LX, BOT, SW, HB, "Story generation", ["8 local open-weights", "LLMs, 4B to 31B", "temperature 0.7", "common seeds", "(Exp. 4, 5)",
      "10,770 stories"], GRAY)
block(RX, BOT + HB + GAP, SW, HB, "External data", ["occupational prestige", "(Hughes et al., 2024)", "human warmth and",
      "competence norms", "(Friehs et al., 2022)"], AQUA, sub="Exp. 4")
block(RX, BOT, SW, HB, "Mitigation", ["equal-agency instruction", "job-blind drafting", "self-revision", "side effects: similarity,",
      "tone, length"], AQUA, sub="Exp. 5")
# central audit with its four levels
rbox(MX, BOT, MW, TOP - BOT, GRID, fc="#f6f6f4", lw=0.8); BOXES.append((MX, BOT, MW, TOP - BOT))
TEXTS.append((ax.text(MX + MW / 2, TOP - 0.1, "Narrative audit", fontsize=TITLE + 0.4, weight="bold", color=INK, va="top", ha="center"),
              (MX, BOT, MW, TOP - BOT)))
TEXTS.append((ax.text(MX + MW / 2, TOP - 0.29, "per persona, within each model", fontsize=BODY, color=MUTED, va="top", ha="center"),
              (MX, BOT, MW, TOP - BOT)))
levels = [("L1 Surface", ["tone, regard, toxicity:", "what usual audits check"], GRAY),
          ("L2 Story shape", ["agency arc: from being", "acted upon to acting"], BLUE),
          ("L3 Actions", ["agency quality: expert", "or routine actions"], BLUE),
          ("L4 Blind rating", ["masked LLM raters: 1-7 scales", "and pairwise judgements"], ORANGE)]
lx, lw_ = MX + 0.1, MW - 0.2
ltop, lbot, lg = TOP - 0.46, BOT + 0.09, 0.07
lh = (ltop - lbot - 3 * lg) / 4
for k, (name, lines, col) in enumerate(levels):
    y = ltop - (k + 1) * lh - k * lg
    block(lx, y, lw_, lh, name, lines, col)
# flow
mid_low, mid_up = BOT + HB / 2, BOT + HB + GAP + HB / 2
arr(LX + SW / 2, BOT + HB + GAP, LX + SW / 2, BOT + HB)               # personas -> generation
arr(LX + SW, mid_low, MX, mid_low)                                     # generation -> audit
arr(RX, mid_up, MX + MW, mid_up)                                       # external data -> audit
arr(MX + MW, mid_low, RX, mid_low)                                     # audit -> mitigation
fy = 0.3                                                               # feedback loop below the boxes
arr(RX + SW / 2, BOT, RX + SW / 2, fy, AQUA, head=False, ls="--")
arr(RX + SW / 2, fy, LX + SW / 2, fy, AQUA, head=False, ls="--")
arr(LX + SW / 2, fy, LX + SW / 2, BOT, AQUA, ls="--")
TEXTS.append((ax.text(W / 2, fy - 0.07, "stories written with a mitigation strategy go through the same audit", fontsize=BODY,
                      color=MUTED, ha="center", va="top"), (0, 0, W, H)))

def check_layout():
    fig.canvas.draw(); r = fig.canvas.get_renderer(); inv = ax.transData.inverted()
    ext = []
    for t, (x, y, w, h) in TEXTS:
        (x0, y0), (x1, y1) = inv.transform(t.get_window_extent(r).get_points())
        assert x - 1e-6 <= x0 and x1 <= x + w - 0.04 and y + 0.02 <= y0 and y1 <= y + h - 0.02, f"text outside box: {t.get_text()}"
        ext.append((t.get_text(), x0, y0, x1, y1))
    for i_, a in enumerate(ext):
        for b_ in ext[i_ + 1:]:
            assert a[3] <= b_[1] or b_[3] <= a[1] or a[4] <= b_[2] or b_[4] <= a[2], f"texts overlap: {a[0]} / {b_[0]}"
        for (ax0, ay0, aw, ah) in ARROWS:
            assert ax0 + aw <= a[1] or a[3] <= ax0 or ay0 + ah <= a[2] or a[4] <= ay0, f"arrow crosses text: {a[0]}"
check_layout()
save(fig, "Fig1")

# ------------------------------------------------------------------ Fig. 2 janitor gaps across models + story shape
T2 = N["table2"]; models = list(T2)
fig, (a, b) = plt.subplots(1, 2, figsize=(7.1, 3.0), gridspec_kw={"width_ratios": [1.45, 1]})
ys = np.arange(len(models))[::-1]
for k, (key, lab, col, mk) in enumerate([("warmth", "Warmth", AQUA, "s"), ("competence", "Competence", BLUE, "o"),
                                         ("agency", "Agency", ORANGE, "D")]):
    for j, m in enumerate(models):
        g = T2[m][key]["diff"]
        a.errorbar(g[0], ys[j] + (1 - k) * 0.24, xerr=[[g[0] - g[1]], [g[2] - g[0]]], fmt=mk, color=col, ms=4.5, lw=0.9,
                   capsize=1.5, mec="white", mew=0.5, zorder=3, label=lab if j == 0 else None)
a.axvline(0, color=MUTED, lw=0.8)
a.set_yticks(ys); a.set_yticklabels(models); a.set_xlabel("Janitor gap in blind rating (janitor minus others, 1-7 scale)")
a.grid(axis="x", color=GRID, lw=0.6); a.set_axisbelow(True); a.axhline(4.5, color=GRID, lw=0.8, ls=":")
a.text(0.62, 4.62, "main corpus", fontsize=6.4, color=MUTED, ha="right", va="bottom")
a.text(0.62, 4.38, "newer models", fontsize=6.4, color=MUTED, ha="right", va="top")
a.set_xlim(-1.5, 0.65); a.set_ylim(-0.95, 7.55)
a.text(-1.46, -0.72, "← less for janitors", fontsize=6.4, color=MUTED, va="center")
a.text(0.62, -0.72, "more →", fontsize=6.4, color=MUTED, va="center", ha="right")
a.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, fontsize=7, handletextpad=0.3, columnspacing=1.0)
a.set_title("a", loc="left", weight="bold", fontsize=9)
con = sqlite3.connect(os.path.join(HERE, "..", "analysis", "master_results.db"))
acts = pd.read_sql_query("""SELECT r.model_name AS model, r.persona_id, p.profession, r.agency_start, r.agency_middle, r.agency_end
    FROM results r JOIN personas p ON r.persona_id = p.id""", con)
u = acts.groupby(["model", "persona_id", "profession"], as_index=False)[["agency_start", "agency_middle", "agency_end"]].mean()
ends = {}
for prof in ["Engineer", "Teacher", "Nurse", "Doctor", "Janitor"]:
    g = u[u.profession == prof][["agency_start", "agency_middle", "agency_end"]]
    m, se = g.mean().values, 1.96 * g.std().values / np.sqrt(len(g))
    jan = prof == "Janitor"
    b.errorbar(range(3), m, yerr=se, color=ORANGE if jan else GRAY, lw=1.6 if jan else 0.9, marker="D" if jan else "o",
               ms=4, capsize=2, alpha=1 if jan else 0.8, zorder=3 if jan else 2)
    ends[prof] = (m[2], m[0])
order = sorted(ends, key=lambda p: ends[p][0])
pos = [ends[order[0]][0] - 0.004]
for p in order[1:]: pos.append(max(ends[p][0], pos[-1] + 0.011))
for p, y in zip(order, pos):
    b.text(2.1, y, p, fontsize=6.4, color=ORANGE if p == "Janitor" else MUTED, va="center")
b.set_xticks(range(3)); b.set_xticklabels(["Act I", "Act II", "Act III"]); b.set_xlim(-0.2, 2.75)
b.set_ylabel("Human subject share (↑ more acting)"); b.grid(axis="y", color=GRID, lw=0.6); b.set_axisbelow(True)
b.set_title("b", loc="left", weight="bold", fontsize=9)
fig.tight_layout(w_pad=2.0); save(fig, "Fig2")

# ------------------------------------------------------------------ Fig. 3 status and warmth norms (Experiment 4)
occ = pd.DataFrame(N["exp4_occ_means_pooled"]).T
pre = pd.read_csv(os.path.join(HERE, "norms", "occupation_prestige_hughes2022.csv")).set_index("occupation").prestige
wn = pd.read_csv(os.path.join(HERE, "norms", "occupation_norms_friehs2022.csv")).set_index("occupation").warmth
fig, (a, b) = plt.subplots(1, 2, figsize=(7.1, 3.0))
# label positions in data coordinates for crowded points (a thin leader line joins label and point)
OFFSET = {"Childcare Worker": (29, 5.27), "Farmer": (36, 5.52), "Teacher": (50, 5.55), "Nurse": (66, 5.21),
          "Police Officer": (48.5, 5.29)}
OFFSET_B = {"Farmer": (-4, 4), "Doctor": (4, 3), "Childcare Worker": (-4, -8)}
def scatter(ax, x, y, xlabel, ylabel, rho, title, offsets):
    for o in y.index:
        if o not in x.index: continue
        jan = o == "Janitor"
        ax.scatter(x[o], y[o], s=30 if jan else 22, color=ORANGE if jan else BLUE, marker="D" if jan else "o",
                   edgecolor="white", linewidth=0.6, zorder=3)
        if o in offsets and title == "a":
            ax.annotate(o, (x[o], y[o]), xytext=offsets[o], textcoords="data", fontsize=6.2, color=MUTED, va="center",
                        arrowprops=dict(arrowstyle="-", color=GRAY, lw=0.5, shrinkA=1, shrinkB=3))
        else:
            dx, dy = offsets.get(o, (4, 2)) if title == "b" else (4, 2)
            ax.annotate(o, (x[o], y[o]), xytext=(dx, dy), textcoords="offset points", fontsize=6.2,
                        color=ORANGE if jan else MUTED, ha="right" if dx < 0 else "left")
    xs = x.reindex(y.index).dropna(); fit = np.polyfit(xs, y[xs.index], 1)
    gx = np.linspace(xs.min(), xs.max(), 10); ax.plot(gx, np.polyval(fit, gx), color=GRAY, lw=0.9, ls="--", zorder=1)
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.grid(color=GRID, lw=0.6); ax.set_axisbelow(True)
    ax.text(0.03, 0.96, rho, transform=ax.transAxes, fontsize=7, color=INK, va="top")
    ax.set_title(title, loc="left", weight="bold", fontsize=9)
r1 = N["exp4"]["pooled"]["rho_prestige_agency"]; r2 = N["exp4"]["pooled"]["rho_warmth_warmth"]
scatter(a, pre, occ["agency"], "Occupational prestige (0-100)", "Rated agency (1-7)", f"Spearman ρ = {r1[0]:.2f}, p = {r1[1]:.3f}", "a", OFFSET)
scatter(b, wn, occ["warmth"], "Human warmth norm (1-5)", "Rated warmth (1-7)", f"Spearman ρ = {r2[0]:.2f}, p = {r2[1]:.3f}", "b", OFFSET_B)
fig.tight_layout(w_pad=2.0); save(fig, "Fig3")

# ------------------------------------------------------------------ Fig. 4 mitigation (Experiment 5)
E5 = N["exp5"]
STRAT = [("baseline", "Original prompt", GRAY, "o"), ("parity", "Equal-agency instruction", BLUE, "s"),
         ("blind", "Job-blind drafting", AQUA, "^"), ("revise", "Self-revision", ORANGE, "D")]
fig, (a, b) = plt.subplots(1, 2, figsize=(7.1, 3.0), gridspec_kw={"width_ratios": [1.2, 1]})
P = N.get("pairs", {})
mods = [m for m in ["Mistral 7B", "Llama 3 8B", "Gemma 4 31B"] if m in P]
w = 0.18
for i, m in enumerate(mods):
    have = [st for st in STRAT if st[0] in P[m]]
    for k, (v, lab, col, mk) in enumerate(have):
        q = P[m][v]
        x, y = i + (k - (len(have) - 1) / 2) * w, 100 * q["janitor_share"]
        a.errorbar(x, y, yerr=[[y - 100 * q["ci"][0]], [100 * q["ci"][1] - y]], fmt=mk, color=col, ms=5, capsize=2, lw=1,
                   mec="white", mew=0.5, label=lab if i == 0 else None)
a.axhline(50, color=MUTED, lw=0.8, ls="--")
a.text(-0.5, 51.5, "equal treatment", fontsize=6.2, color=MUTED, ha="left", va="bottom")
a.set_xticks(range(len(mods))); a.set_xticklabels(mods); a.set_xlim(-0.55, len(mods) - 0.45); a.set_ylim(-3, 85)
a.set_ylabel("Pairs won by the janitor character (%)"); a.grid(axis="y", color=GRID, lw=0.6); a.set_axisbelow(True)
a.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, fontsize=7, handletextpad=0.3)
a.set_title("a", loc="left", weight="bold", fontsize=9)
for m, filled in [("Mistral 7B", True), ("Llama 3 8B", False)]:
    if m not in E5: continue
    pts = [(E5[m][v]["similarity"], E5[m][v]["agency"]["mean"]) for v, *_ in STRAT]
    for (v, lab, col, mk), (x, y) in zip(STRAT, pts):
        if v != "baseline": b.plot([pts[0][0], x], [pts[0][1], y], color=GRID, lw=0.8, zorder=1)
        b.scatter(x, y, s=30, marker=mk, color=col if filled else "white", edgecolor=col, linewidth=1.0 if not filled else 0.6, zorder=3)
    b.annotate(m, pts[0], xytext=(6, 0), textcoords="offset points", fontsize=6.2, color=MUTED, va="center")
b.text(0.98, 0.97, "filled: Mistral 7B   open: Llama 3 8B", transform=b.transAxes, fontsize=6.2, color=MUTED, ha="right", va="top")
b.set_ylim(4.0, 7.25); b.set_xlabel("Similarity of stories for different jobs (TF-IDF cosine)"); b.set_ylabel("Mean rated agency, all characters (1-7)")
b.grid(color=GRID, lw=0.6); b.set_axisbelow(True); b.set_title("b", loc="left", weight="bold", fontsize=9)
fig.tight_layout(w_pad=2.0); save(fig, "Fig4")
print("figures written to", FIG)
