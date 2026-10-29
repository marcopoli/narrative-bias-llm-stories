"""
reproduce_analysis.py
Reproduces every statistic, table and figure reported in the article (main text and appendices)
from the released story database (master_results.db).

Usage:  python reproduce_analysis.py
Outputs: ./results/*.csv, ./results/summary.txt, figures Fig1-Fig4, FigB1, FigC1 (./figures/)
Requires: numpy, pandas, scipy, statsmodels, matplotlib
"""
import os, sqlite3, re, json
import numpy as np, pandas as pd
from itertools import combinations
from scipy import stats
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multitest import multipletests
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
def _find_db():
    """Locates master_results.db: $NARRATIVE_DB, this folder, or ../data/ (repository layout)."""
    for p in (os.environ.get("NARRATIVE_DB", ""), os.path.join(HERE, "master_results.db"),
              os.path.join(HERE, "..", "data", "master_results.db")):
        if p and os.path.exists(p):
            return os.path.abspath(p)
    raise FileNotFoundError("master_results.db not found (set NARRATIVE_DB or place it in ../data/)")
DB = _find_db()
RES = os.path.join(HERE, "results"); os.makedirs(RES, exist_ok=True)
FIG = os.path.abspath(os.path.join(HERE, "..", "figures"))
os.makedirs(FIG, exist_ok=True)

def _save(fig, name):
    """PNG (1200 dpi) for the Word manuscript; vector PDF (LaTeX) and EPS (artwork upload)."""
    fig.savefig(os.path.join(FIG, name + ".png"), dpi=1200)
    fig.savefig(os.path.join(FIG, name + ".pdf"))
    fig.savefig(os.path.join(FIG, name + ".eps"))
SEED = 42
N_BOOT = 2000
N_PERM = 2000

P = ["Engineer", "Teacher", "Nurse", "Doctor", "Janitor"]
G = ["Male", "Female", "Non-Binary"]
N = ["American", "Chinese", "German", "Indian", "Nigerian"]
MOD = ["gemma3:4b", "llama3", "mistral"]
MOD_LAB = {"gemma3:4b": "Gemma 3 4B", "llama3": "Llama 3 8B", "mistral": "Mistral 7B"}
MET = ["sentiment_score", "agency_arc", "agency_quality_score", "concreteness_score"]
MET_LAB = {"sentiment_score": "P1 Tone", "agency_arc": "P2 Agency arc",
           "agency_quality_score": "P3 Agency quality", "concreteness_score": "P4 Lexical expertise"}
# Okabe-Ito palette (colour-blind safe)
OI = ["#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7", "#000000"]
MOD_COL = {"gemma3:4b": OI[4], "llama3": OI[5], "mistral": OI[2]}
MOD_MK = {"gemma3:4b": "o", "llama3": "s", "mistral": "^"}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.spines.top": False,
                     "axes.spines.right": False, "savefig.dpi": 300, "figure.dpi": 100})

out = []
def log(*a):
    s = " ".join(str(x) for x in a); print(s); out.append(s)

# ------------------------------------------------------------------ data
con = sqlite3.connect(DB)
raw = pd.read_sql_query("""SELECT r.id, r.model_name, r.persona_id, p.gender, p.nationality, p.profession,
       r.story_text, r.sentiment_score, r.agency_start, r.agency_middle, r.agency_end, r.agency_arc,
       r.agency_quality_score, r.concreteness_score, r.taxonomy_label
       FROM results r JOIN personas p ON r.persona_id = p.id""", con)
con.close()
raw["words"] = raw.story_text.str.split().str.len()
log("stories:", len(raw), "| per model:", raw.model_name.value_counts().to_dict())
log("story length (words): mean %.1f, SD %.1f, range %d-%d" % (raw.words.mean(), raw.words.std(), raw.words.min(), raw.words.max()))
log("story-level P1: mean %.3f, SD %.3f, min %.3f, max %.3f, share < 0: %.3f" % (
    raw.sentiment_score.mean(), raw.sentiment_score.std(), raw.sentiment_score.min(), raw.sentiment_score.max(), (raw.sentiment_score < 0).mean()))
for c in ["agency_start", "agency_middle", "agency_end"]:
    log(f"fallback (=0.5, no human mention) share in {c}: {(raw[c] == 0.5).mean():.3f}")

keys = ["persona_id", "model_name", "gender", "nationality", "profession"]
agg = raw.groupby(keys)[MET + ["agency_start", "agency_middle", "agency_end"]].mean().reset_index()
agg.to_csv(os.path.join(RES, "analysis_units_225.csv"), index=False)
log("analysis units:", len(agg))

def eta2_h(H, k, n): return (H - k + 1) / (n - k)
def cohen_d(a, b):
    s = np.sqrt((np.var(a, ddof=1) + np.var(b, ddof=1)) / 2); return (np.mean(a) - np.mean(b)) / s
def boot_d(a, b, seed=SEED):
    r = np.random.default_rng(seed)
    bs = [cohen_d(r.choice(a, len(a)), r.choice(b, len(b))) for _ in range(N_BOOT)]
    return np.percentile(bs, [2.5, 97.5])
def stars(q): return "***" if q < .001 else "**" if q < .01 else "*" if q < .05 else "ns"

# ------------------------------------------------------------------ descriptives
desc = agg.groupby("profession")[MET + ["agency_start", "agency_end"]].agg(["mean", "std"]).reindex(P)
desc.to_csv(os.path.join(RES, "T3_profession_descriptives.csv"))
log("\nProfession means (SD):\n", desc.round(3).to_string())
for g in ["gender", "nationality", "model_name"]:
    log(f"\nMeans by {g}:\n", agg.groupby(g)[MET].mean().round(4).to_string())

# ------------------------------------------------------------------ omnibus KW (16 tests, one BH family)
rows = []
for m in MET:
    for fac in ["profession", "gender", "nationality", "model_name"]:
        grps = [d[m].values for _, d in agg.groupby(fac)]
        H, p = stats.kruskal(*grps)
        rows.append(dict(measure=m, factor=fac, H=H, df=len(grps) - 1, p=p, eta2_H=max(0.0, eta2_h(H, len(grps), len(agg)))))
om = pd.DataFrame(rows); om["q"] = multipletests(om.p, method="fdr_bh")[1]; om["sig"] = om.q.apply(stars)
om.to_csv(os.path.join(RES, "T2_omnibus_kruskal.csv"), index=False)
log("\nOmnibus Kruskal-Wallis (eta2_H truncated at 0; BH over 16 tests):\n", om.round(4).to_string())

# per-model replication of the profession effect
rep = []
for m in MET:
    for mod, d in agg.groupby("model_name"):
        H, p = stats.kruskal(*[x[m].values for _, x in d.groupby("profession")])
        means = d.groupby("profession")[m].mean()
        rep.append(dict(measure=m, model=mod, eta2_H=eta2_h(H, 5, len(d)), p=p, lowest=means.idxmin(), highest=means.idxmax()))
rep = pd.DataFrame(rep); rep.to_csv(os.path.join(RES, "appendix_per_model_profession_effect.csv"), index=False)
log("\nPer-model profession effect:\n", rep.round(4).to_string())
log("\nPer-model profession means:")
for m in MET:
    log(m, "\n", agg.groupby(["model_name", "profession"])[m].mean().unstack()[P].round(3).to_string())

# ------------------------------------------------------------------ pairwise profession (40 tests, one BH family)
rows = []
for m in MET:
    for a, b in combinations(P, 2):
        x = agg.loc[agg.profession == a, m].values; y = agg.loc[agg.profession == b, m].values
        U, p = stats.mannwhitneyu(x, y, alternative="two-sided")
        lo, hi = boot_d(x, y)
        rows.append(dict(measure=m, g1=a, g2=b, mean1=x.mean(), mean2=y.mean(), d=cohen_d(x, y), ci_lo=lo, ci_hi=hi, U=U, p=p))
pw = pd.DataFrame(rows); pw["q"] = multipletests(pw.p, method="fdr_bh")[1]; pw["sig"] = pw.q.apply(stars)
pw.to_csv(os.path.join(RES, "appendix_pairwise_profession.csv"), index=False)
log("\nPairwise profession contrasts (BH over 40):\n", pw.round(4).to_string())

# ------------------------------------------------------------------ gender on P3 (3 tests)
rows = []
for a, b in [("Non-Binary", "Male"), ("Non-Binary", "Female"), ("Female", "Male")]:
    x = agg.loc[agg.gender == a, "agency_quality_score"].values; y = agg.loc[agg.gender == b, "agency_quality_score"].values
    U, p = stats.mannwhitneyu(x, y, alternative="two-sided"); lo, hi = boot_d(x, y)
    rows.append(dict(g1=a, g2=b, mean1=x.mean(), mean2=y.mean(), d=cohen_d(x, y), ci_lo=lo, ci_hi=hi, p=p))
gp = pd.DataFrame(rows); gp["q"] = multipletests(gp.p, method="fdr_bh")[1]
gp.to_csv(os.path.join(RES, "gender_P3_pairwise.csv"), index=False)
log("\nGender P3 pairwise:\n", gp.round(4).to_string())
t = agg.groupby(["profession", "gender"]).agency_quality_score.mean().unstack()
log("P3 gender x profession:\n", t.round(4).to_string(), "\nNB highest in", (t.idxmax(axis=1) == "Non-Binary").sum(), "of 5 professions")
t2 = agg.groupby(["model_name", "gender"]).agency_quality_score.mean().unstack()
log("P3 gender x model:\n", t2.round(4).to_string(), "\nNB highest in", (t2.idxmax(axis=1) == "Non-Binary").sum(), "of 3 models")

# pronoun usage by gender (per 100 words)
raw["they_rate"] = raw.story_text.str.count(r"\b[Tt]hey\b") / raw.words * 100
raw["heshe_rate"] = raw.story_text.str.count(r"\b(?:[Hh]e|[Ss]he)\b") / raw.words * 100
log("\nPronoun rates per 100 words by gender:\n", raw.groupby("gender")[["they_rate", "heshe_rate"]].mean().round(2).to_string())
tl = raw.story_text.str.lower()
raw["mentions_nonbinary"] = tl.str.contains(r"non-binary|nonbinary|they/them")
raw["mentions_male"] = tl.str.contains(r"\bmale\b|\bman\b")
raw["mentions_female"] = tl.str.contains(r"\bfemale\b|\bwoman\b")
raw["identity_acceptance_vocab"] = tl.str.contains(r"outsider|male-dominated|pronoun|identity|acceptance|accepted|inclusiv|divers")
log("Explicit identity marking by gender (share of stories):\n",
    raw.groupby("gender")[["mentions_nonbinary", "mentions_male", "mentions_female", "identity_acceptance_vocab"]].mean().round(3).to_string())
x = agg.loc[agg.model_name == "gemma3:4b", "sentiment_score"].values; y = agg.loc[agg.model_name == "llama3", "sentiment_score"].values
log("P1 Gemma vs Llama 3: d = %.2f" % cohen_d(x, y))

# ------------------------------------------------------------------ Table B6: effect sizes on units vs single stories
rows = []
for m in MET:
    rows.append(dict(measure=m, contrast="Engineer vs Janitor",
                     d_units=cohen_d(agg.loc[agg.profession == "Engineer", m].values, agg.loc[agg.profession == "Janitor", m].values),
                     d_stories=cohen_d(raw.loc[raw.profession == "Engineer", m].values, raw.loc[raw.profession == "Janitor", m].values)))
for a, b in [("Non-Binary", "Male"), ("Non-Binary", "Female")]:
    m = "agency_quality_score"
    rows.append(dict(measure=m, contrast=f"{a} vs {b}",
                     d_units=cohen_d(agg.loc[agg.gender == a, m].values, agg.loc[agg.gender == b, m].values),
                     d_stories=cohen_d(raw.loc[raw.gender == a, m].values, raw.loc[raw.gender == b, m].values)))
es = pd.DataFrame(rows); es.to_csv(os.path.join(RES, "appendix_effect_sizes_units_vs_stories.csv"), index=False)
log("\nCohen's d on units vs single stories (Table B6):\n", es.round(2).to_string())

# ------------------------------------------------------------------ additive linear models + permutation p + interactions
rng = np.random.default_rng(7)
lm_rows, int_rows = [], []
for m in MET:
    f = f"{m} ~ C(model_name) + C(gender) + C(nationality) + C(profession)"
    fit = smf.ols(f, agg).fit(); a = anova_lm(fit, typ=2); ssr = a.loc["Residual", "sum_sq"]
    for fac in ["model_name", "gender", "nationality", "profession"]:
        term = f"C({fac})"; F0 = a.loc[term, "F"]; c = 0
        for _ in range(N_PERM):
            d2 = agg.copy(); d2[fac] = rng.permutation(d2[fac].values)
            c += anova_lm(smf.ols(f, d2).fit(), typ=2).loc[term, "F"] >= F0
        lm_rows.append(dict(measure=m, factor=fac, df=int(a.loc[term, "df"]), df_resid=int(a.loc["Residual", "df"]), F=F0,
                            p_param=a.loc[term, "PR(>F)"], p_perm=(c + 1) / (N_PERM + 1),
                            partial_eta2=a.loc[term, "sum_sq"] / (a.loc[term, "sum_sq"] + ssr), R2=fit.rsquared))
    for inter in ["C(gender):C(profession)", "C(nationality):C(profession)", "C(gender):C(nationality)"]:
        full = smf.ols(f + " + " + inter, agg).fit(); tt = anova_lm(fit, full)
        int_rows.append(dict(measure=m, interaction=inter, df=int(tt.df_diff[1]), df_resid=int(tt.df_resid[1]), F=tt.F[1], p=tt["Pr(>F)"][1]))
lm = pd.DataFrame(lm_rows); lm["q_perm"] = multipletests(lm.p_perm, method="fdr_bh")[1]
lm.to_csv(os.path.join(RES, "appendix_additive_models.csv"), index=False)
it = pd.DataFrame(int_rows); it["q"] = multipletests(it.p, method="fdr_bh")[1]
it.to_csv(os.path.join(RES, "appendix_interaction_tests.csv"), index=False)
log("\nAdditive models (Type II, permutation p, BH over 16):\n", lm.round(4).to_string())
log("\nInteraction tests (BH over 12):\n", it.round(4).to_string())

# ------------------------------------------------------------------ nationality x profession cells + additive residuals
cell = agg.groupby(["nationality", "profession"]).agency_arc.mean().unstack()[P]
cell.to_csv(os.path.join(RES, "nat_x_prof_P2_cells.csv"))
log("\nP2 nationality x profession cells:\n", cell.round(3).to_string())
log("spread across nationalities per profession:", (cell.max() - cell.min()).round(3).to_dict())
def resid(df):
    c = df.groupby(["nationality", "profession"]).agency_arc.mean().unstack()[P]
    return c - c.mean(axis=1).values[:, None] - c.mean(axis=0).values[None, :] + c.values.mean()
obs = resid(agg); r = np.random.default_rng(SEED); groups = [v for _, v in agg.groupby(["nationality", "profession"])]
bs = np.stack([resid(pd.concat([v.iloc[r.integers(0, len(v), len(v))] for v in groups])).values for _ in range(N_BOOT)])
res = obs.stack().rename("residual").to_frame()
res["ci_lo"] = pd.DataFrame(np.percentile(bs, 2.5, axis=0), index=obs.index, columns=obs.columns).stack()
res["ci_hi"] = pd.DataFrame(np.percentile(bs, 97.5, axis=0), index=obs.index, columns=obs.columns).stack()
res["observed"] = cell.stack(); res["predicted"] = res.observed - res.residual
res = res.sort_values("residual"); res.to_csv(os.path.join(RES, "appendix_additive_residuals.csv"))
log("\nAdditive residuals (bootstrap 95% CI):\n", res.round(3).to_string())
log("CIs excluding zero:", int(((res.ci_lo > 0) | (res.ci_hi < 0)).sum()), "of 25")
log("marginal effects: nationality", (cell.mean(axis=1) - cell.values.mean()).round(3).to_dict(),
    "| profession", (cell.mean(axis=0) - cell.values.mean()).round(3).to_dict(), "| grand mean %.3f" % cell.values.mean())

# ------------------------------------------------------------------ H2: tone-structure association
def partial_rank(d):
    dd = d.copy(); dd["rs"] = dd.sentiment_score.rank(); dd["ra"] = dd.agency_arc.rank()
    e1 = smf.ols("rs ~ C(model_name)", dd).fit().resid; e2 = smf.ols("ra ~ C(model_name)", dd).fit().resid
    return stats.pearsonr(e1, e2)
groups = [("All units", agg)] + [(p, agg[agg.profession == p]) for p in P] + [(n, agg[agg.nationality == n]) for n in N]
rows = []
for lab, d in groups:
    rho, p = stats.spearmanr(d.sentiment_score, d.agency_arc); rp, pp = partial_rank(d)
    wm = {mod: stats.spearmanr(x.sentiment_score, x.agency_arc)[0] for mod, x in d.groupby("model_name")}
    rows.append(dict(group=lab, n=len(d), rho_pooled=rho, p_pooled=p, rho_model_adjusted=rp, p_adjusted=pp, **{f"rho_{k}": v for k, v in wm.items()}))
h2 = pd.DataFrame(rows)
h2["q_pooled"] = multipletests(h2.p_pooled, method="fdr_bh")[1]; h2["q_adjusted"] = multipletests(h2.p_adjusted, method="fdr_bh")[1]
h2.to_csv(os.path.join(RES, "T4_H2_correlations.csv"), index=False)
log("\nH2 correlations (BH over 11 per column):\n", h2.round(3).to_string())
rs, ps = stats.spearmanr(raw.sentiment_score, raw.agency_arc)
log("story-level rho = %.3f (p=%.2g)" % (rs, ps), {m: round(stats.spearmanr(x.sentiment_score, x.agency_arc)[0], 3) for m, x in raw.groupby("model_name")})
log("Engineer units, model means:\n", agg[agg.profession == "Engineer"].groupby("model_name")[["sentiment_score", "agency_arc"]].mean().round(3).to_string())

# ------------------------------------------------------------------ exploratory: opening reception scenes
def first_third(t):
    s = re.split(r"(?<=[.!?])\s+", t.strip()); return " ".join(s[:max(1, len(s) // 3)]).lower()
raw["reception"] = raw.story_text.apply(first_third).str.contains(r"welcom|greet|introduc|showed (?:him|her|them)|shown around|orientation")
log("\nOpening-third reception scene share by profession:", raw.groupby("profession").reception.mean().round(3).to_dict())
log("Act I agency / P2 by reception:\n", raw.groupby("reception")[["agency_start", "agency_arc"]].mean().round(3).to_string())
log("P2 by profession x reception:\n", raw.groupby(["profession", "reception"]).agency_arc.mean().unstack().round(3).to_string())
rb = stats.pointbiserialr(raw.reception, raw.agency_arc); log("point-biserial r = %.3f, p = %.2g" % (rb[0], rb[1]))
raw["cleaning_tools"] = raw.story_text.str.lower().str.contains(r"\bmop|\bcart\b|broom|vacuum")
log("cleaning-tool vocabulary share by profession:", raw.groupby("profession").cleaning_tools.mean().round(2).to_dict())

# ------------------------------------------------------------------ P5 (negative result)
tax = raw.taxonomy_label.value_counts(normalize=True); log("\nP5 label distribution:", tax.round(3).to_dict())
taxp = pd.crosstab(raw.profession, raw.taxonomy_label, normalize="index").reindex(P); taxp.to_csv(os.path.join(RES, "appendix_P5_labels_by_profession.csv"))
log(taxp.round(3).to_string())

with open(os.path.join(RES, "summary.txt"), "w") as fh: fh.write("\n".join(out))

# ================================================================== FIGURES
def ci95(x): return 1.96 * np.std(x, ddof=1) / np.sqrt(len(x))

# Fig 1: tone vs agency arc by profession and model
fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.1))
for ax, m, xl in [(axes[0], "sentiment_score", "P1 tone (mean signed sentiment)"), (axes[1], "agency_arc", "P2 agency arc (Act III − Act I)")]:
    for j, mod in enumerate(MOD):
        d = agg[agg.model_name == mod]
        for i, p in enumerate(P):
            v = d.loc[d.profession == p, m].values
            ax.errorbar(v.mean(), i + (j - 1) * 0.18, xerr=ci95(v), fmt=MOD_MK[mod], color=MOD_COL[mod], ms=5, capsize=2, lw=1,
                        label=MOD_LAB[mod] if i == 0 else None)
    ax.set_yticks(range(len(P))); ax.set_yticklabels(P); ax.invert_yaxis(); ax.set_xlabel(xl)
    ax.grid(axis="x", color="#dddddd", lw=0.6)
axes[0].set_title("(a) Tone", loc="left", fontsize=9, fontweight="bold")
axes[1].set_title("(b) Agency arc", loc="left", fontsize=9, fontweight="bold")
axes[1].legend(frameon=False, fontsize=8, loc="lower right")
fig.tight_layout(); _save(fig, "FigS4"); plt.close(fig)

# Fig 2: act-level human subject share by profession (pooled, mean ± 95% CI over 45 units)
fig, ax = plt.subplots(figsize=(4.6, 3.3))
cols = {"Engineer": OI[4], "Teacher": OI[0], "Nurse": OI[2], "Doctor": OI[6], "Janitor": OI[5]}
mks = {"Engineer": "o", "Teacher": "s", "Nurse": "^", "Doctor": "D", "Janitor": "v"}
for k, p in enumerate(P):
    d = agg[agg.profession == p]; xs = np.arange(3) + (k - 2) * 0.04
    ys = [d[c].mean() for c in ["agency_start", "agency_middle", "agency_end"]]
    es = [ci95(d[c].values) for c in ["agency_start", "agency_middle", "agency_end"]]
    ax.errorbar(xs, ys, yerr=es, marker=mks[p], color=cols[p], ms=5, capsize=2, lw=1.4, label=p)
ax.set_xticks(range(3)); ax.set_xticklabels(["Act I\n(opening third)", "Act II\n(middle third)", "Act III\n(closing third)"])
ax.set_ylabel("Human subject share\nsubj / (subj + obj)"); ax.grid(axis="y", color="#dddddd", lw=0.6)
ax.legend(frameon=False, fontsize=8, ncol=1, loc="lower right")
fig.tight_layout(); plt.close(fig)   # same data as Fig. 2b of the article (experiments/figures.py)

# Fig 3: P3 by gender x profession
fig, ax = plt.subplots(figsize=(6.2, 3.0))
gcol = {"Male": OI[4], "Female": OI[0], "Non-Binary": OI[6]}; gmk = {"Male": "s", "Female": "o", "Non-Binary": "D"}
for j, g in enumerate(G):
    for i, p in enumerate(P):
        v = agg.loc[(agg.profession == p) & (agg.gender == g), "agency_quality_score"].values
        ax.errorbar(i + (j - 1) * 0.2, v.mean(), yerr=ci95(v), fmt=gmk[g], color=gcol[g], ms=5, capsize=2, lw=1, label=g if i == 0 else None)
ax.set_xticks(range(len(P))); ax.set_xticklabels(P); ax.set_ylabel("P3 agency quality\n(expert / (expert + routine))")
ax.axhline(agg.agency_quality_score.mean(), color="#999999", lw=0.8, ls="--")
ax.grid(axis="y", color="#dddddd", lw=0.6); ax.legend(frameon=False, fontsize=8, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.0))
fig.tight_layout(); _save(fig, "FigS5"); plt.close(fig)

# Fig 4: nationality x profession heatmap of P2
fig, ax = plt.subplots(figsize=(5.4, 3.0))
order = cell.mean(axis=1).sort_values(ascending=False).index
cm = cell.loc[order]
im = ax.imshow(cm.values, cmap="viridis", aspect="auto", vmin=0.03, vmax=0.21)
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        v = cm.values[i, j]; ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=8, color="white" if v < 0.13 else "black")
ax.set_xticks(range(len(P))); ax.set_xticklabels(P); ax.set_yticks(range(len(order))); ax.set_yticklabels(order)
ax.spines[:].set_visible(False); cb = fig.colorbar(im, ax=ax, fraction=0.04); cb.set_label("P2 agency arc (cell mean)")
fig.tight_layout(); _save(fig, "FigS2"); plt.close(fig)

# Online Resource figures
fig, axes = plt.subplots(1, 3, figsize=(8.5, 3.0), sharey=True)
for ax, mod in zip(axes, MOD):
    d0 = agg[agg.model_name == mod]
    for k, p in enumerate(P):
        d = d0[d0.profession == p]
        ax.plot(range(3), [d[c].mean() for c in ["agency_start", "agency_middle", "agency_end"]], marker=mks[p], color=cols[p], lw=1.3, ms=4, label=p)
    ax.set_title(MOD_LAB[mod], fontsize=9); ax.set_xticks(range(3)); ax.set_xticklabels(["Act I", "Act II", "Act III"]); ax.grid(axis="y", color="#dddddd", lw=0.6)
axes[0].set_ylabel("Human subject share"); axes[2].legend(frameon=False, fontsize=7, loc="lower right")
fig.tight_layout(); _save(fig, "FigS1"); plt.close(fig)


fig, ax = plt.subplots(figsize=(6.0, 2.8))
labs = ["Professional_Authority", "Institutional_Subordination", "Adversity_Framing", "Domestic_Relegation", "Cultural_Tokenism", "Invisible_Competence"]
left = np.zeros(len(P)); tcols = [OI[4], OI[5], OI[0], OI[2], OI[6], "#999999"]
for l, c in zip(labs, tcols):
    v = taxp.get(l, pd.Series(0, index=P)).reindex(P).fillna(0).values * 100
    ax.barh(range(len(P)), v, left=left, color=c, label=l.replace("_", " ")); left += v
ax.set_yticks(range(len(P))); ax.set_yticklabels(P); ax.invert_yaxis(); ax.set_xlabel("% of stories")
ax.legend(frameon=False, fontsize=7, bbox_to_anchor=(1.0, 1.0), loc="upper left")
fig.tight_layout(); _save(fig, "FigS3"); plt.close(fig)
print("\nDone. Results in", RES, "| figures in", FIG)
