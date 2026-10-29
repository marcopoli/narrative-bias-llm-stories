"""robustness_summary.py: statistics for the robustness variants (reads results/robustness_story_level.csv)."""
import os
import numpy as np, pandas as pd
from scipy import stats
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm
HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
rob = pd.read_csv(os.path.join(RES, "robustness_story_level.csv"))

# ---------------- summaries
out = []
def log(*a):
    s = " ".join(str(x) for x in a); print(s); out.append(s)
log("Validation (story level): r(P2 stored, P2 re-implemented) = %.3f; r(P3 stored, P3 re-implemented) = %.3f; r(P1 stored, P1 re-implemented) = %.3f" % (
    np.corrcoef(rob.agency_arc, rob.p2_orig_fallback)[0, 1], np.corrcoef(rob.agency_quality_score, rob.p3_orig.fillna(0.5))[0, 1],
    np.corrcoef(rob.sentiment_score, rob.p1_recomputed)[0, 1]))
log("protagonist name found in %.1f%% of stories" % (100 * rob.protagonist_found.mean()))
for c in ["p2_orig", "p2_prot", "p2_name", "p3_prot", "p3_name"]:
    log(f"{c}: defined for {rob[c].notna().mean():.3f} of stories")
log("\nSentence classes: mean share pos %.3f, neu %.3f, neg %.3f; stories with more negative than positive sentences: %d; max share_neg %.2f" % (
    rob.share_pos.mean(), rob.share_neu.mean(), rob.share_neg.mean(), int((rob.share_neg > rob.share_pos).sum()), rob.share_neg.max()))
log("stories with no negative sentence: %.3f" % (rob.share_neg == 0).mean())
log("by profession:\n", rob.groupby("profession")[["share_pos", "share_neu", "share_neg", "tone_posneg"]].mean().round(3).to_string())
log("by model:\n", rob.groupby("model_name")[["share_pos", "share_neu", "share_neg", "tone_posneg"]].mean().round(3).to_string())

keys = ["persona_id", "model_name", "gender", "nationality", "profession"]
VARS = ["tone_posneg", "p2_orig", "p2_prot", "p2_name", "p3_prot", "p3_name"]
units = rob.groupby(keys)[VARS].mean().reset_index()
units.to_csv(os.path.join(RES, "robustness_units.csv"), index=False)
def eta2(H, k, n): return max(0.0, (H - k + 1) / (n - k))
rows = []
for v in VARS:
    u = units.dropna(subset=[v])
    for fac in ["profession", "gender", "nationality", "model_name"]:
        grp = [d[v].values for _, d in u.groupby(fac)]; H, p = stats.kruskal(*grp)
        rows.append(dict(variant=v, factor=fac, eta2_H=eta2(H, len(grp), len(u)), p=p, n_units=len(u)))
kw = pd.DataFrame(rows); kw.to_csv(os.path.join(RES, "robustness_omnibus.csv"), index=False)
log("\nOmnibus Kruskal-Wallis for robustness variants:\n", kw.pivot(index="variant", columns="factor", values="eta2_H").round(3).to_string())
log(kw.pivot(index="variant", columns="factor", values="p").map(lambda x: f"{x:.2g}").to_string())
log("\nProfession means by model (Janitor lowest?):")
for v in VARS:
    t = units.groupby(["model_name", "profession"])[v].mean().unstack()
    log(v, "| lowest per model:", t.idxmin(axis=1).to_dict(), "\n", t.round(3).to_string())
def d(a, b):
    s = np.sqrt((np.var(a, ddof=1) + np.var(b, ddof=1)) / 2); return (np.mean(a) - np.mean(b)) / s
log("\nGender contrasts on P3 variants (unit level):")
for v in ["p3_prot", "p3_name"]:
    u = units.dropna(subset=[v])
    for a, b in [("Non-Binary", "Male"), ("Non-Binary", "Female"), ("Female", "Male")]:
        x = u.loc[u.gender == a, v].values; y = u.loc[u.gender == b, v].values
        log(f"  {v} {a}-{b}: means {x.mean():.4f} vs {y.mean():.4f}, d = {d(x, y):.2f}, MWU p = {stats.mannwhitneyu(x, y).pvalue:.2g}")
    f = f"{v} ~ C(model_name) + C(gender) + C(nationality) + C(profession)"
    a = anova_lm(smf.ols(f, u).fit(), typ=2); log(f"  additive model gender: F = {a.loc['C(gender)','F']:.2f}, p = {a.loc['C(gender)','PR(>F)']:.2g}")
    t = u.groupby(["profession", "gender"])[v].mean().unstack(); log("  NB highest in", int((t.idxmax(axis=1) == "Non-Binary").sum()), "of 5 professions")
log("\nEngineer-Janitor d on P2 variants (unit level):")
for v in ["p2_orig", "p2_prot", "p2_name", "tone_posneg"]:
    u = units.dropna(subset=[v]); x = u.loc[u.profession == "Engineer", v].values; y = u.loc[u.profession == "Janitor", v].values
    log(f"  {v}: {x.mean():.3f} vs {y.mean():.3f}, d = {d(x, y):.2f}, MWU p = {stats.mannwhitneyu(x, y).pvalue:.2g}")
log("\nAct-level protagonist subject share by profession:\n", rob.groupby("profession")[["act1_prot", "act3_prot", "act1_name", "act3_name"]].mean().round(3).to_string())
with open(os.path.join(RES, "robustness_summary.txt"), "w") as fh: fh.write("\n".join(out))
