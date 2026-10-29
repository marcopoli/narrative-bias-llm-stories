"""Analysis of the follow-up experiments.
Exp. 1: newer and larger models (original prompt). Exp. 2: prompt variants (Mistral 7B). Exp. 3: LLM ratings of the
main character (warmth, competence, agency, development, welcome), persona labels masked.
Inputs: results/scores_new.csv (score_new.py), out/ratings_*.jsonl (judge.py), the main corpus database.
Outputs: results/exp_summary.json (numbers used in the article), results/supplement_experiments.md (Online Resource 1,
Section S4), ../figures/Fig4.{png,pdf,eps}.
"""
import glob, json, os, sqlite3
import numpy as np, pandas as pd
from scipy import stats
import statsmodels.formula.api as smf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results"); os.makedirs(RES, exist_ok=True)
FIG = os.path.abspath(os.path.join(HERE, "..", "figures"))
P = ["Engineer", "Teacher", "Nurse", "Doctor", "Janitor"]
G = ["Male", "Female", "Non-Binary"]
MEAS = {"sentiment_score": "P1 Tone", "agency_arc": "P2 Agency arc", "agency_quality_score": "P3 Agency quality",
        "concreteness_score": "P4 Lexical expertise"}
ITEMS = ["warmth", "competence", "agency", "development", "welcome"]
NAMES = {"gemma3:4b": "Gemma 3 4B", "llama3": "Llama 3 8B", "mistral": "Mistral 7B", "gemma3:27b": "Gemma 3 27B",
         "mistral-small3.2:24b": "Mistral Small 3.2 24B", "gpt-oss:20b": "gpt-oss 20B", "gemma4:31b": "Gemma 4 31B",
         "qwen3.8:27b": "Qwen 3.8 27B", "mistral:7b": "Mistral 7B (new run)"}
rng = np.random.default_rng(42)
S = {}   # summary numbers

def eta2_h(groups):
    h, p = stats.kruskal(*groups); k = len(groups); n = sum(len(g) for g in groups)
    return max(0.0, (h - k + 1) / (n - k)), p
def cohen_d(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return (a.mean() - b.mean()) / np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
def boot_diff(a, b, n=2000):
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = [rng.choice(a, len(a)).mean() - rng.choice(b, len(b)).mean() for _ in range(n)]
    return float(np.mean(a) - np.mean(b)), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))
def fmt(x, k=3): return f"{x:.{k}f}".replace("-", "−")
def fp(p): return "< 0.001" if p < .001 else f"{p:.3f}"
def md_table(df):
    cols = list(df.columns)
    out = ["| " + " | ".join(cols) + " |", "|" + "|".join([":--"] + [":--:"] * (len(cols) - 1)) + "|"]
    out += ["| " + " | ".join(str(v) for v in r) + " |" for r in df.values]
    return "\n".join(out)

# ------------------------------------------------------------------ data
con = sqlite3.connect(os.path.join(HERE, "..", "analysis", "master_results.db"))
orig = pd.read_sql_query("""SELECT r.id, r.model_name AS model, p.nationality, p.gender, p.profession, r.persona_id,
    r.sentiment_score, r.agency_arc, r.agency_quality_score, r.concreteness_score, r.story_text
    FROM results r JOIN personas p ON r.persona_id = p.id""", con)
orig["uid"] = "orig-" + orig.id.astype(str); orig["variant"] = "original"; orig["exp"] = "main"
new = pd.read_csv(os.path.join(RES, "scores_new.csv")) if os.path.exists(os.path.join(RES, "scores_new.csv")) else pd.DataFrame()
def units(df):
    return df.groupby(["exp", "variant", "model", "persona_id", "nationality", "gender", "profession"], as_index=False)[list(MEAS)].mean()

def model_summary(df, label):
    """Profession and gender effects for one model (unit = persona mean)."""
    u = units(df); r = {"label": label, "stories": len(df), "per_persona": round(len(df) / 75, 1)}
    for m in MEAS:
        e, p = eta2_h([u.loc[u.profession == q, m] for q in P]); r[f"{m}_eta2_prof"], r[f"{m}_p_prof"] = e, p
        means = u.groupby("profession")[m].mean(); r[f"{m}_lowest"] = means.idxmin()
        r[f"{m}_janitor"] = means["Janitor"]; r[f"{m}_others"] = u.loc[u.profession != "Janitor", m].mean()
    r["d_eng_jan_P2"] = cohen_d(u.loc[u.profession == "Engineer", "agency_arc"], u.loc[u.profession == "Janitor", "agency_arc"])
    r["d_others_jan_P2"] = cohen_d(u.loc[u.profession != "Janitor", "agency_arc"], u.loc[u.profession == "Janitor", "agency_arc"])
    r["p_others_jan_P2"] = stats.mannwhitneyu(u.loc[u.profession != "Janitor", "agency_arc"], u.loc[u.profession == "Janitor", "agency_arc"]).pvalue
    for g in ["Male", "Female"]:
        a, b = u.loc[u.gender == "Non-Binary", "agency_quality_score"], u.loc[u.gender == g, "agency_quality_score"]
        r[f"d_nb_{g}_P3"], r[f"p_nb_{g}_P3"] = cohen_d(a, b), stats.mannwhitneyu(a, b).pvalue
    r["P3_by_gender"] = u.groupby("gender").agency_quality_score.mean().reindex(G).round(3).tolist()
    r["p_gender_P2"] = eta2_h([u.loc[u.gender == g, "agency_arc"] for g in G])[1]
    r["tone_mean"] = df.sentiment_score.mean(); r["tone_min"] = df.sentiment_score.min()
    if "neg_share" in df: r["neg_sentence_share"] = df.neg_share.mean()
    if "reception" in df: r["reception_by_prof"] = df.groupby("profession").reception.mean().reindex(P).round(2).tolist()
    return r

# ------------------------------------------------------------------ Experiment 1: newer and larger models
exp1 = new[new.exp == "exp1"] if len(new) else new
E1 = [model_summary(orig[orig.model == m], NAMES[m]) for m in ["gemma3:4b", "llama3", "mistral"]]
complete = lambda d: d.persona_id.nunique() == 75 and d.groupby("persona_id").size().min() >= 2   # only fully generated runs
E1 += [model_summary(exp1[exp1.model == m], NAMES[m]) for m in exp1.model.unique() if complete(exp1[exp1.model == m])] if len(exp1) else []
exp1 = exp1[exp1.model.isin([m for m in exp1.model.unique() if complete(exp1[exp1.model == m])])] if len(exp1) else exp1
S["exp1"] = E1
if len(exp1):
    u1 = units(exp1)
    S["exp1_tone_eta2_model"] = eta2_h([u1.loc[u1.model == m, "sentiment_score"] for m in u1.model.unique()])[0]
    S["exp1_tone_eta2_prof"] = eta2_h([u1.loc[u1.profession == q, "sentiment_score"] for q in P])[0]
    S["exp1_neg_story_share"] = float((exp1.sentiment_score < 0).mean())

# ------------------------------------------------------------------ Experiment 2: prompt variants (Mistral 7B)
exp2 = new[new.exp == "exp2"] if len(new) else new
if len(exp2) and not all(complete(exp2[exp2.variant == v]) for v in ["original", "ordinary_day", "development"]): exp2 = exp2.iloc[0:0]
E2 = {}
if len(exp2):
    u2 = units(exp2); u2["janitor"] = (u2.profession == "Janitor").astype(int)
    for v in ["original", "ordinary_day", "development"]:
        uv, sv = u2[u2.variant == v], exp2[exp2.variant == v]
        gap = boot_diff(uv.loc[uv.janitor == 1, "agency_arc"], uv.loc[uv.janitor == 0, "agency_arc"])
        E2[v] = dict(stories=len(sv), janitor_gap_P2=gap, P2_by_prof=uv.groupby("profession").agency_arc.mean().reindex(P).round(3).tolist(),
                     eta2_prof_P2=eta2_h([uv.loc[uv.profession == q, "agency_arc"] for q in P])[0],
                     reception_by_prof=sv.groupby("profession").reception.mean().reindex(P).round(2).tolist(),
                     act1_by_prof=sv.groupby("profession").agency_start.mean().reindex(P).round(3).tolist(),
                     tone=sv.sentiment_score.mean(), P3_jan_gap=boot_diff(uv.loc[uv.janitor == 1, "agency_quality_score"], uv.loc[uv.janitor == 0, "agency_quality_score"]))
    fit = smf.ols("agency_arc ~ C(variant) * janitor", data=u2).fit()
    base = smf.ols("agency_arc ~ C(variant) + janitor", data=u2).fit()
    f = fit.compare_f_test(base); E2["interaction_F"], E2["interaction_p"], E2["interaction_df"] = float(f[0]), float(f[1]), (int(f[2]), int(fit.df_resid))
    S["exp2"] = E2

# ------------------------------------------------------------------ Experiment 3: LLM ratings
def load_ratings(pattern):
    fs = glob.glob(os.path.join(HERE, "out", pattern))
    return pd.concat([pd.read_json(f, lines=True) for f in fs]) if fs else pd.DataFrame()
R = load_ratings("ratings_*.jsonl")
E3 = {}
if len(R):
    R = R.dropna(subset=ITEMS)
    meta = pd.concat([orig[["uid", "model", "gender", "nationality", "profession", "persona_id", "agency_arc", "agency_quality_score", "sentiment_score"]].assign(exp="main", variant="original"),
                      new[["uid", "exp", "variant", "model", "gender", "nationality", "profession", "persona_id", "agency_arc", "agency_quality_score", "sentiment_score"]] if len(new) else pd.DataFrame()])
    R = R.merge(meta, on="uid", how="left")
    for judge in R.judge.unique():
        r = R[(R.judge == judge) & (R.exp == "main")]
        if not len(r): continue
        ru = r.groupby(["model", "persona_id", "profession", "gender"], as_index=False)[ITEMS + ["agency_arc", "agency_quality_score", "sentiment_score"]].mean()
        res = {"n_stories": len(r), "n_units": len(ru)}
        for it in ITEMS:
            res[f"{it}_by_prof"] = ru.groupby("profession")[it].mean().reindex(P).round(2).tolist()
            res[f"{it}_by_gender"] = ru.groupby("gender")[it].mean().reindex(G).round(2).tolist()
            res[f"{it}_eta2_prof"], res[f"{it}_p_prof"] = eta2_h([ru.loc[ru.profession == q, it] for q in P])
            res[f"{it}_eta2_model"] = eta2_h([ru.loc[ru.model == m, it] for m in ru.model.unique()])[0]
            res[f"{it}_jan_gap"] = boot_diff(ru.loc[ru.profession == "Janitor", it], ru.loc[ru.profession != "Janitor", it])
            res[f"{it}_d_jan"] = cohen_d(ru.loc[ru.profession != "Janitor", it], ru.loc[ru.profession == "Janitor", it])
            res[f"{it}_nb_vs_m_d"] = cohen_d(ru.loc[ru.gender == "Non-Binary", it], ru.loc[ru.gender == "Male", it])
            res[f"{it}_nb_vs_f_d"] = cohen_d(ru.loc[ru.gender == "Non-Binary", it], ru.loc[ru.gender == "Female", it])
        # convergent validity with the automatic measures (unit level)
        for it, m in [("development", "agency_arc"), ("agency", "agency_arc"), ("competence", "agency_quality_score"),
                      ("agency", "agency_quality_score"), ("warmth", "sentiment_score"), ("welcome", "agency_arc")]:
            rho, p = stats.spearmanr(ru[it], ru[m]); res[f"rho_{it}_{m}"] = (float(rho), float(p))
        E3[judge] = res
    # agreement between the two raters on the same stories
    # the two raters of the full main-corpus sample (the larger rater saw only a subsample and is analysed in final_tables.py)
    judges = [j for j in ("qwen3:8b", "granite3.3:8b") if (R[(R.judge == j) & (R.exp == "main")]).shape[0]]
    if len(judges) >= 2:
        a = R[(R.judge == judges[0]) & (R.exp == "main")].set_index("uid")[ITEMS]
        b = R[(R.judge == judges[1]) & (R.exp == "main")].set_index("uid")[ITEMS]
        both = a.join(b, lsuffix="_a", rsuffix="_b", how="inner")
        agr = {"n": len(both)}
        for it in ITEMS:
            agr[it] = float(stats.spearmanr(both[it + "_a"], both[it + "_b"])[0])
        # unit level
        ua = R[(R.judge == judges[0]) & (R.exp == "main")].groupby(["model", "persona_id"])[ITEMS].mean()
        ub = R[(R.judge == judges[1]) & (R.exp == "main")].groupby(["model", "persona_id"])[ITEMS].mean()
        uu = ua.join(ub, lsuffix="_a", rsuffix="_b", how="inner")
        agr["units"] = {it: float(stats.spearmanr(uu[it + "_a"], uu[it + "_b"])[0]) for it in ITEMS}
        agr["judges"] = judges
        E3["agreement"] = agr
    # ratings of the new stories (Exp. 1 and 2) by the main rater
    rn = R[R.exp.isin(["exp1", "exp2"])]
    if len(rn):
        E3["new"] = {f"{e}|{m}|{v}": {it: boot_diff(g.loc[g.profession == "Janitor", it], g.loc[g.profession != "Janitor", it]) for it in ITEMS}
                     for (e, m, v), g in rn.groupby(["exp", "model", "variant"])}
    S["exp3"] = E3

json.dump(S, open(os.path.join(RES, "exp_summary.json"), "w"), indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
print(json.dumps(S, indent=1, default=str)[:6000])
