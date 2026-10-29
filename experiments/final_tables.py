"""Numbers reported in the HCIS article across all experiments, from one place.

Inputs: the main corpus database, results/scores_new.csv (score_new.py), out/ratings_*.jsonl (judge.py),
results/standard_audit.csv (standard_audit.py), results/flagged_sentences.csv (flagged_sentences.py),
results/exp_summary.json (analyze_experiments.py), results/exp45_summary.json (analyze_45.py).
Output: results/final_numbers.json and results/table2.csv. Run with the system python3.
"""
import glob, json, os, sqlite3
import numpy as np, pandas as pd
from scipy import stats
import statsmodels.formula.api as smf

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
ITEMS = ["warmth", "competence", "agency", "development", "welcome"]
AUTO = ["sentiment_score", "agency_arc", "agency_quality_score"]
LABEL = {"gemma3:4b": "Gemma 3 4B", "llama3": "Llama 3 8B", "mistral": "Mistral 7B", "gemma3:27b": "Gemma 3 27B",
         "mistral-small3.2:24b": "Mistral Small 3.2 24B", "gpt-oss:20b": "gpt-oss 20B", "gemma4:31b": "Gemma 4 31B",
         "qwen3.8:27b": "Qwen 3.8 27B", "mistral:7b": "Mistral 7B", "llama3:8b": "Llama 3 8B"}
ORDER = ["gemma3:4b", "llama3", "mistral", "gemma3:27b", "mistral-small3.2:24b", "gpt-oss:20b", "gemma4:31b", "qwen3.8:27b"]
rng = np.random.default_rng(11)

def cohen_d(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float((a.mean() - b.mean()) / np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2))

def boot_diff(a, b, n=4000):
    a, b = np.asarray(a, float), np.asarray(b, float)
    bs = [rng.choice(a, len(a)).mean() - rng.choice(b, len(b)).mean() for _ in range(n)]
    return [float(a.mean() - b.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]

def jan_contrast(u, m):
    """Janitor minus other professions on persona means: difference with 95% CI, d (others minus janitor), p."""
    j, o = u.loc[u.profession == "Janitor", m].dropna(), u.loc[u.profession != "Janitor", m].dropna()
    return {"diff": boot_diff(j, o), "d": cohen_d(o, j), "p": float(stats.mannwhitneyu(j, o).pvalue),
            "jan": float(j.mean()), "others": float(o.mean())}

# ------------------------------------------------------------------ stories and ratings
con = sqlite3.connect(os.path.join(HERE, "..", "analysis", "master_results.db"))
orig = pd.read_sql_query("""SELECT r.id, r.model_name AS model, p.nationality, p.gender, p.profession, r.persona_id,
    r.sentiment_score, r.agency_arc, r.agency_quality_score FROM results r JOIN personas p ON r.persona_id = p.id""", con)
orig["uid"] = "orig-" + orig.id.astype(str); orig["exp"] = "main"; orig["variant"] = "original"
new = pd.read_csv(os.path.join(RES, "scores_new.csv"))
D = pd.concat([orig.drop(columns="id"), new], ignore_index=True)
R = pd.concat([pd.read_json(f, lines=True) for f in glob.glob(os.path.join(HERE, "out", "ratings_*.jsonl"))])
R = R.dropna(subset=ITEMS).drop_duplicates(["uid", "judge"])
main_rater = R[R.judge == "qwen3:8b"].drop(columns="judge")
D = D.merge(main_rater, on="uid", how="left")
A = pd.read_csv(os.path.join(RES, "standard_audit.csv"))
D = D.merge(A, on="uid", how="left")
OUT = {"n_stories": {k: int(v) for k, v in D.groupby("exp").size().items()}, "n_total": int(len(D)),
       "n_rated_main_rater": int(D.agency.notna().sum())}

# ------------------------------------------------------------------ Table 2: every generator, original prompt
orig_prompt = D[(D.exp == "main") | (D.exp == "exp1")]
rows, T2 = [], {}
for m in ORDER:
    g = orig_prompt[orig_prompt.model == m]
    if not len(g): continue
    u = g.groupby(["persona_id", "profession", "gender"], as_index=False)[AUTO + ITEMS + ["regard", "tox_max"]].mean()
    r = {"stories": len(g), "rated": int(g.agency.notna().sum()), "tone": float(g.sentiment_score.mean()),
         "tone_min": float(g.sentiment_score.min())}
    for x in AUTO + ITEMS + ["regard", "tox_max"]:
        r[x] = jan_contrast(u, x)
    r["nb_vs_male_rated_agency"] = cohen_d(u.loc[u.gender == "Non-Binary", "agency"].dropna(), u.loc[u.gender == "Male", "agency"].dropna())
    r["nb_vs_male_P3"] = cohen_d(u.loc[u.gender == "Non-Binary", "agency_quality_score"], u.loc[u.gender == "Male", "agency_quality_score"])
    r["largest_rated_gap_item"] = min(ITEMS, key=lambda it: r[it]["diff"][0])
    T2[LABEL[m]] = r
    fmtp = lambda c: f"{c['d']:.2f}{'***' if c['p'] < .001 else '**' if c['p'] < .01 else '*' if c['p'] < .05 else ''}"
    rows.append({"Model": LABEL[m], "Stories": len(g), "Tone (P1)": f"{r['tone']:.2f}",
                 "Agency arc (P2) d": fmtp(r["agency_arc"]), "Agency quality (P3) d": fmtp(r["agency_quality_score"]),
                 "Rated agency gap": f"{r['agency']['diff'][0]:.2f} (d {fmtp(r['agency'])})",
                 "Rated warmth gap": f"{r['warmth']['diff'][0]:.2f} (d {fmtp(r['warmth'])})"})
pd.DataFrame(rows).to_csv(os.path.join(RES, "table2.csv"), index=False)
OUT["table2"] = T2
# Holm correction over the 40 janitor tests of Table 2 (8 models x tone, P2, P3, rated agency, rated warmth)
from statsmodels.stats.multitest import multipletests
_tests = [(m, k, r[k]["p"]) for m, r in T2.items() for k in ["sentiment_score", "agency_arc", "agency_quality_score", "agency", "warmth"]]
_q = multipletests([t[2] for t in _tests], method="holm")[1]
OUT["table2_holm"] = {f"{m}|{k}": float(q) for (m, k, _), q in zip(_tests, _q)}

# ------------------------------------------------------------------ standard audits (main corpus and all stories)
def audit(df):
    res = {"share_tone_below_0.5": float((df.sentiment_score < 0.5).mean()), "share_regard_below_0": float((df.regard < 0).mean())}
    for prof, g in df.groupby("profession"):
        res[prof] = {"regard_pos": float(g.regard_pos.mean()), "regard_neg": float(g.regard_neg.mean()),
                     "regard_neutral_other": float((1 - g.regard_pos - g.regard_neg).mean()),
                     "tox_flag_share": float((g.tox_max > 0.5).mean()), "tone": float(g.sentiment_score.mean())}
    return res
OUT["audit_main"] = audit(D[D.exp == "main"].dropna(subset=["regard"]))
allo = D[((D.exp.isin(["main", "exp1"])) | ((D.exp == "exp5") & (D.variant == "baseline")))].dropna(subset=["regard"])
OUT["audit_all_original_prompt"] = {"n": len(allo), "tox_flag_janitor": float((allo[allo.profession == "Janitor"].tox_max > 0.5).mean()),
                                    "tox_flag_others": float((allo[allo.profession != "Janitor"].tox_max > 0.5).mean()),
                                    "share_tone_below_0.5": float((allo.sentiment_score < 0.5).mean()),
                                    "share_regard_below_0": float((allo.regard < 0).mean())}
F = pd.read_csv(os.path.join(RES, "flagged_sentences.csv"))
F["label_or_cleaning"] = F.label | F.cleaning
OUT["flagged"] = {"n": len(F), "by_prof": F.groupby("profession").size().to_dict(),
                  "label_or_cleaning_janitor": float(F[F.profession == "Janitor"].label_or_cleaning.mean()),
                  "label_or_cleaning_others": float(F[F.profession != "Janitor"].label_or_cleaning.mean())}
# main corpus, persona-model units: janitor contrasts of the standard measures
um = D[D.exp == "main"].groupby(["model", "persona_id", "profession"], as_index=False)[["regard", "regard_pos", "regard_neg", "tox_max", "sentiment_score"]].mean()
OUT["audit_main_contrasts"] = {x: jan_contrast(um, x) for x in ["regard", "regard_pos", "regard_neg", "tox_max", "sentiment_score"]}

# ------------------------------------------------------------------ rater reliability
E = json.load(open(os.path.join(RES, "exp_summary.json")))
OUT["agreement"] = E["exp3"].get("agreement")
OUT["convergent_qwen"] = {k: v for k, v in E["exp3"]["qwen3:8b"].items() if k.startswith("rho_")}
OUT["granite_main"] = {it: {"gap": E["exp3"]["granite3.3:8b"][f"{it}_jan_gap"], "d": E["exp3"]["granite3.3:8b"][f"{it}_d_jan"]} for it in ITEMS}
OUT["qwen_main"] = {it: {"gap": E["exp3"]["qwen3:8b"][f"{it}_jan_gap"], "d": E["exp3"]["qwen3:8b"][f"{it}_d_jan"],
                         "by_prof": E["exp3"]["qwen3:8b"][f"{it}_by_prof"]} for it in ITEMS}
big = R[R.judge == "qwen3:32b"]
if len(big):
    b = big.merge(main_rater, on="uid", suffixes=("_big", ""))
    b = b.merge(D[["uid", "model", "persona_id", "profession"]], on="uid")
    res = {"n": len(b), "story_rho": {it: float(stats.spearmanr(b[it + "_big"], b[it])[0]) for it in ITEMS}}
    ub = b.groupby(["model", "persona_id", "profession"], as_index=False)[[it + "_big" for it in ITEMS]].mean()
    res["jan_big"] = {it: jan_contrast(ub, it + "_big") for it in ITEMS}
    OUT["robust_rater"] = res

# ------------------------------------------------------------------ prompt variants (Exp. 2 in the old numbering)
p2 = D[D.exp == "exp2"]
OUT["prompts"] = {}
for v, g in p2.groupby("variant"):
    u = g.groupby(["persona_id", "profession"], as_index=False)[AUTO + ITEMS].mean()
    OUT["prompts"][v] = {"tone": float(g.sentiment_score.mean()), **{x: jan_contrast(u, x) for x in ["agency_arc", "agency", "warmth"]}}
OUT["prompts"]["interaction"] = {k: E["exp2"][k] for k in ["interaction_F", "interaction_p", "interaction_df"]}
# does the prompt change the size of the janitor gap? (F test of the variant x janitor interaction on persona units)
u2 = p2.groupby(["variant", "persona_id", "profession"], as_index=False)[AUTO + ITEMS].mean(); u2["janitor"] = (u2.profession == "Janitor").astype(int)
for x in ["agency_arc", "agency", "warmth"]:
    full = smf.ols(f"{x} ~ C(variant) * janitor", data=u2).fit(); f = full.compare_f_test(smf.ols(f"{x} ~ C(variant) + janitor", data=u2).fit())
    OUT["prompts"]["interaction_" + x] = {"F": float(f[0]), "p": float(f[1]), "df": [int(f[2]), int(full.df_resid)]}

# ------------------------------------------------------------------ occupations and mitigation
X = json.load(open(os.path.join(RES, "exp45_summary.json")))
OUT["exp4"] = {m: {k: v for k, v in r.items() if k.startswith(("rho_", "beta_", "janitor_rank", "n_"))} for m, r in X["exp4"].items()}
OUT["exp4_occ_means_pooled"] = X["exp4"]["pooled"]["occupation_means"]
E5 = {}
for model, r in X["exp5"].items():
    E5[LABEL[model]] = {v: {"words": x["words"], "similarity": x.get("cross_job_similarity"),
                            **{m: {"mean": x[m]["mean"], "gap": x[m]["jan_gap"], "d": x[m]["d"], "change": x[m].get("gap_change")}
                               for m in ["sentiment_score", "agency_arc", "agency_quality_score"] + ITEMS if m in x}}
                        for v, x in r.items()}
# Gemma 4 wrote only the equal-agency condition in Exp. 5. Its baseline consists of its Exp. 4 stories of the same five
# professions, which used the original prompt with the same seeds (repetitions 1 and 2).
g4p = D[(D.exp == "exp5") & (D.model == "gemma4:31b")]
if len(g4p) >= 250 and g4p.agency.notna().sum() >= 250:
    base = D[(D.exp == "exp4") & (D.model == "gemma4:31b") & D.profession.isin(g4p.profession.unique())]
    X4 = X["exp5"].get("gemma4:31b", {}).get("parity", {})
    r, us = {}, []
    for name, g in [("baseline", base), ("parity", g4p)]:
        u = g.groupby(["profession", "gender", "nationality"], as_index=False)[AUTO + ITEMS].mean()
        us.append(u.assign(variant=name))
        r[name] = {"stories": len(g), "tone": float(g.sentiment_score.mean()),
                   **{m: {"mean": float(u[m].mean()), **jan_contrast(u, m)} for m in AUTO + ITEMS}}
    uu = pd.concat(us); uu["janitor"] = (uu.profession == "Janitor").astype(int)
    for m in AUTO + ITEMS:
        fit = smf.ols(f"{m} ~ C(variant, Treatment('baseline')) * janitor", data=uu).fit()
        k = [p for p in fit.params.index if ":janitor" in p][0]
        r["parity"][m]["change"] = [float(fit.params[k]), float(fit.pvalues[k])]
    r["parity"]["words"], r["parity"]["similarity"] = X4.get("words"), X4.get("cross_job_similarity")
    E5["Gemma 4 31B"] = r
OUT["exp5"] = E5

# ------------------------------------------------------------------ pairwise judgements (Exp. 5)
pf = glob.glob(os.path.join(HERE, "out", "pairs_*.jsonl"))
if pf:
    pr = pd.concat([pd.read_json(f, lines=True) for f in pf])
    pr["key"] = pr.pid.str.rsplit("|", n=1).str[0]
    dec = pr.answer.isin(["A", "B"])
    OUT["pairs_position_A"] = float((pr.answer == "A").sum() / dec.sum())
    w = pr.pivot_table(index=["model", "variant", "key"], columns="order", values="janitor_more", aggfunc="first").reset_index()
    def verdict(r):
        a, b = r.get("JO"), r.get("OJ")
        if pd.isna(a) or pd.isna(b) or a != b: return "tie"
        return "janitor" if a else "other"
    w["v"] = w.apply(verdict, axis=1)
    OUT["pairs"] = {}
    for (m, v), g in w.groupby(["model", "variant"]):
        j, o = int((g.v == "janitor").sum()), int((g.v == "other").sum())
        raw = pr[(pr.model == m) & (pr.variant == v)].janitor_more.dropna().astype(bool)
        OUT["pairs"].setdefault(LABEL[m], {})[v] = {"pairs": len(g), "consistent": (j + o) / len(g), "janitor": j, "other": o,
            "janitor_share": j / (j + o) if j + o else None, "p": float(stats.binomtest(j, j + o, 0.5).pvalue) if j + o else None,
            "ci": list(map(float, stats.binomtest(j, j + o).proportion_ci())) if j + o else None,
            "raw_share": float(raw.mean()), "raw_n": int(len(raw))}

json.dump(OUT, open(os.path.join(RES, "final_numbers.json"), "w"), indent=1)
print(pd.DataFrame(rows).to_string(index=False))
print("stories:", OUT["n_stories"], "total", OUT["n_total"], "rated", OUT["n_rated_main_rater"])
