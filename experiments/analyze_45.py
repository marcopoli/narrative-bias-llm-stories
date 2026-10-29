"""Analysis of Experiment 4 (occupations and human stereotype norms) and Experiment 5 (mitigation).

Inputs: results/scores_new.csv (score_new.py), out/ratings_judge_input_45_qwen3-8b.jsonl (judge.py),
norms/occupation_norms_friehs2022.csv. Output: results/exp45_summary.json. Run with the system python3.
"""
import json, os
import numpy as np, pandas as pd
from scipy import stats
import statsmodels.formula.api as smf
from sklearn.feature_extraction.text import TfidfVectorizer

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
ITEMS = ["warmth", "competence", "agency", "development", "welcome"]
AUTO = ["sentiment_score", "agency_arc", "agency_quality_score"]
rng = np.random.default_rng(7)

def boot_diff(a, b, n=2000):
    a, b = np.asarray(a, float), np.asarray(b, float)
    bs = [rng.choice(a, len(a)).mean() - rng.choice(b, len(b)).mean() for _ in range(n)]
    return [float(a.mean() - b.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]

def cohen_d(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float((a.mean() - b.mean()) / np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2))

S = pd.read_csv(os.path.join(RES, "scores_new.csv"))
S = S[S.exp.isin(["exp4", "exp5"])]
rf = [os.path.join(HERE, "out", f"ratings_judge_input_{t}_qwen3-8b.jsonl") for t in ("45", "exp5g")]
rf = [f for f in rf if os.path.exists(f)]
if rf:
    S = S.merge(pd.concat([pd.read_json(f, lines=True) for f in rf]).drop(columns="judge"), on="uid", how="left")
else:
    for it in ITEMS: S[it] = np.nan
stories = {}
for f in os.listdir(os.path.join(HERE, "out")):
    if f.startswith(("stories_exp4", "stories_exp5")):
        tag = f[len("stories_"):-len(".jsonl")]
        for i, l in enumerate(open(os.path.join(HERE, "out", f))): stories[f"{tag}-{i}"] = json.loads(l)["story_text"]
S["text"] = S.uid.map(stories)
MEAS = AUTO + [it for it in ITEMS if S[it].notna().any()]
OUT = {}

# ------------------------------------------------------------------ Experiment 4
E4 = {}
norms = pd.read_csv(os.path.join(HERE, "norms", "occupation_norms_friehs2022.csv")).set_index("occupation")
norms = norms.rename(columns={"warmth": "h_warmth", "competence": "h_competence"})
prestige = pd.read_csv(os.path.join(HERE, "norms", "occupation_prestige_hughes2022.csv")).set_index("occupation").prestige
e4 = S[S.exp == "exp4"]
if len(e4):
    for model, g in list(e4.groupby("model")) + [("pooled", e4)]:
        # unit: occupation x gender x nationality (seeds shared across occupations)
        u = g.groupby(["model", "profession", "gender", "nationality"], as_index=False)[MEAS].mean()
        occ = u.groupby("profession")[MEAS].mean()
        res = {"n_stories": len(g), "occupation_means": occ.round(3).to_dict(orient="index")}
        res["janitor_rank_lowest"] = {m: int(occ[m].rank().loc["Janitor"]) for m in MEAS if "Janitor" in occ.index}
        on = occ.join(norms, how="inner")
        res["n_occ_with_norms"] = len(on)
        for m in MEAS:
            for h in ["competence", "warmth"]:
                rho, p = stats.spearmanr(on["h_" + h], on[m]); res[f"rho_{h}_{m}"] = [float(rho), float(p)]
        # unit-level regression with occupations as clusters: story measure on the two human norms (z-scores)
        un = u.join(norms[["h_warmth", "h_competence"]], on="profession", how="inner")
        for c in ["h_warmth", "h_competence"]: un[c] = (un[c] - un[c].mean()) / un[c].std()
        for m in MEAS:
            un["y"] = (un[m] - un[m].mean()) / un[m].std()
            fit = smf.ols("y ~ h_competence + h_warmth" + (" + C(model)" if model == "pooled" else ""), data=un) \
                     .fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(un.profession)[0]})
            res[f"beta_{m}"] = {k: [float(fit.params[k]), float(fit.pvalues[k])] for k in ["h_competence", "h_warmth"]}
        # occupational status: prestige ratings cover all 13 occupations (janitor included)
        op = occ.join(prestige, how="inner")
        for m in MEAS:
            rho, p = stats.spearmanr(op.prestige, op[m]); res[f"rho_prestige_{m}"] = [float(rho), float(p)]
            rho, p = stats.spearmanr(op.drop("Janitor").prestige, op.drop("Janitor")[m]); res[f"rho_prestige_nojan_{m}"] = [float(rho), float(p)]
        up = u.join(prestige, on="profession", how="inner")
        up["z_prestige"] = (up.prestige - up.prestige.mean()) / up.prestige.std()
        for m in MEAS:
            up["y"] = (up[m] - up[m].mean()) / up[m].std()
            fit = smf.ols("y ~ z_prestige" + (" + C(model)" if model == "pooled" else ""), data=up) \
                     .fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(up.profession)[0]})
            res[f"beta_prestige_{m}"] = [float(fit.params["z_prestige"]), float(fit.pvalues["z_prestige"])]
        # paired janitor contrast: same seed, other occupations
        if "Janitor" in occ.index:
            g2 = g.assign(key=g.seed.astype(str) + g.model)
            jan = g2[g2.profession == "Janitor"].set_index("key")[MEAS]
            oth = g2[g2.profession != "Janitor"].groupby("key")[MEAS].mean()
            pair = (jan - oth.reindex(jan.index)).dropna(how="all")
            res["paired_jan_minus_others"] = {m: [float(pair[m].mean()), float(stats.wilcoxon(pair[m].dropna())[1])] for m in MEAS}
        E4[model] = res
OUT["exp4"] = E4

# ------------------------------------------------------------------ Experiment 5
E5 = {}
e5 = S[S.exp == "exp5"]
VARS = ["baseline", "parity", "blind", "revise"]
for model, g in e5.groupby("model"):
    res = {}
    u = g.groupby(["variant", "persona_id", "profession", "gender", "nationality"], as_index=False)[MEAS + ["words"]].mean()
    for v in VARS:
        uv = u[u.variant == v]
        if len(uv) < 75: continue
        r = {"n_units": len(uv), "words": float(uv.words.mean())}
        for m in MEAS:
            j, o = uv.loc[uv.profession == "Janitor", m], uv.loc[uv.profession != "Janitor", m]
            r[m] = {"mean": float(uv[m].mean()), "jan_gap": boot_diff(j, o), "d": cohen_d(o, j),
                    "by_prof": uv.groupby("profession")[m].mean().round(3).to_dict()}
        # side effect: are stories for different jobs more alike? TF-IDF cosine between stories of the same
        # seed and persona combination but different jobs (higher = less distinct)
        gv = g[g.variant == v]
        if gv.text.notna().all():
            X = TfidfVectorizer(stop_words="english", min_df=2).fit_transform(gv.text)
            gv = gv.reset_index(drop=True); sims = []
            for _, idx in gv.groupby(["seed", "gender", "nationality"]).groups.items():
                idx = list(idx)
                if len(idx) > 1:
                    M = (X[idx] @ X[idx].T).toarray(); sims.append(M[np.triu_indices(len(idx), 1)].mean())
            r["cross_job_similarity"] = float(np.mean(sims))
        res[v] = r
    # does the janitor gap shrink compared with the baseline? (variant x janitor on persona means)
    u["janitor"] = (u.profession == "Janitor").astype(int)
    for v in VARS[1:]:
        uu = u[u.variant.isin(["baseline", v])]
        if uu.variant.nunique() < 2: continue
        for m in MEAS:
            fit = smf.ols(f"{m} ~ C(variant, Treatment('baseline')) * janitor", data=uu).fit()
            k = [p for p in fit.params.index if ":janitor" in p][0]
            res[v][m]["gap_change"] = [float(fit.params[k]), float(fit.pvalues[k])]
    E5[model] = res
OUT["exp5"] = E5

json.dump(OUT, open(os.path.join(RES, "exp45_summary.json"), "w"), indent=1)

# short console report
for model, r in E4.items():
    print(f"E4 {model}: n={r['n_stories']}, janitor rank {r.get('janitor_rank_lowest')}")
    for m in MEAS:
        print(f"   {m:22s} prestige rho={r[f'rho_prestige_{m}'][0]:+.2f} p={r[f'rho_prestige_{m}'][1]:.3f} (w/o janitor {r[f'rho_prestige_nojan_{m}'][0]:+.2f} p={r[f'rho_prestige_nojan_{m}'][1]:.3f}) beta={r[f'beta_prestige_{m}'][0]:+.2f} p={r[f'beta_prestige_{m}'][1]:.3f}")
        print(f"   {m:22s} rho(C)={r[f'rho_competence_{m}'][0]:+.2f} p={r[f'rho_competence_{m}'][1]:.3f}  rho(W)={r[f'rho_warmth_{m}'][0]:+.2f} p={r[f'rho_warmth_{m}'][1]:.3f}"
              f"  beta C={r[f'beta_{m}']['h_competence'][0]:+.2f} (p={r[f'beta_{m}']['h_competence'][1]:.3f}) W={r[f'beta_{m}']['h_warmth'][0]:+.2f} (p={r[f'beta_{m}']['h_warmth'][1]:.3f})")
for model, r in E5.items():
    for v, x in r.items():
        cells = []
        for m in MEAS:
            chg = x[m].get("gap_change")
            cells.append(f"{m[:8]} gap={x[m]['jan_gap'][0]:+.3f}" + (f" (chg p={chg[1]:.3f})" if chg else ""))
        print(f"E5 {model} {v:8s} words={x['words']:.0f} sim={x.get('cross_job_similarity', float('nan')):.3f} " + " ".join(cells))
