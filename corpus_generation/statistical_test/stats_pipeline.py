"""
stats_pipeline.py — HCNLP Narrative Bias: Complete Statistical Analysis
Stages 0–5: Assumptions → Omnibus → Pairwise+FDR → Correlation → Categorical → Cross-model
Primary analysis: persona-aggregated (n=225). Robustness on full n=4500 where noted.
Outputs all results to data/stats_results/ as CSVs + paper-ready stdout.

Run from scripts/ folder:
    python stats_pipeline.py
"""

import sqlite3, os, sys
import numpy as np
import pandas as pd
from scipy import stats
from itertools import combinations

try:
    from pingouin import multicomp
    HAS_PINGOUIN = True
except ImportError:
    HAS_PINGOUIN = False
    print("WARNING: pingouin not found. Using manual BH correction.")
    print("Install: pip install pingouin --break-system-packages\n")

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DB_PATH     = os.path.join(BASE_DIR, '..', '..', 'analysis', 'master_results.db')
OUT_DIR     = os.path.join(BASE_DIR, '..', 'data', 'stats_results')
os.makedirs(OUT_DIR, exist_ok=True)
np.random.seed(42)

CONTINUOUS = [
    "sentiment_score", "sentiment_volatility",
    "agency_arc", "agency_quality_score", "subordination_score",
    "concreteness_score", "dialogue_score", "resolution_score",
]
PROFESSIONS   = ["Engineer", "Doctor", "Nurse", "Teacher", "Janitor"]
GENDERS       = ["Male", "Female", "Non-Binary"]
NATIONALITIES = ["American", "German", "Indian", "Nigerian", "Chinese"]
MODELS        = ["gemma3:4b", "llama3", "mistral"]
GROUPINGS     = [("profession", PROFESSIONS), ("gender", GENDERS),
                 ("nationality", NATIONALITIES), ("model_name", MODELS)]


# ─────────────────────────────────────────────────────────────────────────────
# DATA
# ─────────────────────────────────────────────────────────────────────────────

def load():
    conn = sqlite3.connect(DB_PATH)
    raw = pd.read_sql_query("""
        SELECT r.id, r.model_name, r.persona_id,
               p.gender, p.nationality, p.profession,
               r.sentiment_score, r.sentiment_volatility,
               r.agency_start, r.agency_middle, r.agency_end, r.agency_arc,
               r.agency_quality_score, r.subordination_score,
               r.concreteness_score, r.taxonomy_label, r.topic_bias,
               r.dialogue_score, r.resolution_score
        FROM results r JOIN personas p ON r.persona_id = p.id
    """, conn)
    conn.close()
    print(f"Raw rows: {len(raw)}")

    # Persona × model aggregation → n=225 primary analysis units
    agg = (raw.groupby(["persona_id", "model_name", "gender", "nationality", "profession"])
              .agg(**{m: (m, 'mean') for m in CONTINUOUS},
                   taxonomy_label=('taxonomy_label',
                                   lambda x: x.mode().iloc[0] if len(x) > 0 else np.nan),
                   topic_bias=('topic_bias',
                               lambda x: x.mode().iloc[0] if len(x) > 0 else np.nan))
              .reset_index())
    print(f"Aggregated: {len(agg)} persona×model units\n")
    return raw, agg


# ─────────────────────────────────────────────────────────────────────────────
# UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def cohens_d(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    s = np.sqrt((np.std(a, ddof=1)**2 + np.std(b, ddof=1)**2) / 2)
    return (np.mean(a) - np.mean(b)) / s if s > 0 else 0.0

def bootstrap_d_ci(a, b, n_boot=1000):
    rng = np.random.default_rng(42)
    boot = [cohens_d(rng.choice(a, len(a), replace=True),
                     rng.choice(b, len(b), replace=True)) for _ in range(n_boot)]
    return tuple(np.round(np.percentile(boot, [2.5, 97.5]), 4))

def bh(pvals):
    pvals = np.asarray(pvals, float)
    n = len(pvals)
    if HAS_PINGOUIN:
        _, q = multicomp(pvals, method='fdr_bh')
        return np.minimum(np.asarray(q), 1.0)
    order = np.argsort(pvals)
    q = np.empty(n)
    for rank, idx in enumerate(order, 1):
        q[idx] = pvals[idx] * n / rank
    for i in range(n - 2, -1, -1):
        q[order[i]] = min(q[order[i]], q[order[i + 1]])
    return np.minimum(q, 1.0)

def kw_eta2(H, k, n):
    return round(max(0.0, (H - k + 1) / (n - k)), 4)

def sig_stars(p):
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"

def effect_label(d):
    d = abs(d)
    return "large" if d >= 0.8 else "medium" if d >= 0.5 else "small" if d >= 0.2 else "negligible"


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 0 — ASSUMPTIONS
# ─────────────────────────────────────────────────────────────────────────────

def stage0(agg):
    print("=" * 70)
    print("  STAGE 0 — Normality (Shapiro-Wilk) & Variance (Levene's)")
    print("  Note: many metrics will be non-normal due to bimodality/skew.")
    print("  We use KW throughout Stage 1 as the safe choice.")
    print("=" * 70)
    records = []
    rng = np.random.default_rng(42)   # fixed seed: Shapiro subsample is reproducible
    for m in CONTINUOUS:
        vals = agg[m].dropna().values
        samp = rng.choice(vals, min(200, len(vals)), replace=False)
        sw_s, sw_p = stats.shapiro(samp)
        grps = [agg.loc[agg.profession == p, m].dropna().values for p in PROFESSIONS]
        grps = [g for g in grps if len(g) > 1]
        lev_s, lev_p = stats.levene(*grps)
        normal = sw_p > 0.05
        equal_var = lev_p > 0.05
        rec = "ANOVA" if (normal and equal_var) else "Kruskal-Wallis"
        print(f"  {m:<35} SW-p={sw_p:.4f} {'NORMAL' if normal else 'NON-NORMAL':<12} "
              f"Levene-p={lev_p:.4f} {'EQ-VAR' if equal_var else 'UNEQUAL':<9} → {rec}")
        records.append(dict(metric=m, sw_stat=round(sw_s,4), sw_p=round(sw_p,4),
                            normal=normal, levene_stat=round(lev_s,4), levene_p=round(lev_p,4),
                            recommended_test=rec))
    pd.DataFrame(records).to_csv(os.path.join(OUT_DIR, "stage0_assumptions.csv"), index=False)
    print(f"  → Saved stage0_assumptions.csv\n")
    return pd.DataFrame(records)


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 1 — OMNIBUS KW TESTS
# ─────────────────────────────────────────────────────────────────────────────

def stage1(agg):
    print("=" * 70)
    print("  STAGE 1 — Omnibus Kruskal-Wallis  (persona-aggregated n=225)")
    print("=" * 70)
    records = []
    for m in CONTINUOUS:
        row = {"metric": m}
        print(f"\n  {m}")
        for gcol, glist in GROUPINGS:
            grps = [agg.loc[agg[gcol] == g, m].dropna().values for g in glist]
            grps = [g for g in grps if len(g) > 1]
            if len(grps) < 2: continue
            H, p = stats.kruskal(*grps)
            n = sum(len(g) for g in grps); k = len(grps)
            eta2 = kw_eta2(H, k, n)
            print(f"    {gcol:<15} H={H:8.3f}  p={p:.4f}{sig_stars(p):<4}  η²={eta2:.4f}")
            row[f"{gcol}_H"] = round(H, 3)
            row[f"{gcol}_p"] = round(p, 6)
            row[f"{gcol}_eta2"] = eta2
        records.append(row)
    pd.DataFrame(records).to_csv(os.path.join(OUT_DIR, "stage1_omnibus.csv"), index=False)
    print(f"\n  → Saved stage1_omnibus.csv\n")
    return pd.DataFrame(records)


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 2 — PAIRWISE + COHEN'S d + BH FDR
# ─────────────────────────────────────────────────────────────────────────────

def stage2(agg):
    print("=" * 70)
    print("  STAGE 2 — Pairwise Mann-Whitney + Cohen's d + BH FDR")
    print("  BH applied across ALL tests simultaneously.")
    print("=" * 70)
    records = []
    for m in CONTINUOUS:
        for gcol, glist in GROUPINGS:
            gdata = {g: agg.loc[agg[gcol] == g, m].dropna().values for g in glist}
            for g1, g2 in combinations(glist, 2):
                a, b = gdata[g1], gdata[g2]
                if len(a) < 2 or len(b) < 2: continue
                U, p = stats.mannwhitneyu(a, b, alternative='two-sided')
                d = cohens_d(a, b)
                ci_lo, ci_hi = bootstrap_d_ci(a, b)
                records.append(dict(
                    metric=m, grouping=gcol, group1=g1, group2=g2,
                    n1=len(a), n2=len(b),
                    mean1=round(float(np.mean(a)), 4), mean2=round(float(np.mean(b)), 4),
                    U=round(float(U), 1), p_raw=round(p, 6),
                    cohens_d=round(d, 4), ci_lo=ci_lo, ci_hi=ci_hi
                ))

    df = pd.DataFrame(records)
    df["q_fdr"]      = np.round(bh(df.p_raw.values), 6)
    df["sig"]        = df.q_fdr.apply(sig_stars)
    df["effect_size"]= df.cohens_d.apply(effect_label)

    sig_df = df[(df.q_fdr < 0.05) & (df.cohens_d.abs() >= 0.2)].sort_values(
        "cohens_d", key=abs, ascending=False)

    print(f"\n  Total pairwise tests: {len(df)}")
    print(f"  Significant (q<0.05 AND |d|≥0.2): {len(sig_df)}")
    print(f"\n  {'Metric':<28} {'By':<13} {'G1':<14} {'G2':<14} {'d':>7} "
          f"{'95%CI':>16} {'q':>8}  effect")
    print(f"  {'-'*28} {'-'*13} {'-'*14} {'-'*14} {'-'*7} {'-'*16} {'-'*8}  {'-'*10}")
    for _, r in sig_df.head(40).iterrows():
        ci = f"[{r.ci_lo:.3f},{r.ci_hi:.3f}]"
        print(f"  {r.metric:<28} {r.grouping:<13} {r.group1:<14} {r.group2:<14} "
              f"{r.cohens_d:>7.3f} {ci:>16} {r.q_fdr:>8.4f}  {r.effect_size}")

    df.to_csv(os.path.join(OUT_DIR, "stage2_pairwise.csv"), index=False)
    print(f"\n  → Saved stage2_pairwise.csv ({len(df)} tests)\n")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 3 — H2: AGENCY × SENTIMENT CORRELATION
# ─────────────────────────────────────────────────────────────────────────────

def stage3(agg):
    print("=" * 70)
    print("  STAGE 3 — H2: Agency Arc × Sentiment Score (Spearman)")
    print("  H2: lower-agency personas appear in more negative contexts")
    print("=" * 70)
    records = []

    def run_corr(label, sub):
        r, p = stats.spearmanr(sub.agency_arc.dropna(), sub.sentiment_score.dropna())
        n = len(sub.agency_arc.dropna())
        print(f"  {label:<30} n={n:<4} r={r:>7.4f}  p={p:.4f}  {sig_stars(p)}")
        records.append(dict(group=label, n=n, spearman_r=round(r,4), p=round(p,6)))

    run_corr("Overall", agg)
    print()
    for p in PROFESSIONS:
        run_corr(f"Profession: {p}", agg[agg.profession == p])
    print()
    for g in GENDERS:
        run_corr(f"Gender: {g}", agg[agg.gender == g])
    print()
    for nat in NATIONALITIES:
        run_corr(f"Nationality: {nat}", agg[agg.nationality == nat])

    pd.DataFrame(records).to_csv(os.path.join(OUT_DIR, "stage3_correlation.csv"), index=False)
    print(f"\n  → Saved stage3_correlation.csv\n")
    return pd.DataFrame(records)


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 4 — CHI-SQUARED (taxonomy_label, topic_bias)
# ─────────────────────────────────────────────────────────────────────────────

def cramers_v(chi2, n, r, c):
    return round(np.sqrt(chi2 / (n * (min(r, c) - 1))), 4)

def stage4(agg):
    print("=" * 70)
    print("  STAGE 4 — Chi-squared + Cramér's V + Standardized Residuals")
    print("=" * 70)
    records = []

    def run_chi2(metric, col_filter, cats, group_col, group_list, label):
        if col_filter:
            sub = agg[agg[metric].isin(col_filter)]
        else:
            sub = agg[agg[metric].notna()]
        # reindex ensures all expected categories appear as columns even if
        # absent from the data after aggregation (fills missing with 0)
        ct = (pd.crosstab(sub[group_col], sub[metric])
                .reindex(columns=cats, fill_value=0))
        # Drop any categories that are entirely zero across all groups
        # (avoids division-by-zero in chi2 and degenerate test)
        ct = ct.loc[:, ct.sum(axis=0) > 0]
        active_cats = list(ct.columns)
        if len(active_cats) < 2:
            print(f"\n  {label} × {group_col}  SKIP — fewer than 2 active categories")
            return
        chi2, p, dof, exp = stats.chi2_contingency(ct)
        n = ct.values.sum()
        V = cramers_v(chi2, n, ct.shape[0], ct.shape[1])
        print(f"\n  {label} × {group_col}  [active cats: {active_cats}]")
        print(f"    χ²={chi2:.3f}  df={dof}  p={p:.4f}{sig_stars(p)}  V={V:.4f}")
        # Standardized residuals — use exp matrix from chi2_contingency
        with np.errstate(divide='ignore', invalid='ignore'):
            std_res = np.where(exp > 0, (ct.values - exp) / np.sqrt(exp), 0.0)
        notable = [(ct.index[i], active_cats[j], round(std_res[i, j], 2))
                   for i in range(std_res.shape[0])
                   for j in range(std_res.shape[1])
                   if abs(std_res[i, j]) > 2.0]
        if notable:
            print(f"    Std residuals |r|>2: " +
                  ", ".join(f"{g}×{c}={v:+.2f}" for g, c, v in notable))
        records.append(dict(metric=metric, categories=str(active_cats), grouping=group_col,
                            chi2=round(chi2, 3), df=dof, p=round(p, 6), cramers_v=V))

    # taxonomy_label — collapse to 2 active categories
    ACTIVE_TAX = ["Professional_Authority", "Institutional_Subordination"]
    for gcol, glist in [("profession", PROFESSIONS), ("gender", GENDERS),
                        ("nationality", NATIONALITIES)]:
        run_chi2("taxonomy_label", ACTIVE_TAX, ACTIVE_TAX, gcol, glist,
                 "taxonomy_label (2-cat)")

    # topic_bias — 3 arc types
    ARC_CATS = ["Competence_Arc", "Acceptance_Arc", "Mixed_Arc"]
    for gcol, glist in [("profession", PROFESSIONS), ("gender", GENDERS),
                        ("nationality", NATIONALITIES)]:
        run_chi2("topic_bias", ARC_CATS, ARC_CATS, gcol, glist,
                 "topic_bias arc-type (3-cat)")

    pd.DataFrame(records).to_csv(os.path.join(OUT_DIR, "stage4_chisquared.csv"), index=False)
    print(f"\n  → Saved stage4_chisquared.csv\n")
    return pd.DataFrame(records)


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 5 — CROSS-MODEL REPLICATION
# ─────────────────────────────────────────────────────────────────────────────

def stage5(agg):
    print("=" * 70)
    print("  STAGE 5 — Cross-Model Replication (KW by profession, per model)")
    print("  Key question: do profession-level findings replicate across all 3 LLMs?")
    print("=" * 70)
    KEY = ["agency_arc", "agency_quality_score", "subordination_score",
           "dialogue_score", "resolution_score"]
    records = []
    header = f"  {'Metric':<35} {'Gemma':>12} {'Llama3':>12} {'Mistral':>12}  Replicated"
    print(f"\n  By profession — KW p-values per model:")
    print(header)
    print(f"  {'-'*35} {'-'*12} {'-'*12} {'-'*12}  {'-'*10}")
    for m in KEY:
        row = {"metric": m}
        model_ps = {}
        for model in MODELS:
            sub = agg[agg.model_name == model]
            grps = [sub.loc[sub.profession == p, m].dropna().values for p in PROFESSIONS]
            grps = [g for g in grps if len(g) > 1]
            if len(grps) < 2:
                model_ps[model] = np.nan; continue
            H, p = stats.kruskal(*grps)
            eta2 = kw_eta2(H, len(grps), sum(len(g) for g in grps))
            model_ps[model] = p
            row[f"{model}_H"] = round(H, 3)
            row[f"{model}_p"] = round(p, 6)
            row[f"{model}_eta2"] = eta2
        replicated = all(v < 0.05 for v in model_ps.values() if not np.isnan(v))
        ps = {k: f"{v:.4f}{sig_stars(v)}" if not np.isnan(v) else "n/a"
              for k, v in model_ps.items()}
        print(f"  {m:<35} {ps.get('gemma3:4b','n/a'):>12} {ps.get('llama3','n/a'):>12} "
              f"{ps.get('mistral','n/a'):>12}  {'YES' if replicated else 'NO'}")
        row["replicated_all_models"] = replicated
        records.append(row)

    pd.DataFrame(records).to_csv(os.path.join(OUT_DIR, "stage5_cross_model.csv"), index=False)
    print(f"\n  → Saved stage5_cross_model.csv\n")
    return pd.DataFrame(records)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("\nHCNLP Narrative Bias — Statistical Pipeline v4")
    print("Primary analysis: persona×model aggregated (n=225)\n")
    raw, agg = load()

    stage0(agg)
    stage1(agg)
    s2 = stage2(agg)
    stage3(agg)
    stage4(agg)
    stage5(agg)

    print("=" * 70)
    print("  PIPELINE COMPLETE — All CSVs in data/stats_results/")
    print("  Next: run paper_figures.py to generate paper visualizations.")
    print("=" * 70)

if __name__ == "__main__":
    main()