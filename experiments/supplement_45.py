"""Writes results/supplement_experiments.md: Sections S4 (main-corpus results moved from the article) and S5 (follow-up
experiments) of Online Resource 1. Inputs as in final_tables.py, plus out/pairs_*.jsonl (judge_pairs.py).
Run with the system python3 after final_tables.py."""
import glob, json, os
import numpy as np, pandas as pd
from scipy import stats
from generate import PROMPTS
from generate2 import PARITY, BLIND_DRAFT, BLIND_ADAPT, REVISE
from judge import INSTRUCTION
from judge_pairs import INSTRUCTION as PAIR_INSTRUCTION
from masking import PATTERNS, EXTRA_ROLES

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
AN = os.path.join(HERE, "..", "analysis", "results")
N = json.load(open(os.path.join(RES, "final_numbers.json")))
ITEMS = ["warmth", "competence", "agency", "development", "welcome"]
P = ["Engineer", "Teacher", "Nurse", "Doctor", "Janitor"]

def f(x, k=2, sign=False):
    if x is None or (isinstance(x, float) and np.isnan(x)): return "–"
    s = f"{x:+.{k}f}" if sign else f"{x:.{k}f}"
    return s.replace("-", "−")
def fp(p): return "< .001" if p < .001 else f"{p:.3f}".replace("0.", ".", 1)
def ci(g, k=2):
    # janitor gap with 95% CI; the arrow marks a CI that excludes zero (down: lower for janitors, up: higher)
    arrow = "↓ " if g[2] < 0 else "↑ " if g[1] > 0 else ""
    return f"{arrow}{f(g[0], k)} [{f(g[1], k)}, {f(g[2], k)}]"
def md(df):
    cols = list(df.columns)
    out = ["| " + " | ".join(cols) + " |", "|" + "|".join([":--"] + [":--:"] * (len(cols) - 1)) + "|"]
    return "\n".join(out + ["| " + " | ".join(str(v) for v in r) + " |" for r in df.values])
def quote(t): return "\n".join("> " + (l if l.strip() else "") for l in t.strip().split("\n"))
out = []; A = out.append

# ------------------------------------------------------------------ S4
A("# S4 Additional results for the main corpus\n")
kw = pd.read_csv(os.path.join(AN, "T2_omnibus_kruskal.csv"))
LAB = {"sentiment_score": "P1 Tone", "agency_arc": "P2 Agency arc", "agency_quality_score": "P3 Agency quality", "concreteness_score": "P4 Lexical expertise"}
FAC = ["profession", "gender", "nationality", "model_name"]
rows = []
for m, lab in LAB.items():
    r = [lab]
    for fac in FAC:
        x = kw[(kw.measure == m) & (kw.factor == fac)].iloc[0]
        r.append(f"{x.eta2_H:.3f}" + ("\\*\\*\\*" if x.q < .001 else "\\*\\*" if x.q < .01 else "\\*" if x.q < .05 else ""))
    rows.append(r)
A("Table S15. Kruskal–Wallis effect sizes ($\\eta^2_H$) by measure and factor in the main corpus (*n* = 225 persona–model units; "
  "BH correction across the 16 tests; \\* *q* < .05, \\*\\* *q* < .01, \\*\\*\\* *q* < .001)\n")
A(md(pd.DataFrame(rows, columns=["Measure", "Profession", "Gender", "Nationality", "Model"])) + "\n")
A("![Fig. S4. Mean tone (a) and agency arc (b) by profession and model in the main corpus (± 95% CI over 15 persona units per cell). "
  "Tone differs mainly between models; the janitor persona has the smallest agency arc in all three models](../figures/FigS4.png){width=100%}\n")
A("![Fig. S5. Agency quality (P3) by gender and profession in the main corpus (mean ± 95% CI over 15 units); dashed line: corpus mean]"
  "(../figures/FigS5.png){width=92%}\n")

# ------------------------------------------------------------------ S5
A("# S5 Follow-up experiments\n")
A("## S5.1 Models, prompts and settings\n")
models = [("Gemma 3 4B", "Google", "gemma3:4b", "Main corpus"), ("Llama 3 8B", "Meta", "llama3 / llama3:8b", "Main corpus, Exp. 4, 5"),
          ("Mistral 7B", "Mistral AI", "mistral / mistral:7b", "Main corpus, Exp. 2, 4, 5"), ("Gemma 3 27B", "Google", "gemma3:27b", "Exp. 1"),
          ("Mistral Small 3.2 24B", "Mistral AI", "mistral-small3.2:24b", "Exp. 1"), ("gpt-oss 20B", "OpenAI", "gpt-oss:20b", "Exp. 1"),
          ("Gemma 4 31B", "Google", "gemma4:31b", "Exp. 1, 4, 5"), ("Qwen 3.8 27B", "Alibaba", "qwen3.8:27b", "Exp. 1"),
          ("Qwen3 8B (rater)", "Alibaba", "qwen3:8b; hf.co/Qwen/Qwen3-8B-GGUF:Q4_K_M (pairs)", "Exp. 3, 5"),
          ("Granite 3.3 8B (rater)", "IBM", "granite3.3:8b", "Exp. 3"), ("Qwen3 32B (rater)", "Alibaba", "qwen3:32b", "Exp. 3")]
A("Table S16. Models of the study. All models ran in Ollama with their default 4-bit quantisation; story models used temperature 0.7, "
  "raters temperature 0 and JSON output. Gemma 4 and Qwen 3.8 ran with thinking switched off, gpt-oss with low reasoning effort\n")
A(md(pd.DataFrame(models, columns=["Model", "Developer", "Ollama tag", "Used in"])) + "\n")
A("**Prompts.** Exp. 2 used the original prompt and these two variants:\n")
A(quote(PROMPTS["ordinary_day"].format(n="[nationality]", g="[gender]", p="[profession]")) + "\n")
A(quote(PROMPTS["development"].format(n="[nationality]", g="[gender]", p="[profession]")) + "\n")
A("Exp. 5 added to the original prompt the equal-agency instruction:\n")
A(quote(PARITY.strip()) + "\n")
A("Job-blind drafting used two calls. The draft prompt was:\n")
A(quote(BLIND_DRAFT) + "\n")
A("and the adaptation prompt (with the draft in place of {s}):\n")
A(quote(BLIND_ADAPT.replace("{n} {g} {p}", "[nationality] [gender] [profession]").replace("{s}", "[draft]")) + "\n")
A("Self-revision gave the original story of the same seed to the same model with this prompt:\n")
A(quote(REVISE.replace("{s}", "[story]")) + "\n")
A("**Seeds.** In Exp. 4 and 5 the seed of a story was 1000 × (repetition + 1) + index of the gender–nationality combination, so stories with "
  "the same gender, nationality and repetition share a seed across occupations and conditions. A draft in job-blind drafting is shared by the "
  "five professions of the same seed.\n")
A("**Masking.** Before rating, these case-insensitive patterns were replaced by the placeholder shown: " +
  "; ".join(f"`{p}` → {r}" for p, r in PATTERNS) + f". Exp. 4 also masked `{EXTRA_ROLES}` → [ROLE].\n")
A("**Rater instruction (single story).**\n")
A(quote(INSTRUCTION + "[masked story]") + "\n")
A("**Rater instruction (pairs, Exp. 5).**\n")
A(quote(PAIR_INSTRUCTION.replace("%s", "[masked story]")) + "\n")

A("## S5.2 Prompt variants (Exp. 2)\n")
rows = []
for v, lab in [("original", "Original"), ("ordinary_day", "Without first-day framing"), ("development", "Welcome and development requested")]:
    x = N["prompts"][v]
    rows.append([lab, f(x["tone"]), ci(x["agency_arc"]["diff"], 3), ci(x["agency"]["diff"]), ci(x["warmth"]["diff"])])
A("Table S17. Janitor gap under the three prompts (Mistral 7B, 300 stories each; janitor minus other professions, persona means, 95% bootstrap CI)\n")
A(md(pd.DataFrame(rows, columns=["Prompt", "Tone (P1)", "Agency arc (P2)", "Rated agency", "Rated warmth"])) + "\n")

A("## S5.3 Reliability of the ratings (Exp. 3)\n")
ag = N["agreement"]
rows = [[it.capitalize(), f(ag[it]), f(ag["units"][it]), ci(N["qwen_main"][it]["gap"]), f(N["qwen_main"][it]["d"]),
         ci(N["granite_main"][it]["gap"]), f(N["granite_main"][it]["d"])] for it in ITEMS]
A(f"Table S18. Agreement between Qwen3 8B and Granite 3.3 8B on the main-corpus sample ({ag['n']} stories; Spearman ρ on stories and on "
  "persona–model means) and janitor gap by rater (janitor minus other professions, persona–model means, 95% bootstrap CI; *d*: other minus janitor)\n")
A(md(pd.DataFrame(rows, columns=["Item", "ρ stories", "ρ units", "Gap Qwen3 8B", "*d*", "Gap Granite 3.3", "*d*"])) + "\n")
cv = N["convergent_qwen"]
A("Convergent validity (Qwen3 8B, persona–model means): rated agency with P3 ρ = " + f(cv["rho_agency_agency_quality_score"][0]) +
  ", with P2 ρ = " + f(cv["rho_agency_agency_arc"][0]) + "; rated competence with P3 ρ = " + f(cv["rho_competence_agency_quality_score"][0]) +
  "; rated development with P2 ρ = " + f(cv["rho_development_agency_arc"][0]) + " (*p* = " + fp(cv["rho_development_agency_arc"][1]) +
  "); rated warmth with P1 ρ = " + f(cv["rho_warmth_sentiment_score"][0]) + " (*p* = " + fp(cv["rho_warmth_sentiment_score"][1]) + ").\n")
if "robust_rater" in N:
    rr = N["robust_rater"]
    rows = [[it.capitalize(), f(rr["story_rho"][it]), ci(rr["jan_big"][it]["diff"]), f(rr["jan_big"][it]["d"])] for it in ITEMS]
    A(f"Table S19. Larger rater: Qwen3 32B on a stratified sample of {rr['n']} main-corpus stories (40 per profession and model). Spearman ρ "
      "with Qwen3 8B on the same stories, and janitor gap of Qwen3 32B (persona–model means, 95% bootstrap CI)\n")
    A(md(pd.DataFrame(rows, columns=["Item", "ρ with Qwen3 8B", "Janitor gap", "*d*"])) + "\n")

A("## S5.4 Occupational status (Exp. 4)\n")
occ = pd.DataFrame(N["exp4_occ_means_pooled"]).T
pre = pd.read_csv(os.path.join(HERE, "norms", "occupation_prestige_hughes2022.csv")).set_index("occupation")
nor = pd.read_csv(os.path.join(HERE, "norms", "occupation_norms_friehs2022.csv")).set_index("occupation")
occ = occ.join(pre[["prestige", "opr_titles"]]).join(nor[["warmth", "competence"]].rename(columns={"warmth": "h_w", "competence": "h_c"}))
occ = occ.sort_values("prestige")
rows = [[o, f(r.prestige, 1), f(r.h_w), f(r.h_c), f(r.sentiment_score), f(r.agency_arc, 3), f(r.agency_quality_score, 3)] +
        [f(r[it]) for it in ITEMS] for o, r in occ.iterrows()]
A("Table S20. Occupation means in Exp. 4 (Mistral 7B, Llama 3 8B and Gemma 4 31B; 90 stories per occupation), with occupational "
  "prestige (0–100) [Hughes et al., 2024] and human warmth (W) and competence (C) norms (1–5) [Friehs et al., 2022]. Prestige titles: " +
  "; ".join(f"{o}: {t}" for o, t in pre.opr_titles.items()) + "\n")
A(md(pd.DataFrame(rows, columns=["Occupation", "Prestige", "W", "C", "P1", "P2", "P3"] + [i.capitalize() for i in ITEMS])) + "\n")
rows = []
for m, lab in [("mistral:7b", "Mistral 7B"), ("llama3:8b", "Llama 3 8B"), ("gemma4:31b", "Gemma 4 31B"), ("pooled", "Pooled")]:
    e = N["exp4"][m]
    cell = lambda k: f"{f(e[k][0])} ({fp(e[k][1])})"
    rows.append([lab, cell("rho_prestige_agency"), cell("rho_prestige_nojan_agency"), cell("beta_prestige_agency"), cell("rho_prestige_competence"),
                 cell("rho_prestige_warmth"), cell("rho_warmth_warmth"), cell("rho_competence_agency"), cell("rho_prestige_sentiment_score")])
A("Table S21. Exp. 4: Spearman correlations across occupations (13 with prestige, 11 with human norms; *p* in brackets) and unit-level "
  "regression of rated agency on prestige (β in SD units, standard errors clustered by occupation)\n")
A(md(pd.DataFrame(rows, columns=["Model", "Prestige–agency", "Without janitor", "β prestige", "Prestige–competence", "Prestige–warmth",
                                 "Warmth norm–warmth", "Competence norm–agency", "Prestige–tone"])) + "\n")

A("## S5.5 Mitigation (Exp. 5)\n")
R = pd.concat([pd.read_json(fn, lines=True) for fn in glob.glob(os.path.join(HERE, "out", "ratings_judge_input_45_*.jsonl")) +
               glob.glob(os.path.join(HERE, "out", "ratings_judge_input_exp5g_*.jsonl"))])
meta = []
for fn in glob.glob(os.path.join(HERE, "out", "stories_exp5_*.jsonl")):
    tag = os.path.basename(fn)[len("stories_"):-len(".jsonl")]
    for i, l in enumerate(open(fn)):
        d = json.loads(l); meta.append(dict(uid=f"{tag}-{i}", model=d["model"], variant=d["variant"]))
ceil = R.merge(pd.DataFrame(meta), on="uid").groupby(["model", "variant"]).agency.apply(lambda x: (x == 7).mean())
pairs = pd.concat([pd.read_json(fn, lines=True) for fn in glob.glob(os.path.join(HERE, "out", "pairs_*.jsonl"))]) if glob.glob(os.path.join(HERE, "out", "pairs_*.jsonl")) else pd.DataFrame()
TAG = {"Mistral 7B": "mistral:7b", "Llama 3 8B": "llama3:8b", "Gemma 4 31B": "gemma4:31b"}
VL = {"baseline": "Original prompt", "parity": "Equal-agency instruction", "blind": "Job-blind drafting", "revise": "Self-revision"}
def pairs_cell(m, v):
    q = N.get("pairs", {}).get(m, {}).get(v)
    return f"{q['janitor']}/{q['janitor'] + q['other']} ({100 * q['consistent']:.0f}%)" if q else "–"
rows = []
for m in ["Mistral 7B", "Llama 3 8B"]:
    for v in VL:
        x = N["exp5"][m][v]
        pw = pairs_cell(m, v)
        rows.append([m, VL[v], f(x["words"], 0), f(x["similarity"]), f(x["sentiment_score"]["mean"]), f(x["agency"]["mean"]),
                     f"{100 * ceil.get((TAG[m], v), np.nan):.0f}%", f(x["agency_arc"]["gap"][0], 3), f(x["agency_quality_score"]["gap"][0], 3),
                     ci(x["agency"]["gap"]), f(x["competence"]["gap"][0]), f(x["warmth"]["gap"][0]), f(x["welcome"]["gap"][0]), pw])
if "Gemma 4 31B" in N["exp5"]:
    for fn in glob.glob(os.path.join(HERE, "out", "stories_exp4_gemma4-31b.jsonl")):
        for i, l in enumerate(open(fn)): meta.append(dict(uid=f"exp4_gemma4-31b-{i}", model="gemma4:31b", variant="baseline"))
    ceil = R.merge(pd.DataFrame(meta), on="uid").groupby(["model", "variant"]).agency.apply(lambda x: (x == 7).mean())
    for v in ["baseline", "parity"]:
        x = N["exp5"]["Gemma 4 31B"][v]
        rows.append(["Gemma 4 31B", VL[v] + (" (Exp. 4 stories, same seeds)" if v == "baseline" else ""),
                     f(x["words"], 0) if x.get("words") else "–", f(x["similarity"]) if x.get("similarity") else "–", f(x["tone"]),
                     f(x["agency"]["mean"]), f"{100 * ceil.get(('gemma4:31b', v), np.nan):.0f}%", f(x["agency_arc"]["diff"][0], 3),
                     f(x["agency_quality_score"]["diff"][0], 3), ci(x["agency"]["diff"]), f(x["competence"]["diff"][0]),
                     f(x["warmth"]["diff"][0]), f(x["welcome"]["diff"][0]), pairs_cell("Gemma 4 31B", v)])
A("Table S22. Exp. 5 by model and condition (300 stories each; 150 for the Gemma 4 baseline, two per persona). Similarity: TF-IDF cosine between stories with the same seed, gender and "
  "nationality but different professions. Ceiling: share of stories with rated agency 7. Gaps: janitor minus other professions on persona "
  "means (P2, P3 and ratings); rated agency with 95% bootstrap CI. Pairs: order-consistent pairwise verdicts in favour of the janitor character out of "
  "all order-consistent verdicts (share of the 240 pairs per condition, 120 for the Gemma 4 baseline, with an order-consistent verdict in brackets); under equal treatment, half of the "
  "consistent verdicts should favour the janitor. Pairs were judged by the official GGUF release of Qwen3 8B, which chose the first story "
  f"in {100 * N.get('pairs_position_A', float('nan')):.0f}% of its decisive answers; judging each pair in both orders cancels this position bias\n")
A(md(pd.DataFrame(rows, columns=["Model", "Condition", "Words", "Similarity", "Tone", "Rated agency", "Ceiling", "P2 gap", "P3 gap",
                                 "Rated agency gap", "Competence gap", "Warmth gap", "Welcome gap", "Pairs"])) + "\n")

A("## S5.6 Standard bias checks\n")
au = N["audit_main"]
rows = [[p, f(au[p]["tone"]), f(au[p]["regard_pos"]), f(au[p]["regard_neg"], 3), f(au[p]["regard_neutral_other"]), f"{100 * au[p]['tox_flag_share']:.1f}%",
         str(N["flagged"]["by_prof"].get(p, 0))] for p in P]
A("Table S23. Standard checks in the main corpus by profession (900 stories each): tone, mean sentence probability of positive, negative and "
  "neutral or other regard (`sasha/regardv3`), share of stories with at least one sentence above the default toxicity threshold of 0.5 "
  "(`facebook/roberta-hate-speech-dynabench-r4-target`) and number of flagged sentences\n")
A(md(pd.DataFrame(rows, columns=["Profession", "Tone", "Positive regard", "Negative regard", "Neutral/other", "Flagged stories", "Flagged sentences"])) + "\n")
fl = N["flagged"]
A(f"Of the flagged sentences in janitor stories, {100 * fl['label_or_cleaning_janitor']:.0f}% contained a persona label (masking patterns above) "
  f"or a cleaning word (*mop, sweep, broom, trash, garbage, waste, dirt, clean, scrub, bucket, vacuum, spill, stain, toilet, restroom, dust* and "
  f"their forms), compared with {100 * fl['label_or_cleaning_others']:.0f}% in the other stories. Across all {N['audit_all_original_prompt']['n']:,} "
  f"stories written with the original prompt by the story models, {100 * N['audit_all_original_prompt']['tox_flag_janitor']:.1f}% of janitor "
  f"and {100 * N['audit_all_original_prompt']['tox_flag_others']:.1f}% of other stories were flagged.\n")
open(os.path.join(RES, "supplement_experiments.md"), "w", encoding="utf-8").write("\n".join(out))
print("supplement_experiments.md written")
