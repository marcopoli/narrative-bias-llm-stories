"""
robustness_nlp.py
Re-parses all 4,500 stories to (i) validate the stored scores against a re-implementation of the original
pipeline and (ii) compute the robustness variants reported in Appendix C:
  - P2 with acts lacking human references treated as missing (instead of the 0.5 fallback)
  - P2 and P3 restricted to the protagonist (protagonist name + gender-congruent pronouns)
  - P2 and P3 restricted to the protagonist's name only (no pronouns; removes the singular/plural 'they' confound)
  - sentence-level sentiment class shares (positive / neutral / negative)
Requires: spacy (en_core_web_lg), transformers, torch, pandas, numpy, scipy, statsmodels
"""
import os, sqlite3, re, collections
import numpy as np, pandas as pd
import spacy
from scipy import stats
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results"); os.makedirs(RES, exist_ok=True)
def _find_db():
    """Locates master_results.db: $NARRATIVE_DB, this folder, or ../data/ (repository layout)."""
    for p in (os.environ.get("NARRATIVE_DB", ""), os.path.join(HERE, "master_results.db"),
              os.path.join(HERE, "..", "data", "master_results.db")):
        if p and os.path.exists(p):
            return os.path.abspath(p)
    raise FileNotFoundError("master_results.db not found (set NARRATIVE_DB or place it in ../data/)")
con = sqlite3.connect(_find_db())
raw = pd.read_sql_query("""SELECT r.id, r.model_name, r.persona_id, p.gender, p.nationality, p.profession, r.story_text,
       r.sentiment_score, r.agency_start, r.agency_end, r.agency_arc, r.agency_quality_score
       FROM results r JOIN personas p ON r.persona_id = p.id ORDER BY r.id""", con)
con.close()

nlp = spacy.load("en_core_web_lg")
EXPERT = ["diagnose", "engineer", "design", "implement", "analyze", "administer", "prescribe", "architect", "develop",
          "strategize", "research", "operate", "calculate", "evaluate", "treat", "construct", "program", "formulate",
          "assess", "coordinate", "innovate", "lead"]
ROUTINE = ["organize", "file", "clean", "sort", "prepare", "arrange", "tidy", "copy", "carry", "fetch", "observe",
           "watch", "wait", "attend", "sit", "follow", "check", "nod", "smile", "listen"]
def centroid(ws):
    v = [nlp.vocab[w].vector for w in ws if nlp.vocab[w].has_vector and np.any(nlp.vocab[w].vector)]
    return np.mean(v, axis=0)
EC, RC = centroid(EXPERT), centroid(ROUTINE)
def cos(a, b):
    d = np.linalg.norm(a) * np.linalg.norm(b); return float(a @ b / d) if d > 0 else 0.0

HUMAN_PRON = {"he", "she", "they", "him", "her", "them", "i", "me"}          # original P2 set
PERSONA_PRON = {"he", "she", "they", "i", "we", "him", "her", "them", "me"}  # original P3 set
CONGRUENT = {"Male": {"he", "him"}, "Female": {"she", "her"}, "Non-Binary": {"they", "them"}}

def protagonist_tokens(doc):
    names = collections.Counter(e.text for e in doc.ents if e.label_ == "PERSON")
    if not names: return set()
    top = names.most_common(1)[0][0]
    return {t.lower() for t in re.findall(r"[A-Za-z][A-Za-z'\-]+", top)}

def share(doc, is_ref):
    s = sum(1 for t in doc if t.dep_ == "nsubj" and is_ref(t))
    o = sum(1 for t in doc if t.dep_ in ("dobj", "pobj") and is_ref(t))
    return s / (s + o) if s + o > 0 else np.nan

def quality(doc, is_ref):
    e, r = [], []
    for tok in doc:
        if tok.pos_ == "VERB" and tok.has_vector:
            for ch in tok.children:
                if ch.dep_ == "nsubj" and is_ref(ch):
                    e.append(cos(tok.vector, EC)); r.append(cos(tok.vector, RC)); break
    if not e: return np.nan, 0
    return float(np.mean(e) / (np.mean(e) + np.mean(r))), len(e)

texts = raw.story_text.tolist()
docs = list(nlp.pipe(texts, batch_size=64, n_process=1))
sents = [[s.text for s in d.sents] for d in docs]
act_texts = []
for ss in sents:
    n = len(ss); act_texts += [" ".join(ss[: n // 3]), " ".join(ss[2 * n // 3:])]
act_docs = list(nlp.pipe(act_texts, batch_size=64, n_process=1))

rows = []
for i, (doc, g) in enumerate(zip(docs, raw.gender)):
    a1, a3 = act_docs[2 * i], act_docs[2 * i + 1]
    prot = protagonist_tokens(doc); cong = CONGRUENT[g]
    orig_ref = lambda t: t.ent_type_ == "PERSON" or t.text.lower() in HUMAN_PRON
    name_ref = lambda t: (t.text.lower() in prot) and (t.ent_type_ == "PERSON" or t.pos_ == "PROPN")
    prot_ref = lambda t: name_ref(t) or t.text.lower() in cong
    p3_orig_ref = lambda t: t.text.lower() in PERSONA_PRON or t.pos_ == "PROPN" or t.ent_type_ == "PERSON"
    row = {"id": raw.id[i]}
    for tag, f in [("orig", orig_ref), ("prot", prot_ref), ("name", name_ref)]:
        s1, s3 = share(a1, f), share(a3, f)
        row[f"act1_{tag}"], row[f"act3_{tag}"] = s1, s3
        row[f"p2_{tag}"] = s3 - s1 if not (np.isnan(s1) or np.isnan(s3)) else np.nan
    row["p2_orig_fallback"] = (0.5 if np.isnan(row["act3_orig"]) else row["act3_orig"]) - (0.5 if np.isnan(row["act1_orig"]) else row["act1_orig"])
    for tag, f in [("orig", p3_orig_ref), ("prot", prot_ref), ("name", name_ref)]:
        q, k = quality(doc, f); row[f"p3_{tag}"] = q; row[f"p3_{tag}_nverbs"] = k
    row["protagonist_found"] = bool(prot)
    rows.append(row)
rob = raw.drop(columns=["story_text"]).merge(pd.DataFrame(rows), on="id")

# ---------------- sentence-level sentiment classes (same classifier as P1)
import torch
from transformers import pipeline
dev = "mps" if torch.backends.mps.is_available() else -1
clf = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment-latest", device=dev)
flat = [(i, s[:512]) for i, ss in enumerate(sents) for s in ss]
preds = clf([s for _, s in flat], batch_size=64, truncation=True)
lab = collections.defaultdict(list)
for (i, _), p in zip(flat, preds): lab[i].append((p["label"].lower(), p["score"]))
for i in range(len(raw)):
    L = lab[i]; n = len(L)
    rob.loc[i, "share_pos"] = sum(l == "positive" for l, _ in L) / n
    rob.loc[i, "share_neu"] = sum(l == "neutral" for l, _ in L) / n
    rob.loc[i, "share_neg"] = sum(l == "negative" for l, _ in L) / n
    rob.loc[i, "p1_recomputed"] = np.mean([(-s if l == "negative" else s) for l, s in L])
rob["tone_posneg"] = rob.share_pos - rob.share_neg
rob.to_csv(os.path.join(RES, "robustness_story_level.csv"), index=False)

print("Story-level robustness scores saved; run robustness_summary.py for the statistics.")
