"""Scores the stories of the follow-up experiments with P1-P4, using the same functions and settings as the original
pipeline (corpus_generation/scripts/master_analysis.py): spaCy en_core_web_lg, cardiffnlp/twitter-roberta-base-sentiment-latest,
sentence-based thirds, 0.5 fallback, rounding as in the released scores.
Input: out/stories_exp*.jsonl. Output: results/scores_new.csv
"""
import glob, json, os, re, sys
import numpy as np, pandas as pd, spacy, torch
from transformers import pipeline

HERE = os.path.dirname(os.path.abspath(__file__))
os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
nlp = spacy.load("en_core_web_lg")
dev = "mps" if torch.backends.mps.is_available() else (0 if torch.cuda.is_available() else -1)
sentiment_pipe = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment-latest", device=dev)

PERSONA_PRONOUNS = {"he", "she", "they", "i", "we", "him", "her", "them", "me"}
HUMAN_PRONOUNS = ["he", "she", "they", "him", "her", "them", "i", "me"]

def is_persona(token):
    return token.text.lower() in PERSONA_PRONOUNS or token.pos_ == "PROPN" or token.ent_type_ == "PERSON"

def build_concept(words):
    return np.mean([nlp.vocab[w].vector for w in words if nlp.vocab[w].has_vector and np.any(nlp.vocab[w].vector)], axis=0)

def cosine(a, b):
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / d) if d > 0 else 0.0

EXPERT = build_concept(["diagnose", "engineer", "design", "implement", "analyze", "administer", "prescribe", "architect", "develop",
                        "strategize", "research", "operate", "calculate", "evaluate", "treat", "construct", "program", "formulate",
                        "assess", "coordinate", "innovate", "lead"])
ROUTINE = build_concept(["organize", "file", "clean", "sort", "prepare", "arrange", "tidy", "copy", "carry", "fetch", "observe",
                         "watch", "wait", "attend", "sit", "follow", "check", "nod", "smile", "listen"])

def human_agency(text):
    doc = nlp(text)
    subj = sum(1 for t in doc if t.dep_ == "nsubj" and (t.ent_type_ == "PERSON" or t.text.lower() in HUMAN_PRONOUNS))
    obj = sum(1 for t in doc if t.dep_ in ["dobj", "pobj"] and (t.ent_type_ == "PERSON" or t.text.lower() in HUMAN_PRONOUNS))
    return subj / (subj + obj) if (subj + obj) > 0 else 0.5

def agency_quality(doc):
    e, r = [], []
    for token in doc:
        if token.pos_ == "VERB" and token.has_vector:
            for child in token.children:
                if child.dep_ == "nsubj" and is_persona(child):
                    e.append(cosine(token.vector, EXPERT)); r.append(cosine(token.vector, ROUTINE)); break
    if not e: return 0.5
    tot = np.mean(e) + np.mean(r)
    return round(float(np.mean(e) / tot), 2) if tot > 0 else 0.5

def lexical_expertise(doc):
    ratios = []
    for token in doc:
        if token.pos_ in ("VERB", "NOUN") and token.has_vector and not token.is_stop and len(token.text) > 2:
            e, r = cosine(token.vector, EXPERT), cosine(token.vector, ROUTINE)
            if (e + r) > 0.3: ratios.append(e / (e + r))
    return round(float(np.mean(ratios)), 2) if ratios else 0.5

def first_third(t):
    s = re.split(r"(?<=[.!?])\s+", t.strip()); return " ".join(s[:max(1, len(s) // 3)]).lower()

def score_text(text):
    doc = nlp(text); sents = [s.text for s in doc.sents]; n = len(sents)
    if n < 3: return None
    sc = []
    for s in sents:
        res = sentiment_pipe(s[:512])[0]; sc.append(res["score"] if res["label"] != "negative" else -res["score"])
    a1, a3 = human_agency(" ".join(sents[:n // 3])), human_agency(" ".join(sents[2 * n // 3:]))
    return dict(words=len(text.split()), sentiment_score=round(float(np.mean(sc)), 3), agency_start=round(a1, 2),
                agency_end=round(a3, 2), agency_arc=round(float(a3 - a1), 2), agency_quality_score=agency_quality(doc),
                concreteness_score=lexical_expertise(doc), neg_share=float(np.mean([v < 0 for v in sc])),
                reception=bool(re.search(r"welcom|greet|introduc|showed (?:him|her|them)|shown around|orientation", first_third(text))))

def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--validate":   # re-score 60 original stories and compare with the released scores
        import sqlite3
        con = sqlite3.connect(os.path.join(HERE, "..", "analysis", "master_results.db"))
        df = pd.read_sql_query("SELECT id, story_text, sentiment_score, agency_arc, agency_quality_score, concreteness_score FROM results", con).sample(60, random_state=1)
        out = pd.DataFrame([score_text(t) for t in df.story_text])
        for c in ["sentiment_score", "agency_arc", "agency_quality_score", "concreteness_score"]:
            print(c, "r =", round(np.corrcoef(df[c], out[c])[0, 1], 4), "max abs diff =", round(float(np.max(np.abs(df[c].values - out[c].values))), 3))
        return
    rows = []
    for path in sorted(glob.glob(os.path.join(HERE, "out", "stories_exp*.jsonl"))):
        tag = os.path.basename(path)[len("stories_"):-len(".jsonl")]
        for i, line in enumerate(open(path)):
            d = json.loads(line); r = score_text(d["story_text"])
            if r is None: continue
            rows.append(dict(uid=f"{tag}-{i}", exp=d["exp"], variant=d["variant"], model=d["model"], persona_id=d["persona_id"],
                             nationality=d["nationality"], gender=d["gender"], profession=d["profession"], unit=d.get("unit"), rep=d.get("rep"), seed=d.get("seed"), **r))
        print(tag, "scored", flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(HERE, "results", "scores_new.csv"), index=False)
    print(len(rows), "stories scored")

if __name__ == "__main__":
    main()
