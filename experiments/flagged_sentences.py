"""What do the toxicity flags in the main corpus point at? Collects every sentence with a hate probability above 0.5
(the default decision threshold) and checks whether it contains a persona label (job, gender, nationality) or a
cleaning word. Output: results/flagged_sentences.csv and results/flagged_janitor_sample30.csv
(30 random janitor sentences for the manual check reported in Section 4.4). Run in an environment with spaCy and en_core_web_lg (see requirements.txt)."""
import os, re, sqlite3
import pandas as pd
from standard_audit import load, probs, sentences, HERE
from masking import PATTERNS

CLEAN = r"\b(mop\w*|sweep\w*|broom\w*|trash|garbage|rubbish|waste|dirt\w*|clean\w*|scrub\w*|bucket\w*|vacuum\w*|spill\w*|stain\w*|toilet\w*|restroom\w*|dust\w*)\b"

con = sqlite3.connect(os.path.join(HERE, "..", "analysis", "master_results.db"))
df = pd.read_sql_query("SELECT r.id, r.model_name AS model, p.profession, r.story_text FROM results r JOIN personas p ON p.id = r.persona_id", con)
tok, mod = load("facebook/roberta-hate-speech-dynabench-r4-target")
rows = []
for _, r in df.iterrows():
    ss = sentences(r.story_text)
    p = probs(tok, mod, ss)[:, 1]
    for s, q in zip(ss, p.tolist()):
        if q > 0.5:
            rows.append(dict(id=r.id, model=r.model, profession=r.profession, p_hate=q, sentence=s,
                             label=any(re.search(pat, s, re.I) for pat, _ in PATTERNS),
                             cleaning=bool(re.search(CLEAN, s, re.I))))
out = pd.DataFrame(rows)
out.to_csv(os.path.join(HERE, "results", "flagged_sentences.csv"), index=False)
out["label_or_cleaning"] = out.label | out.cleaning
print(len(out), "flagged sentences")
print(out.groupby("profession")[["label", "cleaning", "label_or_cleaning"]].mean().round(3))
print(out.groupby("profession").size())
# random sample of flagged janitor sentences for the manual hate-speech check
out[out.profession == "Janitor"].sample(30, random_state=5)[["id", "p_hate", "sentence"]].to_csv(
    os.path.join(HERE, "results", "flagged_janitor_sample30.csv"), index=False)
