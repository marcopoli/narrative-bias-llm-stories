"""Would a standard bias check catch the agency gap? Scores every story with the two default measurements of the
Hugging Face `evaluate` library that are used for bias audits of generated text: regard (Sheng et al., 2019;
sasha/regardv3) and toxicity (Vidgen et al., 2021; facebook/roberta-hate-speech-dynabench-r4-target).
Both classifiers work on short texts, so each story is split into sentences and the sentence scores are averaged
(regard) or the maximum is kept (toxicity). Stories already in results/standard_audit.csv are skipped.
Output: results/standard_audit.csv. Run in an environment with spaCy and en_core_web_lg (see requirements.txt).
"""
import glob, json, os, re, sqlite3
import pandas as pd, torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
DEV = "mps" if torch.backends.mps.is_available() else "cpu"

def load(name):
    tok = AutoTokenizer.from_pretrained(name)
    mod = AutoModelForSequenceClassification.from_pretrained(name).to(DEV).eval()
    return tok, mod

def probs(tok, mod, sents, bs=64):
    out = []
    for i in range(0, len(sents), bs):
        enc = tok(sents[i:i + bs], padding=True, truncation=True, max_length=128, return_tensors="pt").to(DEV)
        with torch.no_grad():
            out.append(torch.softmax(mod(**enc).logits, -1).cpu())
    return torch.cat(out) if out else torch.zeros(0, mod.config.num_labels)

def sentences(text):
    return [s for s in re.split(r"(?<=[.!?])\s+", re.sub(r"\s+", " ", text)) if len(s.split()) >= 3]

def stories():
    con = sqlite3.connect(os.path.join(HERE, "..", "analysis", "master_results.db"))
    for i, t in con.execute("SELECT id, story_text FROM results"):
        yield f"orig-{i}", t
    for path in sorted(glob.glob(os.path.join(HERE, "out", "stories_exp*.jsonl"))):
        tag = os.path.basename(path)[len("stories_"):-len(".jsonl")]
        for i, line in enumerate(open(path)):
            yield f"{tag}-{i}", json.loads(line)["story_text"]

def main():
    rtok, rmod = load("sasha/regardv3")
    ttok, tmod = load("facebook/roberta-hate-speech-dynabench-r4-target")
    rl = {v.lower(): k for k, v in rmod.config.id2label.items()}
    tl = {v.lower(): k for k, v in tmod.config.id2label.items()}
    path = os.path.join(HERE, "results", "standard_audit.csv")
    old = pd.read_csv(path) if os.path.exists(path) else pd.DataFrame(columns=["uid"])
    done, rows = set(old.uid), []
    for n, (uid, text) in enumerate(stories(), 1):
        if uid in done: continue
        s = sentences(text)
        r, t = probs(rtok, rmod, s), probs(ttok, tmod, s)
        rows.append(dict(uid=uid, n_sent=len(s),
                         regard_pos=float(r[:, rl["positive"]].mean()), regard_neg=float(r[:, rl["negative"]].mean()),
                         regard=float((r[:, rl["positive"]] - r[:, rl["negative"]]).mean()),
                         tox_max=float(t[:, tl["hate"]].max()) if len(s) else 0.0, tox_mean=float(t[:, tl["hate"]].mean())))
        if n % 500 == 0: print(n, "stories", flush=True)
    pd.concat([old, pd.DataFrame(rows)], ignore_index=True).to_csv(path, index=False)
    print(len(rows), "new stories scored,", len(old) + len(rows), "in total")

if __name__ == "__main__":
    main()
