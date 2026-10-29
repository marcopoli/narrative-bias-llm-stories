"""Draws a stratified sample of the original corpus for the LLM ratings (5 stories per persona-model unit, 1,125
stories) and writes the masked texts to judge_input_original.jsonl."""
import json, os, sqlite3
import pandas as pd
from masking import mask

HERE = os.path.dirname(os.path.abspath(__file__))
con = sqlite3.connect(os.path.join(HERE, "..", "analysis", "master_results.db"))
df = pd.read_sql_query("SELECT r.id, r.model_name, r.persona_id, r.story_text FROM results r", con)
sample = df.groupby(["model_name", "persona_id"], group_keys=False).apply(lambda g: g.sample(5, random_state=42))
with open(os.path.join(HERE, "judge_input_original.jsonl"), "w") as f:
    for r in sample.itertuples():
        f.write(json.dumps({"uid": f"orig-{r.id}", "text": mask(r.story_text)}) + "\n")
print(len(sample), "stories written")
