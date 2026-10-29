"""Writes the masked texts of the Experiment 4 and 5 stories to judge_input_45.jsonl (runs on the server)."""
import glob, json, os
from masking import mask, mask_ext

HERE = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(HERE, "judge_input_45.jsonl"), "w") as f:
    n = 0
    for path in sorted(glob.glob(os.path.join(HERE, "out", "stories_exp[45]_*.jsonl"))):
        tag = os.path.basename(path)[len("stories_"):-len(".jsonl")]
        m = mask_ext if tag.startswith("exp4") else mask
        for i, line in enumerate(open(path)):
            d = json.loads(line)
            f.write(json.dumps({"uid": f"{tag}-{i}", "text": m(d["story_text"])}) + "\n"); n += 1
print(n, "stories written")
