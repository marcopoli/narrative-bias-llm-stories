"""Writes the masked texts of the new stories (Experiments 1 and 2) to judge_input_new.jsonl (runs on the server)."""
import glob, json, os
from masking import mask

HERE = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(HERE, "judge_input_new.jsonl"), "w") as f:
    n = 0
    for path in sorted(glob.glob(os.path.join(HERE, "out", "stories_exp*.jsonl"))):
        tag = os.path.basename(path)[len("stories_"):-len(".jsonl")]
        for i, line in enumerate(open(path)):
            d = json.loads(line)
            f.write(json.dumps({"uid": f"{tag}-{i}", "text": mask(d["story_text"])}) + "\n"); n += 1
print(n, "stories written")
