#!/bin/bash
# Ratings of the Experiment 4 and 5 stories and of the newest models (Exp. 1) with qwen3:8b on the workstation, low priority,
# queued after run_workstation_newest.sh (the Ollama on the GPU server is too old to download models again).
cd "$(dirname "$0")"
until grep -q "ALL DONE" logs_newest.txt; do sleep 300; done
export OLLAMA_URL=http://127.0.0.1:11434 WORKERS=1
nice -n 19 ionice -c3 python3 prepare_judge_45.py
nice -n 19 ionice -c3 python3 - <<'PY'
import glob, json, os
from masking import mask
with open("judge_input_exp1b.jsonl", "w") as f:
    for p in sorted(glob.glob("out/stories_exp1_gemma4*.jsonl") + glob.glob("out/stories_exp1_qwen3.8*.jsonl")):
        tag = os.path.basename(p)[len("stories_"):-len(".jsonl")]
        for i, l in enumerate(open(p)):
            f.write(json.dumps({"uid": f"{tag}-{i}", "text": mask(json.loads(l)["story_text"])}) + "\n")
PY
nice -n 19 ionice -c3 python3 judge.py judge_input_exp1b.jsonl qwen3:8b
nice -n 19 ionice -c3 python3 judge.py judge_input_45.jsonl qwen3:8b
echo ALL DONE
