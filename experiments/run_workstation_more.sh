#!/bin/bash
# Extra runs on the workstation at low priority, queued after run_workstation_judge.sh (models already installed):
# the equal-agency instruction (Experiment 5, "parity") on Gemma 4 31B, its ratings, and a larger rater
# (qwen3:32b) on a stratified subset of the main corpus for rater robustness.
cd "$(dirname "$0")"
until grep -q "ALL DONE" logs_judge.txt; do sleep 300; done
export OLLAMA_URL=http://127.0.0.1:11434 WORKERS=1 MIN_FREE_GB=30
nice -n 19 ionice -c3 python3 generate2.py exp5 gemma4:31b 4 parity
nice -n 19 ionice -c3 python3 - <<'PY'
import json
from masking import mask
with open("judge_input_exp5g.jsonl", "w") as f:
    for i, l in enumerate(open("out/stories_exp5_gemma4-31b.jsonl")):
        f.write(json.dumps({"uid": f"exp5_gemma4-31b-{i}", "text": mask(json.loads(l)["story_text"])}) + "\n")
PY
nice -n 19 ionice -c3 python3 judge.py judge_input_exp5g.jsonl qwen3:8b
nice -n 19 ionice -c3 python3 judge.py judge_input_robust.jsonl qwen3:32b
echo ALL DONE
