#!/bin/bash
# Newest models on the workstation at low priority (already installed in the system Ollama), one request at a time:
# Gemma 4 31B in Experiments 1 and 4, then Qwen 3.8 27B in Experiment 1.
cd "$(dirname "$0")"; mkdir -p out
export OLLAMA_URL=http://127.0.0.1:11434 WORKERS=1 MIN_FREE_GB=30
nice -n 19 ionice -c3 python3 generate.py exp1 gemma4:31b 4
nice -n 19 ionice -c3 python3 generate2.py exp4 gemma4:31b 2
nice -n 19 ionice -c3 python3 generate.py exp1 qwen3.8:27b 4
echo ALL DONE
