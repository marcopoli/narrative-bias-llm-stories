#!/bin/bash
# Experiment 1 on the workstation at low priority: models already installed in the system Ollama, one request at a time,
# paused while free memory is below 30 GB so that other jobs keep priority.
cd "$(dirname "$0")"; mkdir -p out
export OLLAMA_URL=http://127.0.0.1:11434 WORKERS=1 MIN_FREE_GB=30
for m in gemma3:27b mistral-small3.2:24b gpt-oss:20b; do
  nice -n 19 ionice -c3 python3 generate.py exp1 "$m" 4
done
echo ALL DONE
