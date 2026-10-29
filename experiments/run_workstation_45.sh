#!/bin/bash
# Experiment 4 on the workstation at low priority with a model already installed in the system Ollama (as run_workstation.sh).
cd "$(dirname "$0")"; mkdir -p out
export OLLAMA_URL=http://127.0.0.1:11434 WORKERS=1 MIN_FREE_GB=30
nice -n 19 ionice -c3 python3 generate2.py exp4 gemma3:27b 2
echo ALL DONE
