#!/bin/bash
# Ratings of the Experiment 1 stories (masked) with qwen3:8b on two shared GPUs, same setup as run_gpu.sh.
cd "$(dirname "$0")"; mkdir -p logs out models
export OLLAMA_HOST=127.0.0.1:11555 OLLAMA_URL=http://127.0.0.1:11555
export OLLAMA_MODELS="$PWD/models" CUDA_VISIBLE_DEVICES="${GPUS:-6,7}"
export OLLAMA_NUM_PARALLEL=4 OLLAMA_MAX_LOADED_MODELS=1 OLLAMA_KEEP_ALIVE=5m WORKERS=4
nice -n 19 ionice -c3 ollama serve > logs/ollama.log 2>&1 &
SERVER=$!; trap 'kill $SERVER 2>/dev/null' EXIT; sleep 20
used=$(quota -w 2>/dev/null | awk 'NR==3{gsub(/\*/,"",$2); print $2}')
[ -z "$used" ] || [ "$used" -lt 8800000 ] || { echo "quota too high, stopping"; exit 1; }
until timeout 900 ollama pull qwen3:8b > /dev/null 2>&1; do echo "pull interrupted, retrying"; sleep 15; done
echo "pulled qwen3:8b ($(du -sh models | cut -f1))"
nice -n 19 ionice -c3 python3 judge.py judge_input_exp1.jsonl qwen3:8b
ollama rm qwen3:8b > /dev/null; rm -rf models/*
echo "models removed ($(du -sh . | cut -f1))"
echo ALL DONE
