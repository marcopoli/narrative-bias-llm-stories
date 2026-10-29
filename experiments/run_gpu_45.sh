#!/bin/bash
# Experiments 4 and 5 on two shared GPUs, same setup as run_gpu.sh: private low-priority Ollama instance,
# one small model at a time inside ./models (user quota 15 GB), deleted after use.
cd "$(dirname "$0")"; mkdir -p logs out models
export OLLAMA_HOST=127.0.0.1:11555 OLLAMA_URL=http://127.0.0.1:11555
export OLLAMA_MODELS="$PWD/models" CUDA_VISIBLE_DEVICES="${GPUS:-6,7}"
export OLLAMA_NUM_PARALLEL=4 OLLAMA_MAX_LOADED_MODELS=1 OLLAMA_KEEP_ALIVE=5m WORKERS=4
nice -n 19 ionice -c3 ollama serve > logs/ollama.log 2>&1 &
SERVER=$!; trap 'kill $SERVER 2>/dev/null' EXIT; sleep 20
run() { nice -n 19 ionice -c3 python3 "$@"; }
quota_ok() { used=$(quota -w 2>/dev/null | awk 'NR==3{gsub(/\*/,"",$2); print $2}'); [ -z "$used" ] || [ "$used" -lt 8800000 ]; }
use() {
  for i in $(seq 20); do quota_ok && break; sleep 30; done   # the quota needs some time to see deleted models
  quota_ok || { echo "quota too high before pulling $1, stopping"; exit 1; }
  until timeout 900 ollama pull "$1" > /dev/null 2>&1; do echo "pull of $1 interrupted, retrying"; sleep 15; done
  echo "pulled $1 ($(du -sh models | cut -f1))"
}
drop() { ollama rm "$1" > /dev/null; rm -rf models/blobs/*-partial*; echo "removed $1 ($(du -sh models | cut -f1))"; }

for m in ${MODELS:-mistral:7b llama3:8b}; do
  use $m; run generate2.py exp5 $m 4; run generate2.py exp4 $m 2; drop $m
done
use gemma3:4b; run generate2.py exp4 gemma3:4b 2; drop gemma3:4b
use qwen3:8b
run prepare_judge_45.py
run judge.py judge_input_45.jsonl qwen3:8b
drop qwen3:8b; rm -rf models/*
echo ALL DONE
