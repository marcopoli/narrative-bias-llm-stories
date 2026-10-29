#!/bin/bash
# Experiments 2 and 3 on two shared GPUs: private low-priority Ollama instance; small models kept one at a
# time inside ./models (user quota 15 GB) and deleted after use.
cd "$(dirname "$0")"; mkdir -p logs out models
export OLLAMA_HOST=127.0.0.1:11555 OLLAMA_URL=http://127.0.0.1:11555
export OLLAMA_MODELS="$PWD/models" CUDA_VISIBLE_DEVICES="${GPUS:-6,7}"
export OLLAMA_NUM_PARALLEL=4 OLLAMA_MAX_LOADED_MODELS=1 OLLAMA_KEEP_ALIVE=5m WORKERS=4
nice -n 19 ionice -c3 ollama serve > logs/ollama.log 2>&1 &
SERVER=$!; trap 'kill $SERVER 2>/dev/null' EXIT; sleep 10
run() { nice -n 19 ionice -c3 python3 "$@"; }
quota_ok() { used=$(quota -w 2>/dev/null | awk 'NR==3{gsub(/\*/,"",$2); print $2}'); [ -z "$used" ] || [ "$used" -lt 8800000 ]; }
use() {
  quota_ok || { echo "quota too high before pulling $1, stopping"; exit 1; }
  until timeout 900 ollama pull "$1" > /dev/null 2>&1; do echo "pull of $1 interrupted, retrying"; sleep 15; done
  echo "pulled $1 ($(du -sh models | cut -f1))"
}
drop() { ollama rm "$1" > /dev/null; rm -rf models/blobs/*-partial*; echo "removed $1 ($(du -sh models | cut -f1))"; }

use mistral:7b
run generate.py exp2 mistral:7b 4 original ordinary_day development
drop mistral:7b
use qwen3:8b
run prepare_judge_new.py
run judge.py judge_input_original.jsonl qwen3:8b
run judge.py judge_input_new.jsonl qwen3:8b
drop qwen3:8b
use granite3.3:8b
run judge.py judge_input_original.jsonl granite3.3:8b
drop granite3.3:8b
echo ALL DONE
