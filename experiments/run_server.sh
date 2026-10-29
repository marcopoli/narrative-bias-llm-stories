#!/bin/bash
# Runs Experiments 1-3 on two shared GPUs with a private, low-priority Ollama instance.
# Models are stored in ./models (inside this folder), pulled one at a time and deleted after use.
cd "$(dirname "$0")"; mkdir -p logs out models
export OLLAMA_HOST=127.0.0.1:11555 OLLAMA_URL=http://127.0.0.1:11555
export OLLAMA_MODELS="$PWD/models" CUDA_VISIBLE_DEVICES="${GPUS:-6,7}"
export OLLAMA_NUM_PARALLEL=4 OLLAMA_MAX_LOADED_MODELS=1 OLLAMA_KEEP_ALIVE=5m WORKERS=4
nice -n 19 ionice -c3 ollama serve > logs/ollama.log 2>&1 &
SERVER=$!; trap 'kill $SERVER 2>/dev/null' EXIT; sleep 10
run() { nice -n 19 ionice -c3 python3 "$@"; }
use() {  # pull with a watchdog: a stalled download is restarted and resumes from the partial file
  until timeout 900 ollama pull "$1" > /dev/null 2>&1; do echo "pull of $1 interrupted, retrying"; sleep 15; done
  echo "pulled $1 ($(du -sh models | cut -f1))"
}
drop() { ollama rm "$1" > /dev/null && echo "removed $1 ($(du -sh models | cut -f1))"; }

use mistral-small3.2:24b
run generate.py exp1 mistral-small3.2:24b 6
run generate.py exp2 mistral-small3.2:24b 4 ordinary_day development
drop mistral-small3.2:24b
use gemma3:27b
run generate.py exp1 gemma3:27b 6
drop gemma3:27b
use qwen3:32b
run generate.py exp1 qwen3:32b 6
run judge.py judge_input_original.jsonl qwen3:32b
drop qwen3:32b
use gpt-oss:20b
run prepare_judge_new.py
run judge.py judge_input_original.jsonl gpt-oss:20b
run judge.py judge_input_new.jsonl gpt-oss:20b
drop gpt-oss:20b
echo ALL DONE
