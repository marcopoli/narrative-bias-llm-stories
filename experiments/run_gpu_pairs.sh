#!/bin/bash
# Pairwise agency judgements (Experiment 5) on two shared GPUs, same setup as run_gpu.sh. The Ollama on the GPU server
# is too old for the Ollama registry, so the rater is taken from the official Qwen GGUF release on Hugging Face.
cd "$(dirname "$0")"; mkdir -p logs out models
export OLLAMA_HOST=127.0.0.1:11555 OLLAMA_URL=http://127.0.0.1:11555
export OLLAMA_MODELS="$PWD/models" CUDA_VISIBLE_DEVICES="${GPUS:-6,7}"
export OLLAMA_NUM_PARALLEL=4 OLLAMA_MAX_LOADED_MODELS=1 OLLAMA_KEEP_ALIVE=5m WORKERS=4 OLLAMA_CONTEXT_LENGTH=8192
nice -n 19 ionice -c3 ollama serve > logs/ollama.log 2>&1 &
SERVER=$!; trap 'kill $SERVER 2>/dev/null' EXIT; sleep 20
M=hf.co/Qwen/Qwen3-8B-GGUF:Q4_K_M
used=$(quota -w 2>/dev/null | awk 'NR==3{gsub(/\*/,"",$2); print $2}')
[ -z "$used" ] || [ "$used" -lt 8800000 ] || { echo "quota too high, stopping"; exit 1; }
for i in 1 2 3; do timeout 1800 ollama pull $M > logs/pull.txt 2>&1 && break; echo "pull failed: $(tail -c 300 logs/pull.txt)"; sleep 30; done
ollama list | grep -q Qwen3-8B || { echo "no model, stopping"; exit 1; }
echo "pulled $M ($(du -sh models | cut -f1))"
nice -n 19 ionice -c3 python3 judge_pairs.py run $M
ollama rm $M > /dev/null; rm -rf models/*
echo "models removed ($(du -sh . | cut -f1))"
echo ALL DONE
