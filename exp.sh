#!/usr/bin/env bash
set -euo pipefail

MODELS=(
  "meta-llama/Meta-Llama-3-8B-Instruct"
  "Qwen/Qwen2.5-Coder-7B-Instruct"
  "google/gemma-2-9b-it"
  "microsoft/Phi-3-mini-4k-instruct"
  "microsoft/Phi-3-small-8k-instruct"
)

DOMAINS=("airline" "retail")
AGENT_LLM_ARGS='{"temperature":1e-5,"max_new_tokens":512,"device_map":"auto","dtype":"auto"}'

for model in "${MODELS[@]}"; do
  for domain in "${DOMAINS[@]}"; do
    echo "Running domain=${domain}, model=${model}"
    tau2 run \
      --domain "${domain}" \
      --agent-llm-backend transformers \
      --agent-llm "${model}" \
      --agent-llm-args "${AGENT_LLM_ARGS}" \
      --user-llm-backend litellm \
      --user-llm gpt-4.1 \
      --num-trials 1 \
      --task-split-name base
  done
done
