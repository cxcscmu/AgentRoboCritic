#!/usr/bin/env bash
set -euo pipefail

model=cx-cmu/AgentRoboCritic-Muse-Glimmer-30B
revision=5f478c5db3cacc1d1f2232e7ed3bc1a99b8e3753
exec vllm serve "$model" \
    --revision "$revision" --served-model-name "$model" \
    --host 127.0.0.1 --port "${PORT:-8000}" \
    --dtype bfloat16 --tensor-parallel-size "${TENSOR_PARALLEL_SIZE:-2}" \
    --max-model-len 131072 --max-num-seqs 1 --max-num-batched-tokens 8192 \
    --gpu-memory-utilization 0.90 --enforce-eager \
    --limit-mm-per-prompt '{"image":35,"video":0}' \
    --generation-config vllm --reasoning-parser muse_glimmer
