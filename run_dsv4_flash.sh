#!/bin/bash
# DeepSeek-V4-Flash on vLLM v0.23.0, SM120 (RTX PRO 6000).
# Default IMAGE = our SM120-patched build (deep_gemm sm120 + gate patches,
# see dsv4-v023-sm120/Dockerfile). Sparse decode routed to the flashinfer
# TRTLLM path via --attention-backend FLASHINFER_MLA_SPARSE_DSV4.
#
# Env knobs (for fast iteration):
#   IMAGE, CONTAINER_NAME, ATTN_BACKEND, BLOCK_SIZE, MAX_LEN, GPU_MEM, MAX_SEQS
#   ENFORCE_EAGER(0/1)  USE_SPEC(0/1)  USE_EPLB(0/1)  USE_EP(0/1)
#   VLLM_API_KEY  -- bearer token the server requires (set your own; default is a placeholder)
set -euo pipefail

CONTAINER_NAME="${CONTAINER_NAME:-dsv4-flash-v023sm120}"
IMAGE="${IMAGE:-dsv4-flash-v023-sm120:latest}"
API_KEY="${VLLM_API_KEY:-changeme-set-VLLM_API_KEY}"
ATTN_BACKEND="${ATTN_BACKEND:-FLASHINFER_MLA_SPARSE_SM120_DSV4}"
BLOCK_SIZE="${BLOCK_SIZE:-256}"
MAX_LEN="${MAX_LEN:-1048576}"
GPU_MEM="${GPU_MEM:-0.965}"
MAX_SEQS="${MAX_SEQS:-1}"
ENFORCE_EAGER="${ENFORCE_EAGER:-0}"
USE_SPEC="${USE_SPEC:-1}"
USE_EPLB="${USE_EPLB:-1}"
USE_EP="${USE_EP:-1}"

docker stop "$CONTAINER_NAME" 2>/dev/null || true
docker rm "$CONTAINER_NAME" 2>/dev/null || true
mkdir -p /home/ubuntu/cache/dsv4-v023

ARGS=(
  serve /models/DeepSeek-V4-Flash
  --host 0.0.0.0 --port 8002
  --served-model-name starflinger
  --api-key "$API_KEY"
  --trust-remote-code
  --attention-backend "$ATTN_BACKEND"
  --kv-cache-dtype fp8
  --block-size "$BLOCK_SIZE"
  --tensor-parallel-size 2
  --gpu-memory-utilization "$GPU_MEM"
  --max-model-len "$MAX_LEN"
  --max-num-seqs "$MAX_SEQS"
  --no-enable-flashinfer-autotune
  --enable-prefix-caching
  --tokenizer-mode deepseek_v4
  --tool-call-parser deepseek_v4
  --enable-auto-tool-choice
  --structured-outputs-config '{"enable_in_reasoning":true}'
  --reasoning-parser deepseek_v4
  --reasoning-config '{"reasoning_parser":"deepseek_v4","reasoning_start_str":"<think>","reasoning_end_str":"</think>"}'
  --default-chat-template-kwargs '{"enable_thinking":true}'
)
if [ "$ENFORCE_EAGER" = "1" ]; then
  ARGS+=(--enforce-eager)
else
  ARGS+=(--compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE","custom_ops":["all"]}' --async-scheduling)
fi
[ "$USE_EP" = "1" ]   && ARGS+=(--enable-expert-parallel --all2all-backend allgather_reducescatter)
[ "$USE_EPLB" = "1" ] && ARGS+=(--enable-eplb --eplb-config '{"num_redundant_experts":2,"use_async":true}')
[ "$USE_SPEC" = "1" ] && ARGS+=(--speculative-config '{"method":"mtp","num_speculative_tokens":2}')

exec docker run --name "$CONTAINER_NAME" \
  --log-driver=journald \
  --gpus all \
  --network host \
  --ipc host \
  -v /home/ubuntu/models/DeepSeek-V4-Flash:/models/DeepSeek-V4-Flash:ro \
  -v /home/ubuntu/cache/dsv4-v023:/cache \
  -e CUDA_VISIBLE_DEVICES=0,1 \
  -e VLLM_ENFORCE_STRICT_TOOL_CALLING=1 \
  -e VLLM_CACHE_ROOT=/cache/vllm \
  -e FLASHINFER_WORKSPACE_BASE=/cache/flashinfer \
  -e TRITON_CACHE_DIR=/cache/triton \
  -e TILELANG_CACHE_DIR=/cache/tilelang \
  --entrypoint vllm \
  "$IMAGE" \
  "${ARGS[@]}"
