#!/usr/bin/env bash
#
# serve_ornith.sh — serve Ornith-1.0-35B-AEON-Ultimate-Uncensored on the unified
# AEON vLLM Ultimate container (ghcr.io/aeon-7/aeon-vllm-ultimate:latest,
# vLLM 0.23.0 built from source for GB10 / sm_121a).
#
# Base: deepreinforce-ai/Ornith-1.0-35B  (arch qwen3_5_moe; true lineage Qwen3.6-35B-A3B)
#   256 fused experts + shared (top-8), 40 layers (30 GatedDeltaNet + 10 full-attn),
#   hidden 2048, vision tower, 256K ctx. Reasoning model (<think> every turn).
#   NO mtp.* tensors shipped -> no speculative decoding head.
#
# Usage:
#   ./serve_ornith.sh [profile]
#     profile = solo       (default) LLM is the only GPU workload  -> gpu-mem 0.85
#               colocated  leave headroom for ASR/TTS/embeddings   -> gpu-mem 0.70
#
# Env overrides (all optional):
#   MODEL_DIR   host path to the model dir            (default ./aeon-model)
#   IMAGE       container image                       (default ghcr.io/aeon-7/aeon-vllm-ultimate:latest)
#   PORT        host port for the OpenAI endpoint     (default 8000)
#   SERVED_NAME --served-model-name alias             (default ornith-ultimate)
#   MAX_LEN     --max-model-len                       (default 262144)
#   DTYPE       bfloat16 | fp8 (use fp8 for the FP8 weights repo) (default bfloat16)
#
set -euo pipefail

PROFILE="${1:-solo}"
MODEL_DIR="${MODEL_DIR:-./aeon-model}"
IMAGE="${IMAGE:-ghcr.io/aeon-7/aeon-vllm-ultimate:latest}"
PORT="${PORT:-8000}"
SERVED_NAME="${SERVED_NAME:-ornith-ultimate}"
MAX_LEN="${MAX_LEN:-262144}"
DTYPE="${DTYPE:-bfloat16}"

# ---- profile -> gpu-memory-utilization ----
case "$PROFILE" in
  solo)
    GPU_MEM=0.85   # LLM is the only GPU workload. GB10 ceiling; NEVER exceed 0.88.
    ;;
  colocated|co-located|coloc)
    GPU_MEM=0.70   # safe default with ASR/TTS/embedding side services on the same Spark.
    ;;
  *)
    echo "unknown profile '$PROFILE' (use: solo | colocated)" >&2
    exit 2
    ;;
esac

# Resolve to an absolute path so the bind-mount is unambiguous.
MODEL_DIR="$(cd "$MODEL_DIR" && pwd)"
if [[ ! -f "$MODEL_DIR/config.json" ]]; then
  echo "no config.json in $MODEL_DIR — point MODEL_DIR at the downloaded model dir" >&2
  exit 1
fi

echo "=== serve_ornith.sh ==="
echo "  profile      : $PROFILE  (gpu-memory-utilization=$GPU_MEM)"
echo "  image        : $IMAGE"
echo "  model dir    : $MODEL_DIR"
echo "  dtype        : $DTYPE"
echo "  max-model-len: $MAX_LEN"
echo "  endpoint     : http://localhost:$PORT/v1   (served as '$SERVED_NAME')"
echo "======================="

# ---- GB10 / sm_121a env (mirrors the AEON Spark recipe) ----
#   TORCH_CUDA_ARCH_LIST=12.1a   compile/dispatch to the SM120-family GB10 kernels
#   ENABLE_NVFP4_SM100=0         GB10 is sm_121a, not B200 sm_100 (avoid dead kernels)
#   VLLM_USE_FLASHINFER_SAMPLER=1  FlashInfer sampler (faster on GB10)
#   VLLM_NVFP4_GEMM_BACKEND=flashinfer-cutlass  CUTLASS NVFP4 GEMM (only used by FP4 weights)
#   NVIDIA_FORWARD_COMPAT=1      forward-compat for the GB10 driver/runtime split
GB10_ENV=(
  -e TORCH_CUDA_ARCH_LIST=12.1a
  -e ENABLE_NVFP4_SM100=0
  -e VLLM_USE_FLASHINFER_SAMPLER=1
  -e VLLM_NVFP4_GEMM_BACKEND=flashinfer-cutlass
  -e NVIDIA_FORWARD_COMPAT=1
)

# ---- vLLM serve flags (from the project CTX serve recipe) ----
#   BF16 KV is FORCED by the vision tower (non-causal attn -> FA2 only -> no FP4/FP8 KV
#   backend exists for it), so we do NOT pass --kv-cache-dtype.
#   No --speculative-config: the base ships no mtp.* head and no DFlash drafter yet.
SERVE_ARGS=(
  serve /model
  --served-model-name "$SERVED_NAME"
  --dtype "$DTYPE"
  --reasoning-parser qwen3
  --tool-call-parser qwen3_xml
  --enable-auto-tool-choice
  --mamba-cache-dtype float32
  --max-model-len "$MAX_LEN"
  --max-num-batched-tokens 16384
  --gpu-memory-utilization "$GPU_MEM"
  --enable-chunked-prefill
  --enable-prefix-caching
  --trust-remote-code
  --host 0.0.0.0
  --port 8000
)

# If serving the FP8 weights repo, tell vLLM to read the compressed-tensors config.
if [[ "$DTYPE" == "fp8" ]]; then
  SERVE_ARGS=( "${SERVE_ARGS[@]}" --quantization compressed-tensors )
fi

exec docker run --rm --gpus all --ipc host --network host \
  "${GB10_ENV[@]}" \
  -v "$MODEL_DIR:/model:ro" \
  --entrypoint vllm \
  "$IMAGE" \
  "${SERVE_ARGS[@]}"
