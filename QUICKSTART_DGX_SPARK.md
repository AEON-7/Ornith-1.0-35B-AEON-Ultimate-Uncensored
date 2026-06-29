# 🚀 DGX Spark QuickStart — Ornith-1.0-35B AEON Ultimate Uncensored (NVFP4 + DFlash)

> **Built and tuned for the NVIDIA DGX Spark — GB10 (Blackwell sm_120), 128 GB unified memory.**
> Optimal deploy: the **NVFP4** build (23.7 GB) + a **DFlash** drafter for **~1.9× single-stream** speedup, served on the AEON vLLM container.

This guide is **specifically for the DGX Spark**. The settings below (especially `--gpu-memory-utilization 0.7` and `--mamba-cache-dtype float32`) are chosen for the Spark's unified-memory GB10; other GPUs may want different values.

---

## 1. Prerequisites
- **NVIDIA DGX Spark** (GB10, Blackwell). NVFP4 requires a Blackwell GPU.
- Docker with the NVIDIA container runtime.
- AEON vLLM container: `ghcr.io/aeon-7/aeon-vllm-ultimate:latest` (has `qwen3_5_moe` + sm_121a + DFlash support — **stock vLLM is not recommended on the Spark for this arch**).
- ~25 GB disk for the model, ~1 GB for the drafter.

## 2. Download the model + DFlash drafter
```bash
huggingface-cli download AEON-7/Ornith-1.0-35B-AEON-Ultimate-Uncensored-NVFP4 \
  --local-dir ~/models/ornith-nvfp4
# Recommended drafter — all-full-attention, drop-in, NO patch needed:
huggingface-cli download AEON-7/AEON-DFlash-Qwen3.6-35B-A3B \
  --local-dir ~/models/ornith-dflash-drafter
```

> **Drafter choice.** Use the **AEON DFlash** drafter above — it's all-full-attention, so it serves on the stock AEON image with **no patch**, and benchmarks slightly *higher acceptance* (3.71 vs 3.35 tokens/step) than the sliding-window `z-lab/Qwen3.6-35B-A3B-DFlash` drafter. The z-lab drafter also works but its SWA layers trip a KV page-size assert on this hybrid arch (see [issue #1](https://github.com/AEON-7/Ornith-1.0-35B-AEON-Ultimate-Uncensored/issues/1)) — to use it, build the hotfix image in [`hotfixes/ornith_dflash_kvfix/`](hotfixes/ornith_dflash_kvfix/) (thanks @newjordan) and set `ghcr.io/...` → that image.

## 3. Serve — optimal DGX Spark settings
```bash
docker run -d --name ornith --gpus all --ipc=host --net=host \
  -e TORCH_CUDA_ARCH_LIST=12.1a -e CUTE_DSL_ARCH=sm_121a -e VLLM_USE_FLASHINFER_SAMPLER=1 -e VLLM_ENABLE_CUDA_COMPATIBILITY=0 \
  -v ~/models/ornith-nvfp4:/model:ro \
  -v ~/models/ornith-dflash-drafter:/drafter:ro \
  --entrypoint vllm ghcr.io/aeon-7/aeon-vllm-ultimate:latest \
  serve /model --served-model-name ornith aeon-ultimate aeon-fast aeon-deep \
  --quantization compressed-tensors \
  --speculative-config '{"method":"dflash","model":"/drafter","num_speculative_tokens":6}' \
  --gpu-memory-utilization 0.6 \
  --max-model-len 262144 --max-num-seqs 16 --max-num-batched-tokens 16384 \
  --mamba-cache-dtype float32 \
  --reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_coder \
  --limit-mm-per-prompt '{"image":4,"video":2}' --mm-encoder-tp-mode data \
  --attention-backend flash_attn \
  --enable-chunked-prefill --enable-prefix-caching --trust-remote-code
```

> **Aliases for a clean cutover.** `--served-model-name` takes a list — every name resolves to this one model. List whatever model name your existing clients already request (e.g. replacing another model: `--served-model-name ornith my-previous-model-name aeon-ultimate`) and they switch to Ornith with **zero client-side changes** — just point the names at the model.

## 4. Why these settings (DGX Spark specifics)
| Flag | Value | Why on the Spark |
|---|---|---|
| `--gpu-memory-utilization` | **0.6** | The Spark's 128 GB is **unified** (CPU+GPU share it). **0.6** is the validated-stable value with DFlash on — it leaves ~41 GB headroom for the OS, co-located services, *and* DFlash's speculative-verify buffers (which are **not** fully counted in this budget — see Stability below). 0.7 is fine on a dedicated box; do not exceed ~0.7 with DFlash. |
| `--max-num-seqs` | **16** ⚠️ | **Hard cap on the Spark with DFlash.** DFlash's per-sequence draft+verify buffers scale with `max-num-seqs`; at 32–64 they spike past the memory budget and can exhaust the unified pool → **hard box crash**. 16 is stable (peak 80/121 GB). For higher batch throughput, **disable DFlash** rather than raising this. |
| `--speculative-config dflash … n=6` | **n=6** | Optimal single-stream (peaks at 6; higher n wastes draft/verify on low-acceptance positions). Validated stable at `max-num-seqs 16` (n barely affects peak memory there). The Qwen3.6-35B-A3B drafter works because Ornith is a light RL post-train of that base. |
| `--mamba-cache-dtype` | **float32** | Precision for the GatedDeltaNet (SSM) recurrent state. |
| `--quantization` | **compressed-tensors** | The NVFP4 build is `nvfp4-pack-quantized`. |
| *(omit)* `--kv-cache-dtype fp8` | — | The vision tower forces **BF16 KV** (FP8 KV is incompatible with the non-causal vision/DFlash path). |
| `--max-model-len` | 262144 | Full 256K context fits thanks to NVFP4's small footprint. Lower it to free KV memory if you don't need long context. |
| `--attention-backend` | **flash_attn** | **Required.** Vision (non-causal) *and* DFlash (non-causal block-verify) both need FA2; letting vLLM auto-select can pick a backend that can't serve them → engine-init crash. |
| `--limit-mm-per-prompt '{"image":4,"video":2}'` + `--mm-encoder-tp-mode data` | vision | Ornith is multimodal — these enable image/video inputs. |
| `--reasoning-parser qwen3` + `--enable-auto-tool-choice --tool-call-parser qwen3_coder` | parsers | Thinking-block parsing + function/tool calling. Keep both — universally needed for chat + agentic/tool use. |
| `--enable-chunked-prefill` | on | Smooth long-context prefill; pairs with prefix caching. |
| `--served-model-name` | aliases | List every model name your clients already request, all pointing at this one model (see the cutover note above). |

## 5. Smoke test
```bash
curl -s http://localhost:8000/v1/chat/completions -H 'content-type: application/json' -d '{
  "model":"ornith","messages":[{"role":"user","content":"Write a Rust function that returns the median of a slice."}],
  "max_tokens":400,"temperature":0.6,"top_p":0.95,"top_k":20}' | python3 -m json.tool
```
Reasoning model: every reply opens `<think>…</think>`. Recommended sampling: **temperature 0.6, top_p 0.95, top_k 20**.

## 6. Performance (measured on DGX Spark GB10, single-stream)

Same Spark, same prompts, three stacks — so you can see the **quantization** win and the **optimization** win separately:

| Workload | BF16 · stock vLLM | NVFP4 · stock vLLM | **NVFP4 + DFlash · AEON (this guide)** |
|---|---|---|---|
| Coding | 30.8 tok/s · 237 ms | 38.5 · 70 ms | **77.1 · 94 ms** |
| Reasoning | 30.6 · 247 ms | 38.4 · 77 ms | **107.0 · 93 ms** |
| Math | 30.5 · 221 ms | 38.3 · 72 ms | **119.0 · 88 ms** |
| Prose | 30.4 · 193 ms | 38.3 · 69 ms | **70.3 · 91 ms** |
| **Avg decode** | 30.6 | 38.4 | **93.3** |
| Prefill | 3,517 tok/s | 5,203 | **9,661** |

- **Quantization** (BF16 → NVFP4): ~**1.25× decode**, **3.2× faster TTFT**, **1.5× prefill** — and a 64% smaller footprint (66 → 24 GB).
- **Optimization** (NVFP4 → + DFlash on the AEON container): ~**2.4× decode** (DFlash; biggest on math/reasoning).
- **Combined vs a naive BF16 deploy:** **3.05× decode · 2.5× TTFT · 2.75× prefill.** Largest at c=1; tapers under heavy concurrency.

### Concurrency / throughput (NVFP4 + DFlash n=6, `max-num-seqs 16` — the safe envelope)

| Concurrency | aggregate tok/s |
|---|---|
| c=1 (per user) | **70** |
| c=8 | 203 |
| c=16 | **456** |

Validated **stable through c=16** with ~41 GB unified-memory headroom (peak **80 / 121 GB**, zero watchdog kills). DFlash wins single-stream 1.78× (70 vs 39 tok/s) and carries no penalty up to the seqs=16 ceiling.

> ⚠️ **Stability — do not raise `--max-num-seqs` above ~16 with DFlash on the Spark.** DFlash's per-sequence speculative-verify buffers are **not** fully accounted by `--gpu-memory-utilization`; at a cap of 32–64 they spike past the budget and exhaust the 121 GB **unified** pool, which **hard-crashes the whole box** (kernel `NVRM … NV_ERR_NO_MEMORY`). If you need higher batch throughput, run **plain NVFP4 (no DFlash)** — it scales to higher concurrency safely — rather than raising `max-num-seqs` with DFlash on.

## Notes
- **Long-running services:** DFlash drafter acceptance can decay over many hours of continuous traffic; restarting the container restores it (or run a periodic health-check that does).
- **Uncensored:** safety refusals are removed — you are responsible for use; obey applicable law.
