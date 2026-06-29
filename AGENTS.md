# AGENTS.md — deploying & operating Ornith-1.0-35B AEON Ultimate Uncensored

Instructions for an AI agent tasked with standing up and using this model. **Follow the steps in order and respect the ⛔ guardrails — several of them prevent a hard host crash, not just a failed launch.**

---

## 0. What you are deploying

- **Model:** uncensored / abliterated `Ornith-1.0-35B` — base is `Qwen/Qwen3.6-35B-A3B` (verified by weight-correlation; the upstream "Qwen 3.5" label is misleading).
- **Architecture:** `qwen3_5_moe` — **hybrid**: 40 layers = 30 GatedDeltaNet (linear-attn / Mamba-style recurrent) + 10 full-attention; 256 routed experts + 1 shared (A3B ≈ 3B active/token); vision tower; **256K context**; **thinking model** (emits `<think>…</think>` every turn); SOTA **agentic-coding**.
- **Variants:**
  | Repo | Size | Precision | Runs on |
  |---|---|---|---|
  | `AEON-7/Ornith-1.0-35B-AEON-Ultimate-Uncensored-BF16` | ~66 GB | bf16 | any vLLM, any modern GPU |
  | `AEON-7/Ornith-1.0-35B-AEON-Ultimate-Uncensored-NVFP4` | ~23.7 GB | NVFP4 (W4A16, MLP-only) | **Blackwell only** (B200 sm_100 / GB10 sm_120) |
- **Drafter (speculative decoding):** `AEON-7/AEON-DFlash-Qwen3.6-35B-A3B` — all-full-attention, **drop-in, no patch**. Gives ~**1.9×** single-stream. (Do **not** substitute a sliding-window drafter without the hotfix — see Troubleshooting.)

## 1. Detect the environment BEFORE doing anything

1. **GPU arch.** `nvidia-smi --query-gpu=name,compute_cap --format=csv`. Blackwell (sm_100 / sm_120 / sm_121) → NVFP4 is available. Anything older (Hopper sm_90, Ada, Ampere) → **NVFP4 will not run; use BF16.**
2. **Free memory.** NVFP4 + DFlash peaks ~**80 GB**; BF16 needs ~66 GB of weights + KV. On a **DGX Spark the 121 GB is unified (CPU+GPU share one pool)** — `free -g` is the real budget. If another large model is running (e.g. a voice gateway), **stop it first** and confirm `free -g` ≥ ~100 GB before serving.
3. **Container.** `docker pull ghcr.io/aeon-7/aeon-vllm-ultimate:latest`. It carries `qwen3_5_moe` + sm_121a kernels + DFlash. **Stock vLLM is not recommended on the Spark for this arch.**

## 2. Download the model + drafter

```bash
huggingface-cli download AEON-7/Ornith-1.0-35B-AEON-Ultimate-Uncensored-NVFP4 --local-dir ~/models/ornith-nvfp4
huggingface-cli download AEON-7/AEON-DFlash-Qwen3.6-35B-A3B --local-dir ~/models/ornith-dflash-drafter
# BF16 instead, on non-Blackwell:
# huggingface-cli download AEON-7/Ornith-1.0-35B-AEON-Ultimate-Uncensored-BF16 --local-dir ~/models/ornith-bf16
```

## 3. Serve — OPTIMAL config (DGX Spark / GB10)

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

> **Aliases for a clean cutover.** `--served-model-name` accepts a list — every name maps to this one model. Replacing another model? List the name your existing clients already request (e.g. `--served-model-name ornith my-previous-model-name aeon-ultimate`) so they switch to Ornith with **zero client-side reconfiguration** — just point the names at the model.

| Flag | Why |
|---|---|
| `--quantization compressed-tensors` | NVFP4 is `nvfp4-pack-quantized`. (Omit entirely for the BF16 variant.) |
| `--speculative-config … num_speculative_tokens 6` | DFlash; **n=6 is the measured optimum** (n is not a memory lever — keep 6). Omit if you have no Blackwell drafter. |
| `--gpu-memory-utilization 0.6` | Validated-stable with DFlash; leaves headroom for the OS + co-located services + DFlash's *un-budgeted* verify buffers. Never exceed ~0.7 with DFlash; hard ceiling 0.88. |
| `--max-num-seqs 16` | **HARD CAP with DFlash on the Spark** (see guardrails). |
| `--mamba-cache-dtype float32` | GatedDeltaNet recurrent-state precision. |
| `--attention-backend flash_attn` | **Required.** Vision (non-causal) and DFlash (non-causal block-verify) both need FA2; auto-selection can pick a backend that can't serve them → engine-init crash. |
| `--limit-mm-per-prompt '{"image":4,"video":2}'` + `--mm-encoder-tp-mode data` | Enables vision (image/video) inputs — Ornith is multimodal. |
| `--enable-chunked-prefill` | Smooth long-context prefill; pairs with prefix caching. Default-on in v1; keep explicit. |
| `--enable-prefix-caching` | ~**12.6× faster prefill on repeated context** (multi-turn / agentic / shared system prompts). Default-on in the image; keep it. |
| `--reasoning-parser qwen3` | Parses the `<think>` block. |
| `--enable-auto-tool-choice --tool-call-parser qwen3_coder` | Tool/function calling. |
| `--served-model-name ornith …` | List every model name your clients already request → transparent cutover (see note above). |
| *(do NOT add)* `--kv-cache-dtype fp8` | The vision tower + DFlash force **BF16 KV**; FP8 KV crashes at init. |

## 4. Verify it serves

```bash
curl -fsS http://localhost:8000/v1/models   # -> served model "ornith"
curl -s http://localhost:8000/v1/chat/completions -H 'content-type: application/json' -d '{
  "model":"ornith","messages":[{"role":"user","content":"Write a Rust function returning the median of a slice."}],
  "max_tokens":600,"temperature":0.6,"top_p":0.95,"top_k":20}' | python3 -m json.tool
```
Healthy startup log shows `Application startup complete` and **zero** `not found in params_dict` / weight-skip warnings.

## 5. Use the model

- **Thinking model:** every reply opens `<think>…</think>`; the user-facing answer follows `</think>`. Give generous `max_tokens` (≥1024) or responses get cut off mid-reasoning with empty `content`.
- **Sampling:** `temperature 0.6, top_p 0.95, top_k 20`.
- **Vision:** image/video inputs supported (BF16 KV on vision deploys). Tool-calling via the `qwen3_coder` parser. Context up to 262144.
- **Multi-turn / agentic:** keep the conversation/system prefix stable across turns so prefix caching hits (huge prefill savings). DFlash accelerates decode; prefix caching accelerates prefill — they compose.

## ⛔ Guardrails — DO NOT

1. **Do NOT raise `--max-num-seqs` above ~16 with DFlash on a DGX Spark.** DFlash's speculative-verify buffers are not fully counted by `--gpu-memory-utilization`; at 32–64 they exhaust the 121 GB **unified** pool and **hard-crash the whole host** (kernel `NVRM … NV_ERR_NO_MEMORY`, reboot). For high batch throughput, **disable DFlash** (plain NVFP4 scales to higher concurrency safely) — don't raise the cap.
2. **Do NOT exceed `--gpu-memory-utilization 0.7`** (0.6 recommended with DFlash; 0.88 is the absolute ceiling — unified memory thrashes above it).
3. **Do NOT use `--kv-cache-dtype fp8`** — incompatible with the vision/non-causal + DFlash path; crashes at init.
4. **Do NOT use FP8 quantization** — not viable on this hybrid arch (W8A8 degrades coherence; W8A16 has no ScaledMM kernel). NVFP4 is the low-precision path.
5. **Do NOT pair a sliding-window-attention (SWA) DFlash drafter** (e.g. `z-lab/Qwen3.6-35B-A3B-DFlash`) with the stock image — its larger KV page can't unify with the GatedDeltaNet pages → `unify_kv_cache_spec_page_size AssertionError` at init. Use the **AEON all-full-attn drafter** (default), or build the hotfix image (`hotfixes/ornith_dflash_kvfix/`).
6. **Do NOT serve alongside another large model** on a Spark without first stopping it and confirming `free -g` headroom.

## Troubleshooting

| Symptom | Cause → Fix |
|---|---|
| `unify_kv_cache_spec_page_size … AssertionError` at engine init | SWA drafter's KV page can't unify with GDN. → Use the AEON all-full-attn drafter, or `hotfixes/ornith_dflash_kvfix/build_image.sh` then serve with that image. |
| Host **rebooted** / `NVRM … NV_ERR_NO_MEMORY` | DFlash + high `max-num-seqs` exhausted unified memory. → `max-num-seqs ≤ 16`, `gpu-memory-utilization 0.6`. |
| `Failed to find a kernel … ScaledMM` / FP8 won't load | FP8 not supported here. → Use NVFP4 (or BF16). |
| Empty `content` in responses | Thinking model truncated inside `<think>`. → raise `max_tokens` (≥1024). |
| Serve OOMs at load on a shared box | Another model holds memory. → stop it; verify `free -g` ≥ ~100 GB. |

## Performance you should expect (DGX Spark GB10, NVFP4 + DFlash n=6)

- Single-stream: **~74 tok/s** decode, ~90 ms TTFT. Aggregate: ~456 tok/s @ c=16.
- Prefill on repeated context with prefix caching: **~12.6×** faster than cold re-prefill.
- Capability: agentic-coding pass@1 **0.833** (identical to BF16); **0 refusals**; coherent.

## Responsibility

This is an **uncensored, agentic** model — it will comply with harmful requests, and its outputs are often tool calls / code that downstream systems execute. **You, the deployer, are the safety layer** (implement input/output filtering, audit logging, access controls, and human-in-the-loop for autonomous pipelines). Read the full [User Responsibility & Arbitration Clause](https://huggingface.co/AEON-7/Ornith-1.0-35B-AEON-Ultimate-Uncensored-BF16#user-responsibility--arbitration-clause) before deploying.
