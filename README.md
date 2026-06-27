# Ornith-1.0-35B — AEON Ultimate Uncensored

Uncensored / abliterated build of [`deepreinforce-ai/Ornith-1.0-35B`](https://huggingface.co/deepreinforce-ai/Ornith-1.0-35B), DeepReinforce's SOTA agentic-coding MoE — **refusals removed, capability preserved.**

| | |
|---|---|
| **BF16** | [`AEON-7/Ornith-1.0-35B-AEON-Ultimate-Uncensored-BF16`](https://huggingface.co/AEON-7/Ornith-1.0-35B-AEON-Ultimate-Uncensored-BF16) |
| License | MIT (inherited from base) |

## TL;DR
- **0 / 80 refusals (0.0%)** on diverse harmful prompts (base ≈ 94/100) — fully uncensored.
- **0 coding-capability loss** — agentic pass@1 **0.833, identical to the base** family-by-family. Near-lossless (KL ≈ 0.0014).
- Hybrid `qwen3_5_moe`: 40 layers (30 GatedDeltaNet + 10 full-attn), 256 experts + shared (A3B), vision, 256K ctx, thinking model.

## Quickstart (vLLM)
```bash
vllm serve AEON-7/Ornith-1.0-35B-AEON-Ultimate-Uncensored-BF16 \
  --served-model-name ornith --max-model-len 262144 \
  --gpu-memory-utilization 0.85 --max-num-batched-tokens 16384 \
  --mamba-cache-dtype float32 --reasoning-parser qwen3 \
  --enable-prefix-caching --trust-remote-code
```
Thinking model (`<think>…</think>` every turn). Sampling: `temperature 0.6, top_p 0.95, top_k 20`. Vision intact (BF16 KV on vision deploys). See [`serve_ornith.sh`](serve_ornith.sh).

## How it was built (4 stages)
1. **SSM `conv1d` outlier repair** — rescale outlier blocks (layers 36/37) pre-abliteration to prevent coherence collapse.
2. **Abliteration** — `abliterix` v1.9: grimjim norm-preserving biprojection + **Expert-Granular Abliteration** across all 256 fused experts + shared expert + router suppression; Optuna refusals-vs-KL search. Q/K/V untouched (`attn_output_gate`); GatedDeltaNet/SSM internals + vision tower **not modified**. Recipe: [`config/ornith_35b.toml`](config/ornith_35b.toml).
3. **Gentle-knee selection** — the lowest-refusal trial was *over-abliterated* (word-salad on real generation). Shipped a 170× lighter expert edit for the same refusal removal, verified coherent.
4. **Export** bf16 on a box where 70 GB fits in memory (cloud H200; GB10's 121 GB unified pool cannot edit-in-memory). **Quantization note:** FP8 is *not viable* on this hybrid GatedDeltaNet-MoE via standard vLLM — W8A8 FP8 degrades coherence (FP8 activations hurt the reasoning path); W8A16 weight-only FP8 has no ScaledMM kernel. NVFP4 on Blackwell (B200) is the proven low-precision path for this family.

## Validation
| Metric | Base Ornith-1.0-35B | This model |
|---|---|---|
| Refusals (80 harmful: CBRN/cyber/weapons/self-harm) | high | **0 / 80 (0.0%)** |
| Agentic/coding pass@1 (18-task probe) | 0.833 | **0.833 (identical)** |
| First-token KL vs base | — | **~0.0014** |

Base capability context (preserved, per the coding-delta check): Terminal-Bench 2.1 = 64.2, SWE-bench Verified = 75.6.

## Responsibility
Safety refusals removed — **will comply with harmful requests.** For research (alignment, red-teaming, uncensored assistants). You are solely responsible for use; obey applicable law. No warranty.

## Provenance
Base [deepreinforce-ai/Ornith-1.0-35B](https://huggingface.co/deepreinforce-ai/Ornith-1.0-35B) · driver [abliterix](https://github.com/wuwangzhang1216/abliterix) / [heretic](https://github.com/p-e-w/heretic) · methods grimjim (NPBA), Arditi et al. 2024, FernflowerAI (SSM repair) · build AEON-7.
