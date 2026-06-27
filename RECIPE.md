# Reproduction recipe

Base: deepreinforce-ai/Ornith-1.0-35B (qwen3_5_moe; verified init = Qwen3.6-35B-A3B).

1. **SSM conv1d repair** — detect linear_attn.conv1d blocks with sigma > 1.5x median (Ornith: layers 36,37), rescale by median/sigma.
2. **Abliterate** — abliterix v1.9 with `config/ornith_35b.toml` (NPBA `projected_abliteration` + winsorize 0.995 + gaussian decay, EGA on the 256 fused experts + shared expert + router suppression, Q/K/V disabled). On a GB10 (121GB unified) run the SEARCH in NF4 (bnb leaves fused experts BF16). Optuna refusals-vs-KL.
3. **Select the gentle knee** — NOT the lowest-refusal trial (over-abliterates -> word-salad). Pick the lightest expert edit that still removes refusals; verify coherence on long (256-tok) generation.
4. **Export bf16** on a box where 70GB fits in memory (cloud H200 141GB; GB10 cannot edit-in-memory): `abliterix export_model.py --trial <N>`.
5. **Quant FP8** — llm-compressor `QuantizationModifier(scheme="FP8_DYNAMIC")` ignoring `linear_attn.*`, `mlp.gate`, `shared_expert_gate`, embeds, lm_head, norms, visual (never FP8 the GatedDeltaNet/routing). compressed-tensors -> vLLM-serveable.

Validated: 0/80 refusals, 0 coding-capability loss (agentic pass@1 0.833 = base), KL ~0.0014.
