#!/usr/bin/env python3
"""Constructor-only check for the Ornith + DFlash KV page hotfix.

This reproduces the issue using small KVCacheSpec objects that match the
Ornith target attention, Qwen3.6-35B-A3B DFlash drafter, and Ornith GDN page
sizes. It does not load model weights or initialize a vLLM engine.
"""

from __future__ import annotations

import torch

from vllm.v1.core.kv_cache_utils import unify_kv_cache_spec_page_size
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    MambaSpec,
    SlidingWindowSpec,
)


def main() -> None:
    specs = {
        # Ornith full-attention page after hybrid alignment with float32 GDN.
        "target_full": FullAttentionSpec(
            block_size=1168,
            num_kv_heads=2,
            head_size=256,
            dtype=torch.bfloat16,
        ),
        # Qwen3.6-35B-A3B DFlash drafter SWA page at the same block size.
        "draft_swa": SlidingWindowSpec(
            block_size=1168,
            num_kv_heads=8,
            head_size=128,
            dtype=torch.bfloat16,
            sliding_window=4096,
        ),
        # Ornith GatedDeltaNet recurrent state, padded by platform alignment.
        "gdn": MambaSpec(
            block_size=1168,
            shapes=((9, 8192), (32, 128, 128)),
            dtypes=(torch.float32, torch.float32),
            page_size_padded=2_392_064,
        ),
    }

    unified = unify_kv_cache_spec_page_size(specs)
    page_sizes = {name: spec.page_size_bytes for name, spec in unified.items()}
    unique_sizes = set(page_sizes.values())
    if len(unique_sizes) != 1:
        raise SystemExit(f"page sizes still differ: {page_sizes}")

    print(f"ornith kvfix ok: common_page_size={unique_sizes.pop()}")


if __name__ == "__main__":
    main()
