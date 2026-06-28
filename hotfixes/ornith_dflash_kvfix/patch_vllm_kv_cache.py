#!/usr/bin/env python3
"""Patch vLLM KV page-size unification for Ornith GDN + DFlash.

This is a container-build hotfix. It edits the installed vLLM
``kv_cache_utils.py`` in the image and does not load model weights.
"""

from __future__ import annotations

from pathlib import Path

import vllm.v1.core.kv_cache_utils as kv_cache_utils


OLD = """            ratio = max_page_size // layer_page_size
            new_block_size = layer_spec.block_size * ratio
            new_spec = replace(layer_spec, block_size=new_block_size)
            assert new_spec.page_size_bytes == max_page_size
            new_kv_cache_spec[layer_name] = new_spec
"""

NEW = """            if isinstance(layer_spec, MambaSpec):
                # Mamba/GDN state pages do not scale with block_size. Pad the
                # backing page instead, matching the hybrid mamba_page_size_padded
                # mechanism used during block-size alignment.
                new_spec = replace(layer_spec, page_size_padded=max_page_size)
            else:
                ratio = max_page_size // layer_page_size
                new_block_size = layer_spec.block_size * ratio
                new_spec = replace(layer_spec, block_size=new_block_size)
            assert new_spec.page_size_bytes == max_page_size
            new_kv_cache_spec[layer_name] = new_spec
"""


def main() -> None:
    path = Path(kv_cache_utils.__file__).resolve()
    text = path.read_text()

    if NEW in text:
        print(f"already patched: {path}")
        return

    if OLD not in text:
        raise SystemExit(f"patch anchor not found in {path}")

    path.write_text(text.replace(OLD, NEW, 1))
    print(f"patched: {path}")


if __name__ == "__main__":
    main()
