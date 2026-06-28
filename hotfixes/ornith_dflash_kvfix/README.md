# Ornith + DFlash KV Page Hotfix

This hotfix is for DGX Spark runs that fail during vLLM engine init with:

```text
AssertionError in unify_kv_cache_spec_page_size
```

The failure is not caused by weights, argument parsing, or OOM. Ornith mixes
full attention with GatedDeltaNet recurrent state, and the DFlash drafter adds
sliding-window attention layers with a larger KV page. vLLM can increase
attention `block_size` to unify page sizes, but Mamba/GatedDeltaNet state pages
are shape-based and do not scale with `block_size`.

The patch pads smaller `MambaSpec` pages to the common max page size instead of
trying to scale them by `block_size`.

Build a local patched image:

```bash
./hotfixes/ornith_dflash_kvfix/build_image.sh
```

Then use the patched image in the QuickStart command:

```bash
ORNITH_VLLM_IMAGE=aeon-vllm-ornith-dflash-kvfix:local
```

The image build runs a constructor-only verifier for the problematic KV page
mix. It does not load model weights or initialize a vLLM engine.
