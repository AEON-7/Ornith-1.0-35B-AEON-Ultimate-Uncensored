ARG BASE_IMAGE=ghcr.io/aeon-7/aeon-vllm-ultimate:latest
FROM ${BASE_IMAGE}

COPY patch_vllm_kv_cache.py /tmp/patch_vllm_kv_cache.py
COPY verify_ornith_kvfix.py /tmp/verify_ornith_kvfix.py

RUN python3 /tmp/patch_vllm_kv_cache.py \
    && python3 /tmp/verify_ornith_kvfix.py \
    && rm -f /tmp/patch_vllm_kv_cache.py /tmp/verify_ornith_kvfix.py

LABEL org.opencontainers.image.title="AEON vLLM Ornith DFlash KV hotfix"
LABEL org.opencontainers.image.description="Pads Mamba/GDN KV pages during vLLM page-size unification so Ornith NVFP4 + DFlash can initialize."
