# DeepSeek-V4-Flash on vLLM v0.23.0, made to run on SM120 (RTX PRO 6000).
#
# Mirrors the working lucifer1004 image, ported onto v0.23.0:
#   1. Swap vendored DeepGEMM for lucifer1004's SM120 build (dense block-scaled
#      FP8 GEMM kernels the stock copy lacks).
#   2. Swap flashinfer for lucifer1004's (same 0.6.12 + the sparse-sm120 MLA
#      kernel: run_sparse_mla + flashinfer/sparse_mla_sm120.py). ABI-safe: both
#      images are py3.12.13 / torch 2.11.0+cu130 / cuda13.0.
#   3. Drop in the ported nvidia/sm120.py sparse-MLA attention layer.
#   4. Patch: open deep_gemm SM120 gates; register FLASHINFER_MLA_SPARSE_SM120_DSV4;
#      route _select_dsv4_attn_cls to the SM120 layer on capability major==12.
#
# Build:  docker build -t dsv4-flash-v023-sm120:latest /home/ubuntu/services/inference/dsv4-v023-sm120
# Run:    bash run_dsv4_flash_v0230.sh   (defaults to this image)

FROM vllm/vllm-openai:v0.23.0

ARG LUC=lucifer1004/dsv4-flash-sm120:20260604
ARG SP=/usr/local/lib/python3.12/dist-packages

# 1. SM120 DeepGEMM (dense GEMM).
RUN rm -rf $SP/vllm/third_party/deep_gemm
COPY --from=lucifer1004/dsv4-flash-sm120:20260604 \
     /opt/env/lib/python3.12/site-packages/vllm/third_party/deep_gemm \
     /usr/local/lib/python3.12/dist-packages/vllm/third_party/deep_gemm

# 2. flashinfer with the sparse-sm120 MLA kernel (additive over stock 0.6.12).
RUN rm -rf $SP/flashinfer
COPY --from=lucifer1004/dsv4-flash-sm120:20260604 \
     /opt/env/lib/python3.12/site-packages/flashinfer \
     /usr/local/lib/python3.12/dist-packages/flashinfer

# 3. The ported SM120 sparse-MLA attention layer + the SWA metadata builder
#    extended to compute prefill_swa_indices/prefill_swa_lens (run_sparse_mla
#    prefill needs them; stock v0.23.0 only builds decode SWA indices).
COPY src/sm120.py \
     /usr/local/lib/python3.12/dist-packages/vllm/models/deepseek_v4/nvidia/sm120.py
COPY src/sparse_swa.py \
     /usr/local/lib/python3.12/dist-packages/vllm/v1/attention/backends/mla/sparse_swa.py

# 4. Open the gates + register/route the SM120 backend (asserts on source drift).
COPY apply_sm120_patches.py /tmp/apply_sm120_patches.py
RUN python3 /tmp/apply_sm120_patches.py && rm /tmp/apply_sm120_patches.py

# Sanity: deep_gemm bundled, sm120 module + backend enum import, selector present.
RUN python3 -c "\
from vllm.utils.import_utils import has_deep_gemm; assert has_deep_gemm();\
from vllm.v1.attention.backends.registry import AttentionBackendEnum as E; assert hasattr(E,'FLASHINFER_MLA_SPARSE_SM120_DSV4');\
import flashinfer.mla as m; assert hasattr(m.BatchMLAPagedAttentionWrapper,'run_sparse_mla');\
print('sm120 build sanity OK')"
