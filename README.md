# vllm-dsv4-sm120

Run **DeepSeek-V4-Flash** on **vLLM v0.23.0** on **SM120** (RTX PRO 6000 / Blackwell, cc 12.0) — coherent, with the full feature set, at **~180 tok/s warm single-stream decode and 1M context**.

This is a derived image: it takes the stock `vllm/vllm-openai:v0.23.0` and adds the SM120 sparse-MLA support that upstream vLLM doesn't ship, forward-ported from [lucifer1004]'s SM120 work. The point of this repo is to make **future vLLM upgrades a patch-and-rebuild** instead of a from-scratch re-diagnosis — see [Upgrade guide](#upgrade-guide).

| | |
|---|---|
| Target model | `DeepSeek-V4-Flash` (FP8/MXFP4, sparse MLA + lightning indexer, C4A/C128A hybrid) |
| Base image | `vllm/vllm-openai:v0.23.0` (vllm 0.23.0, py3.12.13, torch 2.11.0+cu130, cuda 13.0) |
| Hardware | 2× RTX PRO 6000 Blackwell, cc 12.0 (sm_120), no NVLink (PCIe) |
| Result | **~180 tok/s** warm single-stream decode (measured: 5×800-tok gen, 170–190 tok/s), 1M ctx, MTP-2 accept-len ~2.1, tool-calling + reasoning OK |
| Baseline | the `lucifer1004/dsv4-flash-sm120` 0.22.x reference image was documented at ~120 tok/s + 1M ctx; this image reuses its SM120 kernels on a newer vLLM (see [What this adds](#what-this-adds-over-lucifer1004dsv4-flash-sm120)) |

## Why stock v0.23.0 doesn't work on SM120

`deep_gemm` IS bundled in v0.23.0 (vendored at `vllm/third_party/deep_gemm`), but several SM120 paths are missing or gated off. Bring-up hit **5 walls**, each a separate fix:

1. **Dense GEMM** — `RuntimeError: dispatch_scaled_mm ... w8a8/cutlass/c3x` in memory profiling. `support_deep_gemm()` (cuda.py) only accepts `is_device_capability(90) or is_device_capability_family(100)`; SM120 (cap 120 → `120//10=12`) falls through to a CUTLASS path with no SM120 block-scaled kernel. Stock DeepGEMM also has no SM120 dense kernels (cmake targets 9.0a/10.0f/10.0a).
2. **Sparse-decode selection** — the DSV4 model **hardcodes** its decode class in `models/deepseek_v4/nvidia/model.py:_select_dsv4_attn_cls` (NOT the pluggable attention-backend priority list — that was a red herring). Default picks the FlashMLA path.
3. **Cudagraph page-size assert** — `kv_cache_utils.py: assert max(sm_page_sizes) <= max(all_page_sizes)`. Only fires with a backend whose full-MLA page size differs from the SWA layers'. Goes away once the SM120 backend (which subclasses the V4 FlashMLA backend) sets consistent page sizes.
4. **Sparse-decode kernel** — `Unsupported architecture for sparse decode fwd`. BOTH stock v0.23.0 DSV4 sparse backends are SM100-only: FlashMLA (`_flashmla_C.sparse_decode_fwd`) and FlashInfer TRTLLM-gen (`trtllm_batch_decode_sparse_mla_dsv4` → `TllmGenFmhaRunner`). The only SM120 sparse-MLA decode is `flashinfer.mla.BatchMLAPagedAttentionWrapper(backend="sparse-sm120").run_sparse_mla(...)`, which stock flashinfer 0.6.12 **does not expose**.
5. **Prefill metadata** — v0.23.0's `DeepseekSparseSWAMetadata` builder computes only **decode** SWA indices; the `run_sparse_mla` prefill path needs `prefill_swa_indices`/`prefill_swa_lens` too.

## The fix (what this image changes)

Four component swaps/adds + two source patches, all driven by the `Dockerfile` + `apply_sm120_patches.py`:

1. **DeepGEMM (SM120 dense GEMM)** — replace vendored `vllm/third_party/deep_gemm` with lucifer1004's SM120 build (has the `sm120_blockwise/blockscaled` MMA kernels stock lacks). ABI-safe (same py/torch/cuda).
2. **flashinfer (SM120 sparse-MLA kernel)** — replace the whole `flashinfer` package with lucifer1004's (same 0.6.12 + added `sparse_mla_sm120.py`, `swa_indices.py`, `run_sparse_mla`, and the extended `BatchMLAPagedAttentionWrapper.__init__`). Still ships `fused_moe`/b12x, so v0.23.0's MoE is intact.
3. **`src/sm120.py`** → installed as `vllm/models/deepseek_v4/nvidia/sm120.py` — the ported SM120 sparse-MLA attention **layer** (`DeepseekV4FlashInferSM120SparseAttention(DeepseekV4Attention)` + `DeepseekV4FlashInferSM120SparseBackend(DeepseekV4FlashMLABackend)`). Adapted from lucifer1004's 0.22.1 backend-impl form to v0.23.0's layer form; `run_sparse_mla` decode+prefill logic kept faithful.
4. **`src/sparse_swa.py`** → installed over `vllm/v1/attention/backends/mla/sparse_swa.py` — adds `prefill_swa_indices`/`prefill_swa_lens` (fields + buffers + `build()` computation via `flashinfer.swa_indices.compute_swa_indices_and_lens` with `token_offset=num_decode_tokens`). **Do not** reuse v0.23.0's local `_compute_swa_indices_and_lens_kernel` for prefill — it's decode-only (`pos = prefix_len + program_id - query_start` treats `program_id` as an absolute token index).
5. **`apply_sm120_patches.py`** — exact-string, assert-on-drift in-place edits:
   - `platforms/cuda.py` `support_deep_gemm()` += `is_device_capability_family(120)`
   - `utils/deep_gemm.py` `should_auto_disable_deep_gemm` + E8M0 oracle += family(120) (SM120 uses packed UE8M0 like datacenter Blackwell)
   - `platforms/cuda.py` `_get_backend_priorities` SM120 branch (cosmetic; not on the DSV4 path)
   - `v1/attention/backends/registry.py` register `FLASHINFER_MLA_SPARSE_SM120_DSV4` (backend advertises `capability.major == 12`)
   - `models/deepseek_v4/nvidia/model.py` `_select_dsv4_attn_cls` routes to the SM120 layer when `is_device_capability_family(120)`
   - `tool_parsers/deepseekv4_tool_parser.py` + `parser/abstract_parser.py` — DSML tool-call-leak fix: disables the strict-tool-call grammar (an EOS-trap where a `<｜DSML｜tool_calls>` opened mid-`<think>` could never terminate → ~100k-token runaway) and recovers tool-calls the model emits inside `<think>` by treating the tool-call start as the end of reasoning (gated to that abnormal case, wrapped to fall back to baseline)

## What this adds over `lucifer1004/dsv4-flash-sm120`

This is **not** a kernel rewrite — the SM120 DeepGEMM and flashinfer `sparse_mla_sm120` are reused from lucifer1004 as-is. The speed comes from the base + serving config, not from faster kernels. What it adds on top:

- **~180 tok/s warm single-stream decode** (measured on this image: `bench.py 5 800` → 170–190 tok/s, avg ~178; short prompt, 800-token generations, MTP-2, temp 0, reasoning on). The 0.22.x reference image was documented at ~120 tok/s. This is **not a controlled head-to-head** (context length and measurement method may differ), but on the same box this image clears the reference figure comfortably.
- **Newer vLLM base — v0.23.0 vs the 0.22.x of the upstream image.** Brings upstream's SM120 b12x MoE + FP4 GEMM, the decoupled DSv4 sparse-MLA metadata, and TRTLLM-gen attention — and keeps tracking upstream.
- **Tuned, documented serving defaults** in `run_dsv4_flash.sh` — the config that produces the speed ladder below: cudagraph `FULL_AND_PIECEWISE` + `custom_ops:["all"]` + async scheduling, MTP-2 (accept-len ~2.1), expert-parallel (`allgather_reducescatter`) + EPLB with redundant experts, `block-size 256`, `gpu-memory-utilization 0.965`, 1M context.
- **Patch-and-rebuild upgrade path.** `apply_sm120_patches.py` asserts each anchor matches exactly once, so the *next* vLLM bump fails loudly at the moved line instead of breaking silently — see the [Upgrade guide](#upgrade-guide). The whole repo exists so a base bump is a re-diff, not a from-scratch re-diagnosis.
- **DSML tool-call-leak fix.** Disables the strict-tool-call EOS-trap grammar (a `<｜DSML｜tool_calls>` opened mid-`<think>` could never terminate → ~100k-token runaway) and recovers tool-calls emitted inside `<think>`. A correctness/robustness fix, gated to the abnormal case and wrapped to fall back to baseline.

> Reproduce the decode number: `BENCH_MODEL=starflinger BENCH_KEY=<key> python bench.py 5 800` against a warm server.

## Quickstart

```bash
# Build (GPU-free, ~2 min; pulls the base + lucifer1004 image for the swaps):
docker build -t dsv4-flash-v023-sm120:latest .

# Run (defaults = cudagraph + MTP-2 + EP + EPLB + 1M ctx, ~180 tok/s warm decode):
./run_dsv4_flash.sh
# health:  curl -H "Authorization: Bearer $KEY" http://localhost:8002/health
```

Run-script env knobs (fast iteration): `IMAGE CONTAINER_NAME ATTN_BACKEND BLOCK_SIZE MAX_LEN GPU_MEM MAX_SEQS ENFORCE_EAGER(0/1) USE_SPEC(0/1) USE_EPLB(0/1) USE_EP(0/1)`. For bring-up/debug use `ENFORCE_EAGER=1 USE_SPEC=0 MAX_LEN=32768` (correctness-only, ~12 tok/s).

Speed ladder (this image): eager+32k ≈ 12 tok/s → +cudagraph ≈ 37 → +MTP-2+EP+1M ≈ **170–190** (warm single-stream decode). First request after boot is slow (cute_dsl + MTP cold-warm), then steady.

## Repo layout

```
Dockerfile                 # base + 4 swaps/adds + patch step + build-time sanity asserts
apply_sm120_patches.py     # in-place source patches (each asserts exactly-one match → fails on drift)
run_dsv4_flash.sh          # serve script (env-parameterized)
src/sm120.py               # ported SM120 sparse-MLA attention layer  (COPY'd into image)
src/sparse_swa.py          # patched SWA metadata builder            (COPY'd into image)
docs/PORT_PLAN.md          # full field→source trace + integration steps + risk list
docs/*.patch.md            # derivation notes per concern (selector, metadata, base classes)
tools/extract_reference_sources.sh   # docker cp the luc + v0.23.0 source for diffing
```

`src/` and `tools/extract_reference_sources.sh` regenerate the vendored reference under `port/` (gitignored).

## Upgrade guide

To move to a future vLLM (e.g. v0.24.x):

1. **Bump the base** in `Dockerfile` (`FROM vllm/vllm-openai:vX.Y.Z`).
2. **Check ABI of the swapped deps.** The DeepGEMM + flashinfer swaps come from `lucifer1004/dsv4-flash-sm120:<tag>`. Confirm the new base image's `python`/`torch`/`cuda` versions still match that tag (`docker run --rm --entrypoint python3 <img> -c "import sys,torch;print(sys.version,torch.__version__,torch.version.cuda)"`). If they diverge, you need a lucifer1004 tag built against the matching toolchain (or rebuild deep_gemm/flashinfer from their source repos).
3. **Re-run the patcher.** `apply_sm120_patches.py` asserts each anchor matches exactly once. If vLLM moved/renamed the code, the build fails loudly naming the file — update that edit's `old` string. (This is the whole point: drift surfaces as a build error, not silent breakage.)
4. **Re-diff the two full-file replacements** (`src/sm120.py`, `src/sparse_swa.py`). Run `tools/extract_reference_sources.sh` to pull the new base's `models/deepseek_v4/nvidia/{flashmla,flashinfer_sparse,model}.py`, `sparse_mla.py`, and `v1/attention/backends/mla/sparse_swa.py`. Diff against the previous `port/v023/*` to see what upstream changed; re-apply our deltas (the SM120 attention class; the `prefill_swa_indices` additions). Watch the metadata field names — `DeepseekSparseSWAMetadata` / `DeepseekV4FlashMLAMetadata` field renames are the most likely break.
5. **Validate.** `ENFORCE_EAGER=1 USE_SPEC=0 MAX_LEN=32768 ./run_dsv4_flash.sh`, then check coherence ("capital of France" → Paris; a 100-token essay), tool-calling (no raw `<｜DSML｜tool_calls>` leak), then enable cudagraph + MTP and benchmark.

### Top correctness risks when re-porting (silent-garbage class — verify with known-answer prompts)
- **`run_sparse_mla` signature/cache-layout** — the decode hinges on flashinfer's keyword contract and on feeding the FlashMLA-packed `fp8_ds_mla` cache + raw `scale` (not the per-tensor-fp8 layout the TRTLLM path uses). Dump the real signature before trusting it.
- **`padded_heads` ladder** — `get_padded_num_q_heads` returns {16,32,64,128}; q-padding, the wrapper, and `attn_sink` length must all agree.
- **prefill SWA-index `token_offset`** — must be `num_decode_tokens` (decodes-first batch ordering); a wrong offset → wrong attention windows.

## Provenance / credits
SM120 DeepGEMM, flashinfer `sparse_mla_sm120`, and the original `nvidia/sm120.py` are from **lucifer1004**'s DeepSeek-V4-Flash SM120 work (image `lucifer1004/dsv4-flash-sm120`, repos: `lucifer1004/{vllm@dsv4-sm120, DeepGEMM@sm120, sparse_mla_sm120}`). This repo forward-ports that onto stock upstream vLLM v0.23.0. vLLM is Apache-2.0.
