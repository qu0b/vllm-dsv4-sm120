# PORT_PLAN — DeepSeek-V4-Flash SM120 sparse-MLA on vLLM v0.23.0

Goal: serve DeepSeek-V4-Flash on RTX PRO 6000 (SM120) by forward-porting
lucifer1004's 0.22.1 `nvidia/sm120.py` onto v0.23.0. The only sparse-MLA decode
path that runs on sm12x is
`flashinfer.mla.BatchMLAPagedAttentionWrapper(backend="sparse-sm120").run_sparse_mla(...)`.

## Architecture delta that drove the port

0.22.1 ran V4 attention through a *backend impl object*
(`DeepseekV4SparseMLAAttentionImpl`, reached the layer via `self.layer`) and
piggy-backed on the V3.2 `FlashMLASparseMetadata`. v0.23.0 made V4 attention a
*model layer* (`DeepseekV4Attention` subclasses) and gave V4 its own
metadata/backend module `models/deepseek_v4/sparse_mla.py`
(`DeepseekV4FlashMLABackend` / `DeepseekV4FlashMLAMetadata` /
`DeepseekV4FlashMLAMetadataBuilder`), into which the entire C128A computation was
moved verbatim. The port is therefore a **layer subclass**, modeled on the
existing v0.23.0 `DeepseekV4FlashInferMLAAttention` (`nvidia/flashinfer_sparse.py`).

## Ordered integration steps

1. **Drop in the port file.** Copy `out/sm120.py` to
   `vllm/models/deepseek_v4/nvidia/sm120.py`. It defines
   `DeepseekV4FlashInferSM120SparseBackend(DeepseekV4FlashMLABackend)` and
   `DeepseekV4FlashInferSM120SparseAttention(DeepseekV4Attention)`.
   - No edits to `flashmla.py`, `sparse_mla.py`, `flashmla_sparse.py`,
     `sparse_swa.py` are required (see the two `.patch.md` "NO PATCH NEEDED"
     findings).

2. **Register the backend.** Edit
   `vllm/v1/attention/backends/registry.py`: add the
   `FLASHINFER_MLA_SPARSE_SM120_DSV4` enum member (see
   `model.selector.patch.md` Edit 1).

3. **Route the selector.** Edit `vllm/models/deepseek_v4/model.py`
   `_select_dsv4_attn_cls` to return
   `DeepseekV4FlashInferSM120SparseAttention` when
   `current_platform.is_device_capability_family(120)` (or when the new enum is
   explicitly passed). See `model.selector.patch.md` Edit 2. Import of `sm120`
   is function-local to keep non-sm120 imports clean.

4. **Confirm the SWA builder already skips FlashMLA planning on sm12x.**
   `sparse_swa.py:build_tile_scheduler` (lines 432-464) returns all-`None`
   `tile_sched_*` when `current_platform.is_device_capability_family(120)`. The
   SM120 impl never reads `tile_sched_*` — it uses the FlashInfer wrapper — so
   this is already correct and needs no change. (If this guard were missing, the
   FlashMLA planner `get_mla_metadata()` would run pointlessly; it would not
   corrupt output but wastes a host call.)

5. **No KV-cache layout change.** `DeepseekV4FlashMLABackend.get_kv_cache_shape`
   returns the 584-byte `fp8_ds_mla` row for V4; the SM120 backend inherits it.
   The SWA cache shape comes from `DeepseekSparseSWABackend` (also 584B fp8_ds_mla)
   — unchanged. The wrapper consumes both via `.unsqueeze(-2)` exactly as the
   FlashMLA path does.

## Every file touched

| file | change |
|---|---|
| `vllm/models/deepseek_v4/nvidia/sm120.py` | **new** (the port) |
| `vllm/v1/attention/backends/registry.py` | +1 enum member |
| `vllm/models/deepseek_v4/model.py` | `_select_dsv4_attn_cls` body |

Files inspected and confirmed **unchanged**: `sparse_mla.py`,
`nvidia/flashmla.py`, `v1/.../flashmla_sparse.py`, `sparse_swa.py`,
`nvidia/flashinfer_sparse.py`.

## Field/attr trace — every thing `sm120.py` reads, and where v0.23.0 fills it

| read in `out/sm120.py` | source in v0.23.0 |
|---|---|
| `self.compress_ratio`, `.window_size`, `.max_model_len`, `.max_num_batched_tokens`, `.padded_heads`, `.scale`, `.attn_sink`, `.topk_indices_buffer`, `.prefix`, `.swa_cache_layer`, `.kv_cache` | `DeepseekV4Attention` layer attrs (also read identically by `flashmla.py`/`flashinfer_sparse.py` off `self`) |
| `attn_metadata.block_size`, `.block_table`, `.c128a_global_decode_topk_indices`, `.c128a_decode_topk_lens`, `.c128a_prefill_topk_indices` | `DeepseekV4FlashMLAMetadata` populated by `DeepseekV4FlashMLAMetadataBuilder.build` / `_build_c128a_metadata` (`sparse_mla.py`) |
| `swa_metadata.num_decodes/num_prefills/num_decode_tokens/num_prefill_tokens`, `.is_valid_token`, `.token_to_req_indices`, `.decode_swa_indices`, `.decode_swa_lens`, `.prefill_swa_indices`, `.prefill_swa_lens`, `.query_start_loc_cpu` | `DeepseekSparseSWAMetadata` populated by `DeepseekSparseSWAMetadataBuilder.build` (`sparse_swa.py`) — all fields present, confirmed |
| `compute_global_topk_indices_and_lens(...)` (C4A decode + prefill global-slot conversion) | `vllm.models.deepseek_v4.common.ops` — present (also used by `flashmla.py`) |
| `current_workspace_manager().get_simultaneous(...)` | `vllm.v1.worker.workspace` — present |

---

# RISK / UNCERTAINTY LIST

Ordered by "wrong guess → silent garbage" first (hardest to catch on GPU),
then "wrong guess → crash" (loud, easy to catch).

## A. GARBAGE-OUTPUT risks (no crash; verify with a known-answer/perplexity check)

1. **`run_sparse_mla` argument semantics / signature drift (HIGHEST).** The whole
   port hinges on flashinfer 0.6.12's
   `BatchMLAPagedAttentionWrapper.run_sparse_mla` keyword contract:
   `q, kv_cache, sparse_indices, out, sm_scale, sparse_lengths, sinks,
   extra_kv_cache, extra_sparse_indices, extra_sparse_lengths, mid_out, mid_lse`.
   This is copied verbatim from luc and was validated against 0.6.12 by luc, but
   it is *not* something v0.23.0 exercises. If any arg was renamed/re-ordered, or
   if `sm_scale` here must already fold a per-tensor fp8 scale (as
   `flashinfer_sparse.py` does for `bmm1_scale`), output is silently wrong. The
   port passes the raw `self.scale` (bf16-cache assumption). **Action: dump the
   real `run_sparse_mla` signature from the installed flashinfer before trusting
   this.**

2. **bf16 vs per-tensor-fp8 cache assumption for the wrapper.** luc's sm120 path
   feeds `self.scale` directly and `self.attn_sink` directly, implying the
   sparse-sm120 wrapper reads the **FlashMLA-packed fp8_ds_mla** cache (UE8M0
   block-scaled, the 584-byte row), NOT the per-tensor-fp8 layout that
   `flashinfer_sparse.py` uses (which needs `_flashinfer_fp8_bmm1_scale`). The
   port keeps the FlashMLA cache layout (backend subclasses
   `DeepseekV4FlashMLABackend`, so `use_flashmla_fp8_layout` is **not** set to
   `False` the way `DeepseekV4FlashInferMLAAttention` sets it). **If the
   sparse-sm120 wrapper actually expects the de-scaled/per-tensor layout, the KV
   bytes are mis-interpreted → garbage.** This is the single most likely
   correctness trap because it is a cross-image assumption. Confirm which cache
   layout `run_sparse_mla` decodes.

3. **`.unsqueeze(-2)` KV-cache reshaping.** Both decode and prefill do
   `swa_cache = self.swa_cache_layer.kv_cache.unsqueeze(-2)` and
   `kv_cache.unsqueeze(-2)` to turn `(num_blocks, block_size, head_bytes)` into
   `(num_blocks, block_size, 1, head_bytes)` while preserving strides. This is
   identical to v0.23.0's FlashMLA path (`flashmla.py:190,193`), so the cache
   *shape* is right. Risk: the sparse-sm120 wrapper may want the singleton head
   dim in a different position than `-2`, or may want `.view(torch.uint8)` first
   (the FP8 decode kernel in `flashmla_sparse.py:945` does
   `.view(torch.uint8).unsqueeze(-2)`). Port omits the `view(torch.uint8)` — if
   the wrapper needs raw bytes, add it. Wrong → garbage.

4. **`block_size // compress_ratio` for C4A global-index conversion.** In
   `_forward_decode`/`_forward_prefill`, `block_size = attn_metadata.block_size
   // self.compress_ratio` feeds `compute_global_topk_indices_and_lens`.
   `attn_metadata.block_size` is the V4 256-token block; for C4A (ratio 4) the
   compressed block is 64. This matches `flashmla.py:162,310` exactly, so it is
   correct *as long as* `DeepseekV4FlashMLAMetadata.block_size` is the storage
   block size (256), not an already-compressed value. Confirmed: the builder sets
   `block_size=self.kv_cache_spec.block_size` (sparse_mla.py:240), the uncompressed
   256. Low risk but flagged because an off-by-`compress_ratio` here produces
   valid-looking but wrong slot ids → garbage.

5. **`padded_heads` value / `get_padded_num_q_heads` for the wrapper.** The
   wrapper is constructed with `max_num_heads=self.padded_heads`, and the impl's
   `get_padded_num_q_heads` returns {16,32,64,128} (luc's sm120 ladder), which is
   **finer-grained** than the V4 FlashMLA ladder ({64,128}). The outer V4 wrapper
   pads q/output to `self.padded_heads`. If `self.padded_heads` is computed by the
   *parent* using the FlashMLA {64,128} ladder rather than this impl's ladder,
   then for a TP that yields e.g. 16 local heads the wrapper is sized for 16 but q
   arrives padded to 64 (or vice-versa) → shape/stride mismatch that may not
   crash if 64≥16 but wastes work, or → garbage if the wrapper indexes heads it
   wasn't given. **Verify `padded_heads` on the constructed layer equals
   `get_padded_num_q_heads(n_local_heads)` of *this* class.** (In luc this worked
   because the impl's classmethod fed the layer's `padded_heads`; confirm the
   v0.23.0 layer plumbing calls the subclass's classmethod, not the base's.)

6. **`attn_sink` shape vs padded heads.** `self.attn_sink` is a
   `(padded_heads,)` `-inf` parameter (attention.py:758-761) sized via
   `get_deepseek_v4_padded_num_q_heads(n_local_heads)`. If that helper uses the
   FlashMLA ladder while the wrapper uses the {16,32,...} ladder (risk #5), the
   sink vector length won't match the wrapper's head count → masked-wrong or
   garbage. Tie this check to #5.

## B. CRASH risks (loud; easy to catch in first boot/decode)

7. **`PREFILL_CHUNK_SIZE` value mismatch.** Port hardcodes `= 4`, matching the V4
   FlashMLA layer. If the parent defines a different value and the workspace
   reservation in `_reserve_decode_workspace` under-sizes, you get an allocator
   assert, not garbage. Re-declared locally; see basesclasses patch.

8. **`_has_flashinfer_sparse_mla()` false-negative.** If flashinfer exposes
   `run_sparse_mla` only on an *instance* (not the class), the `hasattr(class,...)`
   probe fails and `__init__` raises `RuntimeError` at layer build. Fallback:
   probe `backend="sparse-sm120"` by constructing a throwaway wrapper in a
   `try/except`. Crash, easy to spot.

9. **Selector imports `sm120` on non-sm120 hardware.** Mitigated by function-local
   import; if someone hoists it to module scope, importing `model.py` anywhere
   triggers the wrapper-availability `RuntimeError`. Keep the import local.

10. **`max_q_len` / two-call decode-prefill split.** The SM120 wrapper port does
    NOT use the decode/prefill `cum_seq_lens_q` two-call split that
    `flashinfer_sparse.py` uses for the TRTLLM-gen launcher; it uses luc's
    per-chunk `run_sparse_mla` loop. This is correct *for `run_sparse_mla`*
    (different launcher), but if `run_sparse_mla` actually needs cumulative-q
    args the call will assert. Crash.

11. **`c128a_max_compressed` formula divergence (workspace under-size).** Port's
    local `_c128a_max_compressed` must stay numerically equal to the builder's
    `self.c128a_max_compressed` (both `cdiv(cdiv(L,r),128)*128`). If one is
    edited and not the other, the C128A decode scratch is under-reserved →
    allocator/stride crash on the first C128A decode. Documented in the metadata
    patch.

## C. Things deliberately NOT tuned (per MEMORY: DSv4-Flash SM120 tuning trap)

- `--gpu-memory-utilization`, sparse-MLA env vars: untouched.
- MoE backend on sm12x: out of scope for this attention port (separate marlin /
  deep_gemm gap tracked elsewhere). This port only fixes the sparse-MLA decode
  kernel selection; it assumes the rest of the V4 stack already boots on the
  target image.
