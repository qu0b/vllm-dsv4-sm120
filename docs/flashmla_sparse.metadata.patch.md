# C128A metadata fields + builder — port status for v0.23.0

## TL;DR — NO PATCH NEEDED

The C128A metadata that `sm120.py` reads (`c128a_global_decode_topk_indices`,
`c128a_decode_topk_lens`, `c128a_prefill_topk_indices`) is **already fully
present and computed** in v0.23.0, but it lives in a *different file* than in
lucifer1004's 0.22.1 tree:

| concern | luc 0.22.1 | v0.23.0 |
|---|---|---|
| metadata dataclass | `FlashMLASparseMetadata` (`flashmla_sparse.py`) | `DeepseekV4FlashMLAMetadata` (`sparse_mla.py`) |
| metadata builder | `FlashMLASparseMetadataBuilder.build` / `_build_c128a_metadata` (`flashmla_sparse.py`) | `DeepseekV4FlashMLAMetadataBuilder.build` / `_build_c128a_metadata` (`sparse_mla.py`) |
| triton kernel | `build_c128a_topk_metadata` + `_build_c128a_topk_metadata_kernel` (`flashmla_sparse.py`) | same two functions, **already moved into** `sparse_mla.py` |

In 0.22.1 the V4 sparse impl rode on the V3.2 `FlashMLASparseMetadata`, so the
C128A fields were bolted onto that V3.2 dataclass and computed inside the V3.2
builder (gated by `self.is_deepseek_v4 and self.compress_ratio == 128`). In
v0.23.0 DeepSeek-V4 got its **own** metadata/backend module
(`vllm/models/deepseek_v4/sparse_mla.py`), and the entire C128A path was lifted
verbatim into it. The v0.23.0 `flashmla_sparse.py` (V3.2-generic) was therefore
*stripped* of the C128A fields and builder — that is correct and intended.

The SM120 port in `out/sm120.py` consequently imports
`DeepseekV4FlashMLAMetadata` (not `FlashMLASparseMetadata`) and reads the same
three attributes off it. Because `DeepseekV4FlashInferSM120SparseBackend`
subclasses `DeepseekV4FlashMLABackend`, it inherits
`DeepseekV4FlashMLAMetadataBuilder` unchanged — so the metadata it receives at
runtime already carries the populated `c128a_*` tensors. **Nothing to add.**

## Proof that the v0.23.0 computation == luc's computation

### 1. Dataclass fields

luc `flashmla_sparse.py` lines 226-231:

```python
    # Pre-computed C128A metadata (DeepseekV4 only, compress_ratio == 128).
    # Decode: global slot ids + valid-entry counts (fused from positions).
    c128a_global_decode_topk_indices: torch.Tensor | None = None
    c128a_decode_topk_lens: torch.Tensor | None = None
    # Prefill: local topk indices (used by combine_topk_swa_indices).
    c128a_prefill_topk_indices: torch.Tensor | None = None
```

v0.23.0 `sparse_mla.py` lines 124-129 — identical fields on
`DeepseekV4FlashMLAMetadata`:

```python
    # Pre-computed C128A metadata (compress_ratio == 128 only).
    # Decode: global slot ids + valid-entry counts (fused from positions).
    c128a_global_decode_topk_indices: torch.Tensor | None = None
    c128a_decode_topk_lens: torch.Tensor | None = None
    # Prefill: local topk indices (used by combine_topk_swa_indices).
    c128a_prefill_topk_indices: torch.Tensor | None = None
```

### 2. Builder allocation (`__init__`)

luc `flashmla_sparse.py` lines 349-388 (gated by `if self.compress_ratio == 128`)
allocate `c128a_max_compressed`, `c128a_global_decode_buffer`,
`c128a_decode_lens_buffer`, `c128a_prefill_buffer` with the
`_C128A_TOPK_ALIGNMENT = 128` rounding. v0.23.0 `sparse_mla.py` lines 166-193 do
the exact same allocation inside `DeepseekV4FlashMLAMetadataBuilder.__init__`
(same alignment constant at module scope, line 33).

### 3. `_build_c128a_metadata`

luc `flashmla_sparse.py` lines 639-686 == v0.23.0 `sparse_mla.py` lines 249-296,
character-for-character (same `split_decodes_and_prefills` without
`require_uniform`, same `block_size = self.kv_cache_spec.block_size //
self.compress_ratio`, same `build_c128a_topk_metadata(...)` call, same
`.view(num_decode_tokens, 1, -1)` reshape of the decode indices).

### 4. Triton kernel

luc `flashmla_sparse.py` lines 1032-1150 (`build_c128a_topk_metadata` +
`_build_c128a_topk_metadata_kernel`) == v0.23.0 `sparse_mla.py` lines 299-417,
identical.

### 5. `build()` wiring

luc threads the fields via `**c128a_fields` (lines 616-635). v0.23.0 threads them
explicitly (lines 227-247):

```python
        c128a_fields: dict[str, torch.Tensor | None] = {}
        if self.compress_ratio == 128:
            c128a_fields = self._build_c128a_metadata(cm, req_id_per_token)

        return DeepseekV4FlashMLAMetadata(
            ...
            c128a_global_decode_topk_indices=c128a_fields.get(
                "c128a_global_decode_topk_indices"
            ),
            c128a_decode_topk_lens=c128a_fields.get("c128a_decode_topk_lens"),
            c128a_prefill_topk_indices=c128a_fields.get("c128a_prefill_topk_indices"),
        )
```

Same effect; the explicit form is the v0.23.0 style.

## The ONE difference that matters for the SM120 port

luc's `_reserve_decode_workspace` (sm120.py lines 191-195) sizes the dummy-run
scratch with `_c128a_max_compressed(layer.max_model_len, layer.compress_ratio)`,
a *local helper* that re-derives the same 128-aligned width the builder stores as
`self.c128a_max_compressed`. The ported `out/sm120.py` keeps that local helper
(`_c128a_max_compressed`) verbatim — it does NOT read the builder's
`self.c128a_max_compressed`, because the layer has no handle on the builder
instance. The two formulas are identical (both compute
`cdiv(cdiv(max_model_len, compress_ratio), 128) * 128`), so the reserved
workspace matches the buffer width the kernel writes. Verify this equality holds
if either formula is ever touched (see PORT_PLAN risk list).
