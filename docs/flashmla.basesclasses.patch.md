# Base classes for sm120.py — what changed between 0.22.1 and v0.23.0

## TL;DR — NO base-class additions/aliases needed in `flashmla.py`

luc's `sm120.py` extended two impl/backend base classes that **do not exist in
v0.23.0**:

```python
# luc/sm120.py lines 13-16
from vllm.models.deepseek_v4.nvidia.flashmla import (
    DeepseekV4FlashMLASparseBackend,
    DeepseekV4SparseMLAAttentionImpl,
)
...
class DeepseekV4FlashInferSM120SparseBackend(DeepseekV4FlashMLASparseBackend): ...
class DeepseekV4FlashInferSM120SparseImpl(DeepseekV4SparseMLAAttentionImpl): ...
```

These came from 0.22.1's **impl-object** architecture: a backend
(`DeepseekV4FlashMLASparseBackend(FlashMLASparseBackend)`) returned an impl class
(`DeepseekV4SparseMLAAttentionImpl(SparseMLAAttentionImpl)`) whose instances were
owned by a `DeepseekV4MLAAttention` layer and reached the layer via `self.layer`.

**v0.23.0 deleted that whole layer of indirection.** DeepSeek-V4 attention is now
a model *layer* that subclasses `DeepseekV4Attention` directly and implements
`forward_mqa` itself. There is no separate impl object and no
`DeepseekV4SparseMLAAttentionImpl`. The backend's `get_impl_cls()` is a stub that
raises `NotImplementedError` (see `sparse_mla.py` lines 66-74).

### v0.23.0 equivalents (the classes the port actually builds on)

| luc 0.22.1 (`nvidia/flashmla.py`) | v0.23.0 equivalent | module |
|---|---|---|
| `DeepseekV4SparseMLAAttentionImpl` (impl base, holds `self.layer`, `PREFILL_CHUNK_SIZE`, `get_padded_num_q_heads`) | `DeepseekV4FlashMLAAttention` *layer* (and its parent `DeepseekV4Attention`) | `vllm/models/deepseek_v4/nvidia/flashmla.py` + `.../attention.py` |
| `DeepseekV4FlashMLASparseBackend(FlashMLASparseBackend)` | `DeepseekV4FlashMLABackend(AttentionBackend)` | `vllm/models/deepseek_v4/sparse_mla.py` |
| `FlashMLASparseMetadata` (carried `c128a_*`) | `DeepseekV4FlashMLAMetadata` | `vllm/models/deepseek_v4/sparse_mla.py` |

Because the port is now a *layer*, it inherits everything it needs from
`DeepseekV4Attention` (which the FlashMLA and FlashInfer V4 layers also subclass).
That parent supplies, as plain `self.<attr>`, every field luc read via
`self.layer.<attr>`:

* `compress_ratio`, `window_size`, `max_model_len`, `max_num_batched_tokens`
* `prefix`, `swa_cache_layer`, `kv_cache`
* `scale`, `attn_sink`, `topk_indices_buffer`, `padded_heads`
* `n_local_heads`, `n_local_groups`, `nope_head_dim`, `rope_head_dim`,
  `o_lora_rank`, `rotary_emb`, `wo_a`, `wo_b`, `kv_cache_torch_dtype`

(Confirmed present by grepping `v023/flashmla.py` and `v023/flashinfer_sparse.py`,
both of which read exactly these off `self`.)

`PREFILL_CHUNK_SIZE` was a `ClassVar` on the 0.22.1 impl base. v0.23.0's
`DeepseekV4FlashMLAAttention` references `self.PREFILL_CHUNK_SIZE` (e.g.
`flashmla.py:99,289`), so it is defined on the V4 layer hierarchy. The ported
`out/sm120.py` re-declares `PREFILL_CHUNK_SIZE: ClassVar[int] = 4` on itself to
be self-contained and to match the value the V4 FlashMLA layer uses; if the
parent already defines it, this is a harmless same-value override. **If a future
base-class refactor lowers `PREFILL_CHUNK_SIZE` onto `DeepseekV4Attention`,
delete the redeclaration in `out/sm120.py` to avoid divergence.**

### What to actually do

Nothing in `flashmla.py` / `sparse_mla.py`. The port file `out/sm120.py`:

1. imports `DeepseekV4Attention` (layer parent) instead of the deleted impl base;
2. imports `DeepseekV4FlashMLABackend` + `DeepseekV4FlashMLAMetadata` from
   `sparse_mla.py` instead of the deleted `DeepseekV4FlashMLASparseBackend` /
   `FlashMLASparseMetadata`;
3. copies the `_o_proj` override + `compute_fp8_einsum_recipe()` init that every
   v0.23.0 V4 nvidia layer carries (taken from `flashmla.py` /
   `flashinfer_sparse.py`) — this is new boilerplate the 0.22.1 impl did NOT have
   because o_proj lived elsewhere in that tree.

### Removed/renamed import: `has_flashinfer_sparse_mla`

luc imported `from vllm.utils.flashinfer import has_flashinfer_sparse_mla`. That
symbol is gone in v0.23.0. `out/sm120.py` replaces it with a local
`_has_flashinfer_sparse_mla()` that try/imports
`flashinfer.mla.BatchMLAPagedAttentionWrapper` and checks for the
`run_sparse_mla` attribute. This avoids depending on any vLLM-side detection
helper and probes the actual capability the kernel needs.

### Removed/renamed import: `vllm.v1.attention.backend`

luc imported `AttentionBackend` from `vllm.v1.attention.backend` (singular). In
v0.23.0 this module path **still exists** (every v023 file under test imports
`from vllm.v1.attention.backend import (...)`, e.g. `sparse_mla.py:16`,
`flashmla_sparse.py:20`, `sparse_swa.py:12`). So `vllm.v1.attention.backend` is
NOT renamed to `...backends.abstract` in this build — the KNOWN-GAPS guess does
not apply. `out/sm120.py` does not need to import `AttentionBackend` at all
(the backend it declares subclasses `DeepseekV4FlashMLABackend`, which already
pulls `AttentionBackend` in transitively), so this point is moot for the port,
but documented in case a future cleanup touches the import.
