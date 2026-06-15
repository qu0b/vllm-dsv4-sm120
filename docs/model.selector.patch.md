# Selector + registry wiring for the SM120 sparse-MLA impl

Two edits: (1) register the new backend in `registry.py`, (2) route
`_select_dsv4_attn_cls` in `model.py` to the new layer on SM120.

The port file is assumed installed as
`vllm/models/deepseek_v4/nvidia/sm120.py` exposing
`DeepseekV4FlashInferSM120SparseAttention` and
`DeepseekV4FlashInferSM120SparseBackend`.

---

## Edit 1 — `vllm/v1/attention/backends/registry.py`

Add a new enum member next to the existing DSV4 backends (after
`FLASHINFER_MLA_SPARSE_DSV4`, anchor lines 83-86):

```python
    FLASHINFER_MLA_SPARSE_DSV4 = (
        "vllm.models.deepseek_v4.nvidia.flashinfer_sparse."
        "DeepseekV4FlashInferMLASparseBackend"
    )
    # SM120 (RTX PRO 6000): the only sparse-MLA decode path on sm12x, via
    # FlashInfer's BatchMLAPagedAttentionWrapper(backend="sparse-sm120").
    FLASHINFER_MLA_SPARSE_SM120_DSV4 = (
        "vllm.models.deepseek_v4.nvidia.sm120."
        "DeepseekV4FlashInferSM120SparseBackend"
    )
```

(Adding it as a *distinct* enum member keeps `get_supported_kernel_block_sizes`,
`get_kv_cache_shape`, etc. resolving through the SM120 backend's MRO — it
subclasses `DeepseekV4FlashMLABackend`, so those all inherit correctly — while
letting `supports_compute_capability` return `capability.major == 12` only.)

---

## Edit 2 — `vllm/models/deepseek_v4/model.py`, `_select_dsv4_attn_cls`

### Decision (per task brief)

Trigger on **device capability major == 12** while keeping the existing
`--attention-backend FLASHINFER_MLA_SPARSE_DSV4` flag as the FlashInfer opt-in.
Rationale:

* On sm12x, both stock V4 sparse decode paths crash ("Unsupported
  architecture"): FlashMLA (`_flashmla_C.sparse_decode_fwd`) and the FlashInfer
  TRTLLM-gen launcher (`trtllm_batch_decode_sparse_mla_dsv4` →
  `TllmGenFmhaRunner`). So on sm12x there is exactly one correct choice and the
  selector should force it regardless of which DSV4 backend the user named.
* Auto-detect (rather than requiring a 4th `--attention-backend` value) means
  existing run scripts that pass `FLASHMLA_SPARSE_DSV4` (the default) or
  `FLASHINFER_MLA_SPARSE_DSV4` both land on the working impl when run on an
  RTX PRO 6000.

### Current code (model.py lines 716-727)

```python
def _select_dsv4_attn_cls(vllm_config: VllmConfig) -> type[DeepseekV4Attention]:
    """Pick the CUDA sparse-MLA attention class for the configured backend.

    An explicit ``--attention-backend FLASHINFER_MLA_SPARSE_DSV4`` selects the
    FlashInfer TRTLLM-gen path; otherwise the FlashMLA path is used.
    """
    if (
        vllm_config.attention_config.backend
        == AttentionBackendEnum.FLASHINFER_MLA_SPARSE_DSV4
    ):
        return DeepseekV4FlashInferMLAAttention
    return DeepseekV4FlashMLAAttention
```

### Replacement

```python
def _select_dsv4_attn_cls(vllm_config: VllmConfig) -> type[DeepseekV4Attention]:
    """Pick the CUDA sparse-MLA attention class for the configured backend.

    On SM120 (RTX PRO 6000, capability major == 12) neither stock sparse-MLA
    decode path has a kernel (FlashMLA ``sparse_decode_fwd`` and the FlashInfer
    TRTLLM-gen launcher are both sm100-only). The only working path is
    FlashInfer's ``BatchMLAPagedAttentionWrapper(backend="sparse-sm120")``, so
    force the SM120 impl there regardless of the requested DSV4 backend.

    Otherwise: an explicit ``--attention-backend FLASHINFER_MLA_SPARSE_DSV4``
    selects the FlashInfer TRTLLM-gen path; the default is FlashMLA.
    """
    from vllm.platforms import current_platform

    backend = vllm_config.attention_config.backend

    # Explicit override always wins, even on sm12x (lets a user force a
    # non-sm120 path for A/B testing on capable hardware).
    if backend == AttentionBackendEnum.FLASHINFER_MLA_SPARSE_SM120_DSV4:
        from vllm.models.deepseek_v4.nvidia.sm120 import (
            DeepseekV4FlashInferSM120SparseAttention,
        )

        return DeepseekV4FlashInferSM120SparseAttention

    # Auto-route on SM120 unless the user pinned a specific backend above.
    if (
        current_platform.is_cuda()
        and current_platform.is_device_capability_family(120)
        and backend
        not in (
            AttentionBackendEnum.FLASHMLA_SPARSE_DSV4,
            AttentionBackendEnum.FLASHINFER_MLA_SPARSE_DSV4,
        )
    ):
        from vllm.models.deepseek_v4.nvidia.sm120 import (
            DeepseekV4FlashInferSM120SparseAttention,
        )

        return DeepseekV4FlashInferSM120SparseAttention

    if backend == AttentionBackendEnum.FLASHINFER_MLA_SPARSE_DSV4:
        return DeepseekV4FlashInferMLAAttention
    return DeepseekV4FlashMLAAttention
```

Notes:

* The `sm120` import is local (inside the function) so importing `model.py` on a
  non-sm120 / no-flashinfer box never triggers the wrapper-availability
  `RuntimeError` in `DeepseekV4FlashInferSM120SparseAttention.__init__`.
* `is_device_capability_family(120)` is the same helper `sparse_swa.py` already
  uses (line 455) to *skip* FlashMLA tile-scheduler planning on sm12x — that skip
  is exactly what makes the SWA metadata safe to feed to the FlashInfer wrapper
  (no FlashMLA `tile_sched_*` is built, and the SM120 impl never reads those
  fields). So the two changes are consistent.
* The second `if` deliberately lets the user *opt out* of auto-routing by naming
  `FLASHMLA_SPARSE_DSV4` or `FLASHINFER_MLA_SPARSE_DSV4` explicitly — but those
  will then crash at kernel launch on sm12x. That is acceptable (explicit user
  choice) and mirrors how vLLM treats unsupported explicit backends elsewhere.
  If you prefer SM120 to be unconditionally forced, drop the
  `backend not in (...)` guard.

### Alternative trigger (simpler, brief's fallback)

If you do not want a new enum member, reuse the existing
`FLASHINFER_MLA_SPARSE_DSV4` flag and branch inside the existing FlashInfer arm:

```python
    if backend == AttentionBackendEnum.FLASHINFER_MLA_SPARSE_DSV4:
        from vllm.platforms import current_platform
        if current_platform.is_device_capability_family(120):
            from vllm.models.deepseek_v4.nvidia.sm120 import (
                DeepseekV4FlashInferSM120SparseAttention,
            )
            return DeepseekV4FlashInferSM120SparseAttention
        return DeepseekV4FlashInferMLAAttention
```

This needs no `registry.py` edit, but then the *backend class* used for
KV-cache-shape / capability queries is still
`DeepseekV4FlashInferMLASparseBackend` (whose `supports_compute_capability`
returns `major in [9, 10]`, i.e. it would *reject* sm12x at backend-selection
time). The standalone-enum approach (Edit 1) is therefore preferred: its backend
advertises `major == 12`, so backend selection accepts sm12x cleanly.
