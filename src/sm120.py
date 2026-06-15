# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FlashInfer SM120 sparse-MLA attention layer for DeepSeek-V4.

Forward-port of lucifer1004's vLLM 0.22.1 ``nvidia/sm120.py``
(``DeepseekV4FlashInferSM120SparseImpl``) onto vLLM v0.23.0.

The 0.22.1 source was a *backend impl* object
(``DeepseekV4SparseMLAAttentionImpl`` subclass) owned by a ``DeepseekV4MLAAttention``
layer, so every attribute was reached through ``self.layer.<attr>`` and the
sparse metadata was a ``FlashMLASparseMetadata``.

In v0.23.0 DeepSeek-V4 attention is a *model layer* (``DeepseekV4Attention``
subclass; cf. ``DeepseekV4FlashMLAAttention`` and ``DeepseekV4FlashInferMLAAttention``):
``forward_mqa`` is a method on the layer, ``self`` *is* the layer, and the sparse
metadata is a ``DeepseekV4FlashMLAMetadata`` (which carries the ``c128a_*`` fields).
This port therefore:

* subclasses ``DeepseekV4Attention`` (not a backend impl base class);
* replaces every ``self.layer.<attr>`` with ``self.<attr>``;
* replaces ``FlashMLASparseMetadata`` with ``DeepseekV4FlashMLAMetadata``;
* keeps ``run_sparse_mla`` decode + prefill logic byte-for-byte identical;
* declares a backend (``DeepseekV4FlashInferSM120SparseBackend``) that subclasses
  the V4 FlashMLA backend so it inherits the ``DeepseekV4FlashMLAMetadataBuilder``
  (the only place the ``c128a_*`` metadata is computed), the 256-token blocks,
  head_size 512, and the 584-byte ``fp8_ds_mla`` cache shape;
* selects the FlashInfer ``sparse-sm120`` wrapper, which is the only sparse-MLA
  decode path that runs on SM120 (RTX PRO 6000).
"""

from typing import TYPE_CHECKING, ClassVar, cast

import torch

from vllm.forward_context import get_forward_context
from vllm.models.deepseek_v4.attention import DeepseekV4Attention
from vllm.models.deepseek_v4.common.ops import (
    compute_global_topk_indices_and_lens,
)
from vllm.models.deepseek_v4.nvidia.ops.o_proj import (
    compute_fp8_einsum_recipe,
    deep_gemm_fp8_o_proj,
)
from vllm.models.deepseek_v4.sparse_mla import (
    DeepseekV4FlashMLABackend,
    DeepseekV4FlashMLAMetadata,
)
from vllm.v1.worker.workspace import current_workspace_manager

if TYPE_CHECKING:
    from vllm.v1.attention.backends.mla.sparse_swa import DeepseekSparseSWAMetadata


def _has_flashinfer_sparse_mla() -> bool:
    """Detect FlashInfer's sparse-sm120 MLA wrapper.

    0.22.1 imported ``vllm.utils.flashinfer.has_flashinfer_sparse_mla``; that
    helper does not exist in v0.23.0, so probe the wrapper directly. flashinfer
    0.6.12 (both images) exposes ``BatchMLAPagedAttentionWrapper`` with a
    ``run_sparse_mla`` method and a ``backend="sparse-sm120"`` path.
    """
    try:
        from flashinfer.mla import BatchMLAPagedAttentionWrapper
    except Exception:
        return False
    return hasattr(BatchMLAPagedAttentionWrapper, "run_sparse_mla")


_DECODE_MAX_TOKENS = 64
_DECODE_SPLIT_TILE = 64
_C128A_TOPK_ALIGNMENT = 128


def _cdiv(x: int, y: int) -> int:
    return (int(x) + int(y) - 1) // int(y)


def _decode_num_splits(topk: int, extra_topk: int = 0) -> int:
    return _cdiv(topk, _DECODE_SPLIT_TILE) + _cdiv(extra_topk, _DECODE_SPLIT_TILE)


def _max_decode_workspace_tokens(max_num_batched_tokens: int) -> int:
    return min(int(max_num_batched_tokens), _DECODE_MAX_TOKENS)


def _c128a_max_compressed(max_model_len: int, compress_ratio: int) -> int:
    return (
        _cdiv(
            _cdiv(max_model_len, compress_ratio),
            _C128A_TOPK_ALIGNMENT,
        )
        * _C128A_TOPK_ALIGNMENT
    )


def _get_decode_scratch(
    num_tokens: int,
    num_heads: int,
    d_v: int,
    topk: int,
    extra_topk: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    num_splits = _decode_num_splits(topk, extra_topk)
    mid_out, mid_lse = current_workspace_manager().get_simultaneous(
        ((num_tokens, num_heads, num_splits, d_v), torch.bfloat16),
        ((num_tokens, num_heads, num_splits), torch.float32),
    )
    return mid_out, mid_lse


class DeepseekV4FlashInferSM120SparseBackend(DeepseekV4FlashMLABackend):
    """DeepSeek-V4 FlashInfer SM120 sparse-MLA backend.

    Subclasses the V4 FlashMLA backend to inherit ``DeepseekV4FlashMLAMetadata``,
    its builder (the C128A ``c128a_*`` metadata is computed there), 256-token
    blocks, head_size 512, and the (num_blocks, block_size, 584) fp8_ds_mla cache
    shape. Only ``supports_compute_capability`` and ``get_name`` differ.
    """

    @staticmethod
    def get_name() -> str:
        return "FLASHINFER_MLA_SPARSE_SM120_DSV4"

    @classmethod
    def supports_compute_capability(cls, capability) -> bool:
        # The only architecture with a working sparse-MLA decode path here:
        # FlashInfer's sparse-sm120 wrapper (SM120 == RTX PRO 6000).
        return capability.major == 12


class DeepseekV4FlashInferSM120SparseAttention(DeepseekV4Attention):
    """SM120 FlashInfer-wrapper-driven sparse-MLA attention layer for DeepSeek-V4.

    Layer-level equivalent of 0.22.1's ``DeepseekV4FlashInferSM120SparseImpl``.
    """

    backend_cls = DeepseekV4FlashInferSM120SparseBackend

    # Mirrors the V4 FlashMLA layer: prefill is processed in fixed-size chunks.
    # Read by the dummy-run workspace reservation below.
    PREFILL_CHUNK_SIZE: ClassVar[int] = 4

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._einsum_recipe, self._tma_aligned_scales = compute_fp8_einsum_recipe()

        if not _has_flashinfer_sparse_mla():
            raise RuntimeError(
                "DeepSeek V4 SM120 sparse MLA requires FlashInfer's "
                "sparse-sm120 MLA wrapper (flashinfer.mla."
                "BatchMLAPagedAttentionWrapper.run_sparse_mla)."
            )

        from flashinfer.mla import BatchMLAPagedAttentionWrapper

        wrapper_device = torch.device("cuda", torch.accelerator.current_device_index())
        self._sparse_mla_wrapper = BatchMLAPagedAttentionWrapper(
            torch.empty(1, dtype=torch.int8, device=wrapper_device),
            backend="sparse-sm120",
            max_num_tokens=self.max_num_batched_tokens,
            max_num_heads=self.padded_heads,
            d_v=512,
        )

    def _o_proj(self, o: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        return deep_gemm_fp8_o_proj(
            o,
            positions,
            self.rotary_emb.cos_sin_cache,
            self.wo_a,
            self.wo_b,
            n_groups=self.n_local_groups,
            heads_per_group=self.n_local_heads // self.n_local_groups,
            nope_dim=self.nope_head_dim,
            rope_dim=self.rope_head_dim,
            o_lora_rank=self.o_lora_rank,
            einsum_recipe=self._einsum_recipe,
            tma_aligned_scales=self._tma_aligned_scales,
        )

    @classmethod
    def get_padded_num_q_heads(cls, num_heads: int) -> int:
        if num_heads <= 16:
            return 16
        if num_heads <= 32:
            return 32
        if num_heads <= 64:
            return 64
        if num_heads <= 128:
            return 128
        raise ValueError(
            f"DeepseekV4 SM120 sparse MLA does not support {num_heads} heads "
            "(kernel requires h_q in {16, 32, 64, 128})."
        )

    def forward_mqa(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        positions: torch.Tensor,
        output: torch.Tensor,
    ) -> None:
        assert output.shape == q.shape, (
            f"output buffer shape {output.shape} must match q shape {q.shape}"
        )
        assert output.dtype == q.dtype, (
            f"output buffer dtype {output.dtype} must match q dtype {q.dtype}"
        )

        forward_context = get_forward_context()
        attn_metadata = forward_context.attn_metadata
        if attn_metadata is None:
            self._reserve_decode_workspace()
            output.zero_()
            return

        assert isinstance(attn_metadata, dict)
        flashmla_metadata = cast(
            DeepseekV4FlashMLAMetadata | None, attn_metadata.get(self.prefix)
        )
        swa_metadata = cast(
            "DeepseekSparseSWAMetadata | None",
            attn_metadata.get(self.swa_cache_layer.prefix),
        )
        assert swa_metadata is not None

        swa_only = self.compress_ratio <= 1
        # SWA-only layers (compress_ratio <= 1) don't have their own KV cache
        # allocation; self.kv_cache may be empty after profiling cleanup.
        self_kv_cache = self.kv_cache if not swa_only else None
        swa_kv_cache = self.swa_cache_layer.kv_cache

        num_decodes = swa_metadata.num_decodes
        num_prefills = swa_metadata.num_prefills
        num_decode_tokens = swa_metadata.num_decode_tokens

        if num_prefills > 0:
            self._forward_prefill(
                q=q[num_decode_tokens:],
                compressed_k_cache=self_kv_cache,
                swa_k_cache=swa_kv_cache,
                output=output[num_decode_tokens:],
                attn_metadata=flashmla_metadata,
                swa_metadata=swa_metadata,
            )
        if num_decodes > 0:
            self._forward_decode(
                q=q[:num_decode_tokens],
                kv_cache=self_kv_cache,
                swa_metadata=swa_metadata,
                attn_metadata=flashmla_metadata,
                swa_only=swa_only,
                output=output[:num_decode_tokens],
            )

    def _reserve_decode_workspace(self) -> None:
        if self.compress_ratio <= 1:
            extra_topk = 0
        elif self.compress_ratio == 4:
            assert self.topk_indices_buffer is not None
            extra_topk = self.topk_indices_buffer.shape[-1]
        elif self.compress_ratio == 128:
            extra_topk = _c128a_max_compressed(
                self.max_model_len,
                self.compress_ratio,
            )
        else:
            raise ValueError(
                f"Unsupported compress_ratio={self.compress_ratio}; "
                "expected 1, 4, or 128."
            )
        _get_decode_scratch(
            _max_decode_workspace_tokens(self.max_num_batched_tokens),
            self.padded_heads,
            512,
            self.window_size,
            extra_topk,
        )

    def _forward_decode(
        self,
        q: torch.Tensor,
        kv_cache: torch.Tensor | None,  # only used when compress_ratio > 1
        swa_metadata: "DeepseekSparseSWAMetadata",
        attn_metadata: DeepseekV4FlashMLAMetadata | None,
        swa_only: bool,
        output: torch.Tensor,
    ) -> None:
        num_decodes = swa_metadata.num_decodes
        num_decode_tokens = swa_metadata.num_decode_tokens

        extra_sparse_indices = None
        extra_sparse_lengths = None
        if not swa_only:
            if attn_metadata is None:
                raise RuntimeError(
                    "Sparse MLA metadata is required for compressed layers."
                )
            if swa_metadata.is_valid_token is None:
                raise RuntimeError(
                    "SWA validity metadata is required for compressed layers."
                )
            is_valid = swa_metadata.is_valid_token[:num_decode_tokens]
            if self.compress_ratio == 4:
                # C4A: local indices differ per layer (filled by Indexer).
                if self.topk_indices_buffer is None:
                    raise RuntimeError(
                        "C4A decode requires top-k indices from the indexer."
                    )
                block_size = attn_metadata.block_size // self.compress_ratio
                global_indices, extra_sparse_lengths = (
                    compute_global_topk_indices_and_lens(
                        self.topk_indices_buffer[:num_decode_tokens],
                        swa_metadata.token_to_req_indices,
                        attn_metadata.block_table[:num_decodes],
                        block_size,
                        is_valid,
                    )
                )
                extra_sparse_indices = global_indices.view(num_decode_tokens, 1, -1)
            else:
                # C128A: pre-computed during metadata build.
                extra_sparse_indices = attn_metadata.c128a_global_decode_topk_indices
                extra_sparse_lengths = attn_metadata.c128a_decode_topk_lens

        swa_indices = swa_metadata.decode_swa_indices
        swa_lens = swa_metadata.decode_swa_lens
        assert swa_indices is not None
        assert swa_lens is not None
        extra_topk = (
            extra_sparse_indices.shape[-1] if extra_sparse_indices is not None else 0
        )
        mid_out, mid_lse = _get_decode_scratch(
            num_decode_tokens,
            q.shape[1],
            output.shape[-1],
            swa_indices.shape[-1],
            extra_topk,
        )

        # The wrapper attends through generated sparse indices only.
        q = q.unsqueeze(1)
        swa_cache = self.swa_cache_layer.kv_cache.unsqueeze(-2)
        if kv_cache is not None:
            kv_cache = kv_cache.unsqueeze(-2)

        assert self._sparse_mla_wrapper is not None, (
            "DeepseekV4FlashInferSM120SparseAttention requires FlashInfer's "
            "sparse-sm120 MLA wrapper to be available."
        )
        self._sparse_mla_wrapper.run_sparse_mla(
            q=q,
            kv_cache=swa_cache,
            sparse_indices=swa_indices,
            out=output,
            sm_scale=self.scale,
            sparse_lengths=swa_lens,
            sinks=self.attn_sink,
            extra_kv_cache=kv_cache if not swa_only else None,
            extra_sparse_indices=extra_sparse_indices,
            extra_sparse_lengths=extra_sparse_lengths,
            mid_out=mid_out,
            mid_lse=mid_lse,
        )

    def _forward_prefill(
        self,
        q: torch.Tensor,
        compressed_k_cache: torch.Tensor | None,
        swa_k_cache: torch.Tensor,
        output: torch.Tensor,
        attn_metadata: DeepseekV4FlashMLAMetadata | None,
        swa_metadata: "DeepseekSparseSWAMetadata",
    ) -> None:
        # `_dummy_run` passes synthetic non-None attn_metadata for swa-only
        # layers during cudagraph capture, so check compress_ratio directly.
        swa_only = self.compress_ratio <= 1

        num_prefills = swa_metadata.num_prefills
        num_decodes = swa_metadata.num_decodes
        num_decode_tokens = swa_metadata.num_decode_tokens
        num_prefill_tokens = swa_metadata.num_prefill_tokens

        # Derive prefill-local token offsets from the full query_start_loc_cpu.
        query_start_loc_cpu = swa_metadata.query_start_loc_cpu
        assert query_start_loc_cpu is not None
        prefill_token_base = query_start_loc_cpu[num_decodes]

        local_topk_indices: torch.Tensor | None
        if swa_only:
            local_topk_indices = None
        elif self.compress_ratio == 4:
            if self.topk_indices_buffer is None:
                raise RuntimeError(
                    "C4A prefill requires top-k indices from the indexer."
                )
            local_topk_indices = self.topk_indices_buffer[
                num_decode_tokens : num_decode_tokens + num_prefill_tokens
            ]
        else:
            # C128A: pre-computed during metadata build.
            if attn_metadata is None:
                raise RuntimeError("C128A prefill metadata is missing.")
            local_topk_indices = attn_metadata.c128a_prefill_topk_indices

        extra_sparse_indices: torch.Tensor | None = None
        extra_sparse_lengths: torch.Tensor | None = None
        if local_topk_indices is not None:
            if attn_metadata is None:
                raise RuntimeError("C4A prefill metadata is missing.")
            if swa_metadata.token_to_req_indices is None:
                raise RuntimeError("C4A prefill request mapping is missing.")
            if swa_metadata.is_valid_token is None:
                raise RuntimeError("C4A prefill validity metadata is missing.")
            prefill_token_slice = slice(
                num_decode_tokens, num_decode_tokens + num_prefill_tokens
            )
            # FlashInfer prefill expects physical KV slots; keep padding rows
            # masked through the metadata validity mask.
            block_size = attn_metadata.block_size // self.compress_ratio
            extra_sparse_indices, extra_sparse_lengths = (
                compute_global_topk_indices_and_lens(
                    local_topk_indices,
                    swa_metadata.token_to_req_indices[prefill_token_slice],
                    attn_metadata.block_table,
                    block_size,
                    swa_metadata.is_valid_token[prefill_token_slice],
                )
            )

        assert swa_metadata.prefill_swa_indices is not None
        assert swa_metadata.prefill_swa_lens is not None
        assert self._sparse_mla_wrapper is not None

        swa_kv_paged = swa_k_cache.unsqueeze(-2)
        if swa_only:
            extra_kv_paged = None
        else:
            if compressed_k_cache is None:
                raise RuntimeError(
                    "Compressed sparse MLA layers require their compressed KV cache."
                )
            extra_kv_paged = compressed_k_cache.unsqueeze(-2)

        num_chunks = (
            num_prefills + self.PREFILL_CHUNK_SIZE - 1
        ) // self.PREFILL_CHUNK_SIZE
        for chunk_idx in range(num_chunks):
            chunk_start = chunk_idx * self.PREFILL_CHUNK_SIZE
            chunk_end = min(chunk_start + self.PREFILL_CHUNK_SIZE, num_prefills)
            query_start = (
                query_start_loc_cpu[num_decodes + chunk_start] - prefill_token_base
            )
            query_end = (
                query_start_loc_cpu[num_decodes + chunk_end] - prefill_token_base
            )

            extra_sparse_indices_chunk = (
                extra_sparse_indices[query_start:query_end]
                if extra_sparse_indices is not None
                else None
            )
            extra_sparse_lengths_chunk = (
                extra_sparse_lengths[query_start:query_end]
                if extra_sparse_lengths is not None
                else None
            )
            chunk_tokens = query_end - query_start

            mid_out = None
            mid_lse = None
            if chunk_tokens <= _DECODE_MAX_TOKENS:
                extra_topk = (
                    extra_sparse_indices_chunk.shape[-1]
                    if extra_sparse_indices_chunk is not None
                    else 0
                )
                mid_out, mid_lse = _get_decode_scratch(
                    chunk_tokens,
                    q.shape[1],
                    output.shape[-1],
                    swa_metadata.prefill_swa_indices.shape[-1],
                    extra_topk,
                )

            self._sparse_mla_wrapper.run_sparse_mla(
                q=q[query_start:query_end],
                kv_cache=swa_kv_paged,
                sparse_indices=swa_metadata.prefill_swa_indices[query_start:query_end],
                out=output[query_start:query_end],
                sm_scale=self.scale,
                sparse_lengths=swa_metadata.prefill_swa_lens[query_start:query_end],
                sinks=self.attn_sink,
                extra_kv_cache=extra_kv_paged,
                extra_sparse_indices=extra_sparse_indices_chunk,
                extra_sparse_lengths=extra_sparse_lengths_chunk,
                mid_out=mid_out,
                mid_lse=mid_lse,
            )
