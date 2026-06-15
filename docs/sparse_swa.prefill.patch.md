# Prefill SWA-index metadata — port for v0.23.0 `sparse_swa.py`

## Problem

`sm120.py:_forward_prefill` reads `swa_metadata.prefill_swa_indices` /
`swa_metadata.prefill_swa_lens` (lines ~362-363, ~420, ~423). v0.23.0's
`DeepseekSparseSWAMetadata` only carries the **decode** SWA indices
(`decode_swa_indices` / `decode_swa_lens`); prefill in stock v0.23.0 goes through
the FlashMLA prefill kernel and only needs `prefill_seq_lens` /
`prefill_gather_lens`. So the first request that hits the prefill path crashes
with `AttributeError: 'DeepseekSparseSWAMetadata' object has no attribute
'prefill_swa_indices'`.

This patch adds the prefill SWA-index metadata so the lucifer1004 `run_sparse_mla`
prefill path works.

## Kernel decision — REUSE v0.23.0's local triton kernel (NO flashinfer)

I reuse v0.23.0's own `_compute_swa_indices_and_lens_kernel` — the same kernel the
decode path already calls — rather than `flashinfer.swa_indices.compute_swa_indices_and_lens`.

Why it works without a `token_offset` parameter: the kernel uses
`token_idx = tl.program_id(0)` to index **both** input reads
(`is_valid_token_ptr + token_idx`, `token_to_req_indices_ptr + token_idx`) and
output writes (`swa_lens_ptr + token_idx`, `swa_indices_ptr + token_idx*stride`).
lucifer1004's flashinfer helper adds `token_offset` precisely so output starts at
index 0 while inputs are read at absolute prefill positions
`[num_decode_tokens : num_decode_tokens + num_prefill_tokens]`.

We get the same effect by passing **sliced views** of the per-token inputs:

- `token_to_req_indices_ptr` ← `token_to_req_indices[num_decode_tokens:]`
- `is_valid_token_ptr`        ← `is_valid_token[num_decode_tokens:]`

and **un-sliced** per-request inputs (`query_start_loc`, `seq_lens`, `block_table`)
because the kernel indexes those by `req_idx` (an **absolute** request index taken
from `token_to_req_indices`, which is built from `torch.arange(num_reqs)` over the
whole batch — line 297). Launching grid `(num_prefill_tokens,)` then makes
`token_idx` run `[0:num_prefill_tokens]`, reading the offset input views and
writing output at `[0:num_prefill_tokens]` — exactly the layout
`_forward_prefill` indexes with prefill-local offsets
(`prefill_swa_indices[query_start:query_end]`, where `query_start`/`query_end` are
relative to `prefill_token_base`).

Both sliced tensors are contiguous 1-D slices, so the kernel's pointer arithmetic
stays correct. This keeps the image free of any hard `flashinfer.swa_indices`
import in the metadata builder (decode already uses the local kernel; prefill now
matches).

## Buffer sizing

Persistent prefill buffers are sized identically to `decode_swa_indices` /
`decode_swa_lens`: `max_tokens = scheduler_config.max_num_batched_tokens`, shape
`[max_tokens, 1, window_size]` (int32) and `[max_tokens]` (int32). This matches
lucifer1004 (`port/luc/sparse_swa.py` lines ~295-306) and is safe because
`num_prefill_tokens <= max_num_batched_tokens`. The returned slice is
`[:num_prefill_tokens]`, giving shape `[num_prefill_tokens, 1, window_size]` — the
3-D `[tokens, 1, window]` layout `run_sparse_mla` expects for `sparse_indices`
(same as decode's `decode_swa_indices`).

## Edits (append these to `EDITS` in `apply_sm120_patches.py`, rel path `v1/attention/backends/mla/sparse_swa.py`)

Each `old` block below is **verbatim** v0.23.0 `sparse_swa.py` and is unique in the
file.

### Edit 1 — dataclass fields

OLD:
```python
    decode_swa_indices: torch.Tensor | None = None  # [num_decode_tokens, window_size]
    decode_swa_lens: torch.Tensor | None = None  # [num_decode_tokens]

    # Number of decode/prefill requests/tokens (batch is reordered: decodes first)
```

NEW:
```python
    decode_swa_indices: torch.Tensor | None = None  # [num_decode_tokens, window_size]
    decode_swa_lens: torch.Tensor | None = None  # [num_decode_tokens]
    # Paged-coordinate prefill SWA indices/lens (SM120 run_sparse_mla prefill).
    prefill_swa_indices: torch.Tensor | None = (
        None  # [num_prefill_tokens, 1, window_size]
    )
    prefill_swa_lens: torch.Tensor | None = None  # [num_prefill_tokens]

    # Number of decode/prefill requests/tokens (batch is reordered: decodes first)
```

### Edit 2 — builder `__init__` buffer allocation

OLD:
```python
        self.decode_swa_lens = torch.zeros(
            max_tokens,
            dtype=torch.int32,
            device=self.device,
        )
        self.is_valid_token = torch.zeros(
```

NEW:
```python
        self.decode_swa_lens = torch.zeros(
            max_tokens,
            dtype=torch.int32,
            device=self.device,
        )
        # Prefill SWA indices/lens for the SM120 run_sparse_mla prefill path.
        # Allocated unconditionally; sized like the decode buffers since
        # num_prefill_tokens <= max_num_batched_tokens.
        self.prefill_swa_indices = torch.zeros(
            max_tokens,
            1,
            self.window_size,
            dtype=torch.int32,
            device=self.device,
        )
        self.prefill_swa_lens = torch.zeros(
            max_tokens,
            dtype=torch.int32,
            device=self.device,
        )
        self.is_valid_token = torch.zeros(
```

### Edit 3 — `build()` prefill computation (after the decode `if`-block)

OLD:
```python
        if num_decode_tokens > 0:
            self.decode_swa_lens[num_decode_tokens:] = 0
            _compute_swa_indices_and_lens_kernel[(num_decode_tokens,)](
                self.decode_swa_indices,
                self.decode_swa_indices.stride(0),
                self.decode_swa_lens,
                self.window_size,
                query_start_loc,
                seq_lens,
                token_to_req_indices,
                is_valid_token,
                block_table,
                block_table.stride(0),
                self.block_size,
                TRITON_BLOCK_SIZE=1024,
            )

        # Pre-compute DeepseekV4 prefill metadata shared across all attention layers.
```

NEW:
```python
        if num_decode_tokens > 0:
            self.decode_swa_lens[num_decode_tokens:] = 0
            _compute_swa_indices_and_lens_kernel[(num_decode_tokens,)](
                self.decode_swa_indices,
                self.decode_swa_indices.stride(0),
                self.decode_swa_lens,
                self.window_size,
                query_start_loc,
                seq_lens,
                token_to_req_indices,
                is_valid_token,
                block_table,
                block_table.stride(0),
                self.block_size,
                TRITON_BLOCK_SIZE=1024,
            )

        # Prefill SWA indices for the SM120 run_sparse_mla prefill path. Reuse the
        # same local kernel as decode: feed it the prefill slice of the per-token
        # inputs (token_to_req_indices / is_valid_token) so output is written at
        # [0:num_prefill_tokens] while inputs are read at absolute prefill
        # positions [num_decode_tokens:]. Per-request inputs (query_start_loc,
        # seq_lens, block_table) stay un-sliced: the kernel indexes them by the
        # absolute req_idx carried in token_to_req_indices.
        if num_prefill_tokens > 0:
            self.prefill_swa_lens[num_prefill_tokens:] = 0
            _compute_swa_indices_and_lens_kernel[(num_prefill_tokens,)](
                self.prefill_swa_indices,
                self.prefill_swa_indices.stride(0),
                self.prefill_swa_lens,
                self.window_size,
                query_start_loc,
                seq_lens,
                token_to_req_indices[num_decode_tokens:],
                is_valid_token[num_decode_tokens:],
                block_table,
                block_table.stride(0),
                self.block_size,
                TRITON_BLOCK_SIZE=1024,
            )

        # Pre-compute DeepseekV4 prefill metadata shared across all attention layers.
```

### Edit 4 — `build()` return: thread the prefill fields onto the metadata object

OLD:
```python
            decode_swa_indices=self.decode_swa_indices[:num_decode_tokens],
            decode_swa_lens=self.decode_swa_lens[:num_decode_tokens],
            block_size=self.block_size,
```

NEW:
```python
            decode_swa_indices=self.decode_swa_indices[:num_decode_tokens],
            decode_swa_lens=self.decode_swa_lens[:num_decode_tokens],
            prefill_swa_indices=(
                self.prefill_swa_indices[:num_prefill_tokens]
                if num_prefill_tokens > 0
                else None
            ),
            prefill_swa_lens=(
                self.prefill_swa_lens[:num_prefill_tokens]
                if num_prefill_tokens > 0
                else None
            ),
            block_size=self.block_size,
```

## Apply-script note

Add `"sparse_swa"` to the bytecode-drop stem list in `apply_sm120_patches.py`:

```python
for stem in ("cuda", "deep_gemm", "registry", "model", "sparse_swa"):
```

so the edited source is recompiled. (The 4 EDITS go in the `EDITS` list with rel
path `"v1/attention/backends/mla/sparse_swa.py"`; the script's existing
`count != 1` assertion guards each old-string — all four are unique in the file.)

## Correctness risks / notes

1. **window_size source** — `self.window_size = hf_config.sliding_window` (builder
   `__init__`), identical to the decode path. The buffer's last dim is `window_size`;
   `swa_len <= window_size` always (kernel computes `end_pos - start_pos` with
   `start_pos = max(pos - window_size + 1, 0)`). No risk.

2. **`block_size // compress_ratio` for prefill** — NOT applicable here. This kernel
   writes the **SWA** indices in the SWA cache's own paged coordinates
   (`self.block_size`, the 64-token SWA page). The `block_size // compress_ratio`
   adjustment is only for the **C128A/C4A** compressed `extra_sparse_indices`, which
   `sm120.py:_forward_prefill` computes separately via
   `compute_global_topk_indices_and_lens` (lines ~351-360). So we correctly pass
   `self.block_size` unchanged, matching the decode call.

3. **Padding-row masking via `is_valid_token`** — preserved. The kernel early-exits
   (`swa_len = 0`) for invalid tokens. By slicing `is_valid_token[num_decode_tokens:]`
   we feed the prefill validity mask aligned with the prefill output rows, so padding
   rows get `swa_len = 0` and are skipped by `run_sparse_mla`. The
   `self.prefill_swa_lens[num_prefill_tokens:] = 0` line clears stale tail lengths
   (mirrors the decode `decode_swa_lens[num_decode_tokens:] = 0`); the slice returned
   to the metadata is `[:num_prefill_tokens]` so the tail is never read anyway, but
   keeping it matches the decode-side hygiene.

4. **Slice-view pointer arithmetic** — `token_to_req_indices` and `is_valid_token`
   are 1-D contiguous; `[num_decode_tokens:]` is a contiguous view with the right base
   pointer, so the kernel's `+ token_idx` reads land on the intended absolute element.
   `req_idx` read from the sliced `token_to_req_indices` is still the **absolute**
   request index (values were filled from `torch.arange(num_reqs)` over the full
   batch), so the un-sliced `query_start_loc` / `seq_lens` / `block_table` lookups are
   correct.

5. **Cost** — one extra Triton launch per build when `num_prefill_tokens > 0`
   (grid `num_prefill_tokens`), same shape as the decode launch. Pure prefill steps
   previously did zero SWA-index work here; this adds it unconditionally. There is no
   SM120 gate because the buffers/return are cheap and harmless on other platforms
   (the fields are simply unread there), and gating on capability-family(120) inside
   the shared builder would diverge it from stock for no benefit. If a pure-decode
   path is ever the hot loop, note that `num_prefill_tokens == 0` skips the launch
   entirely, so there is no decode-path regression.
