#!/usr/bin/env python3
"""Open vLLM's DeepGEMM SM120 gates, mirroring lucifer1004/dsv4-flash-sm120.

Stock v0.23.0 gates DeepGEMM on is_device_capability(90) | family(100), which
excludes SM120 (RTX PRO 6000, cap 120). On SM120 the block-scaled FP8 dense
linears fall back to CUTLASS c3x which has no SM120 dispatch -> crash. We add
`or is_device_capability_family(120)` in the same 3 places lucifer1004 did, so
SM120 uses DeepGEMM with the packed UE8M0 scale format (same as datacenter
Blackwell). The actual SM120 GEMM kernels come from the swapped-in DeepGEMM.

Exact-string replacement with assertions: if any target text isn't found
verbatim (e.g. a vLLM version drift), the build fails loudly instead of
silently shipping an unpatched image.
"""
import importlib.util
import pathlib
import sys

vllm_dir = pathlib.Path(importlib.util.find_spec("vllm").origin).parent

EDITS = [
    # (relative path, old, new)
    (
        "platforms/cuda.py",
        "        return cls.is_device_capability(90) or cls.is_device_capability_family(100)",
        "        return (\n"
        "            cls.is_device_capability(90)\n"
        "            or cls.is_device_capability_family(100)\n"
        "            or cls.is_device_capability_family(120)\n"
        "        )",
    ),
    (
        "utils/deep_gemm.py",
        "    if not current_platform.is_device_capability_family(100):\n"
        "        return False",
        "    if not (\n"
        "        current_platform.is_device_capability_family(100)\n"
        "        or current_platform.is_device_capability_family(120)\n"
        "    ):\n"
        "        return False",
    ),
    (
        "utils/deep_gemm.py",
        "        cls._oracle_cache = (  # type: ignore\n"
        "            cls.UE8M0\n"
        "            if current_platform.is_device_capability_family(100)\n"
        "            else cls.FLOAT32_CEIL_UE8M0\n"
        "        )",
        "        cls._oracle_cache = (  # type: ignore\n"
        "            cls.UE8M0\n"
        "            if (\n"
        "                current_platform.is_device_capability_family(100)\n"
        "                or current_platform.is_device_capability_family(120)\n"
        "            )\n"
        "            else cls.FLOAT32_CEIL_UE8M0\n"
        "        )",
    ),
    # Route SM120 sparse MLA to FLASHINFER_MLA_SPARSE (sparse-sm120 kernels)
    # instead of falling into the generic `else` -> FLASHMLA_SPARSE, which is
    # gated to family 90/100 and raises "Unsupported architecture for sparse
    # decode fwd" on SM120. Mirrors lucifer1004's elif major==12 branch.
    (
        "platforms/cuda.py",
        "                *sparse_backends,\n"
        "            ]\n"
        "        else:",
        "                *sparse_backends,\n"
        "            ]\n"
        "        elif device_capability.major == 12:\n"
        "            # FLASHINFER_MLA_SPARSE dispatches SM12 sparse MLA to sparse-sm120.\n"
        "            return [\n"
        "                AttentionBackendEnum.TRITON_MLA,\n"
        "                AttentionBackendEnum.FLASHINFER_MLA_SPARSE,\n"
        "            ]\n"
        "        else:",
    ),
    # Register the SM120 sparse-MLA backend (advertises capability.major==12).
    (
        "v1/attention/backends/registry.py",
        '    FLASHINFER_MLA_SPARSE_DSV4 = (\n'
        '        "vllm.models.deepseek_v4.nvidia.flashinfer_sparse."\n'
        '        "DeepseekV4FlashInferMLASparseBackend"\n'
        '    )\n',
        '    FLASHINFER_MLA_SPARSE_DSV4 = (\n'
        '        "vllm.models.deepseek_v4.nvidia.flashinfer_sparse."\n'
        '        "DeepseekV4FlashInferMLASparseBackend"\n'
        '    )\n'
        '    FLASHINFER_MLA_SPARSE_SM120_DSV4 = (\n'
        '        "vllm.models.deepseek_v4.nvidia.sm120."\n'
        '        "DeepseekV4FlashInferSM120SparseBackend"\n'
        '    )\n',
    ),
    # Route DSV4 attention to the SM120 sparse-MLA layer on RTX PRO 6000.
    (
        "models/deepseek_v4/nvidia/model.py",
        "    if (\n"
        "        vllm_config.attention_config.backend\n"
        "        == AttentionBackendEnum.FLASHINFER_MLA_SPARSE_DSV4\n"
        "    ):\n"
        "        return DeepseekV4FlashInferMLAAttention\n"
        "    return DeepseekV4FlashMLAAttention",
        "    from vllm.platforms import current_platform\n"
        "\n"
        "    backend = vllm_config.attention_config.backend\n"
        "    # SM120 (RTX PRO 6000): the only sparse-MLA decode kernel is FlashInfer's\n"
        "    # BatchMLAPagedAttentionWrapper(backend='sparse-sm120'); both stock paths\n"
        "    # (FlashMLA sparse_decode_fwd, FlashInfer TRTLLM-gen) are sm100-only.\n"
        "    if backend == AttentionBackendEnum.FLASHINFER_MLA_SPARSE_SM120_DSV4 or (\n"
        "        current_platform.is_cuda()\n"
        "        and current_platform.is_device_capability_family(120)\n"
        "    ):\n"
        "        from vllm.models.deepseek_v4.nvidia.sm120 import (\n"
        "            DeepseekV4FlashInferSM120SparseAttention,\n"
        "        )\n"
        "\n"
        "        return DeepseekV4FlashInferSM120SparseAttention\n"
        "    if backend == AttentionBackendEnum.FLASHINFER_MLA_SPARSE_DSV4:\n"
        "        return DeepseekV4FlashInferMLAAttention\n"
        "    return DeepseekV4FlashMLAAttention",
    ),
    # DSML-tool-call-leak fix, part 1/2: disable the strict-tool-calling grammar
    # for DeepSeek-V4. It was an EOS-trap -- a <｜DSML｜tool_calls> started mid-<think>
    # could not terminate until a </think> that never came -> a ~100k-token runaway
    # in the thinking channel. The model self-formats DSML reliably; leaked
    # tool-calls are recovered by part 2/2 (parser gate fix).
    (
        "tool_parsers/deepseekv4_tool_parser.py",
        "    def get_structural_tag(self, request: ChatCompletionRequest):",
        "    def adjust_request(self, request):\n"
        "        # Disable structured-output constraint for tool calls. The strict\n"
        "        # grammar created an EOS-trap: a <｜DSML｜tool_calls> started mid-<think>\n"
        "        # could not terminate until a </think> that never came -> a ~100k-token\n"
        "        # runaway. The model self-formats DSML reliably; tool-calls that leak\n"
        "        # into the reasoning span are recovered by the gate fix in\n"
        "        # DelegatingParser.parse_delta.\n"
        "        return request\n\n"
        "    def get_structural_tag(self, request: ChatCompletionRequest):",
    ),
    # DSML-tool-call-leak fix, part 2/2: recover tool-calls the model emits inside
    # <think> without first closing </think>. The reasoning gate only opens on
    # </think>, so such a block would leak as thinking text and never parse. Treat
    # the tool-call start as the end of reasoning. Gated to this abnormal case and
    # wrapped so any failure falls back to baseline behaviour.
    (
        "parser/abstract_parser.py",
        "            if self.is_reasoning_end_streaming(current_token_ids, delta_token_ids):\n"
        "                state.reasoning_ended = True\n"
        "                current_token_ids = self.extract_content_ids(delta_token_ids)\n"
        "                current_text = (\n"
        "                    delta_message.content\n"
        "                    if delta_message and delta_message.content\n"
        "                    else \"\"\n"
        "                )\n"
        "                delta_text = current_text\n"
        "                delta_token_ids = current_token_ids\n",
        "            if self.is_reasoning_end_streaming(current_token_ids, delta_token_ids):\n"
        "                state.reasoning_ended = True\n"
        "                current_token_ids = self.extract_content_ids(delta_token_ids)\n"
        "                current_text = (\n"
        "                    delta_message.content\n"
        "                    if delta_message and delta_message.content\n"
        "                    else \"\"\n"
        "                )\n"
        "                delta_text = current_text\n"
        "                delta_token_ids = current_token_ids\n"
        "            elif (\n"
        "                self._tool_parser is not None\n"
        "                and (\n"
        "                    _tc_start := getattr(\n"
        "                        self._tool_parser, \"tool_call_start_token\", None\n"
        "                    )\n"
        "                )\n"
        "                and \"｜\" in delta_text\n"
        "                and _tc_start[:7] in current_text\n"
        "            ):\n"
        "                # Recovery: the model emitted a tool-call block while still\n"
        "                # inside <think> (no </think> first) -- a DeepSeek-V4 slip. The\n"
        "                # gate only opens on </think>, so the block would leak as\n"
        "                # thinking text and never parse. Treat the tool-call start as\n"
        "                # the end of reasoning. Gated to this abnormal case (normal\n"
        "                # calls hit the </think> branch above first) and wrapped so any\n"
        "                # failure falls back to baseline behaviour.\n"
        "                try:\n"
        "                    _idx = current_text.find(_tc_start[:7])\n"
        "                    _prev_len = len(state.previous_text)\n"
        "                    _reasoning_delta = (\n"
        "                        current_text[_prev_len:_idx] if _idx > _prev_len else \"\"\n"
        "                    )\n"
        "                    _content_text = current_text[_idx:]\n"
        "                    delta_message = (\n"
        "                        DeltaMessage(reasoning=_reasoning_delta)\n"
        "                        if _reasoning_delta\n"
        "                        else None\n"
        "                    )\n"
        "                    current_text = _content_text\n"
        "                    delta_text = _content_text\n"
        "                    delta_token_ids = current_token_ids\n"
        "                    state.reasoning_ended = True\n"
        "                except Exception:\n"
        "                    pass\n",
    ),
]

for rel, old, new in EDITS:
    f = vllm_dir / rel
    text = f.read_text()
    count = text.count(old)
    if count != 1:
        sys.exit(f"PATCH FAILED: {rel}: expected exactly 1 match, found {count}\n--- looking for ---\n{old}")
    f.write_text(text.replace(old, new))
    print(f"PATCHED {rel}")

# Drop stale bytecode so the edited sources are recompiled.
for stem in ("cuda", "deep_gemm", "registry", "model", "deepseekv4_tool_parser", "abstract_parser"):
    for pyc in vllm_dir.rglob(f"__pycache__/{stem}*.pyc"):
        pyc.unlink(missing_ok=True)

print("All SM120 gates opened (deep_gemm + sparse-MLA backend + selector).")
