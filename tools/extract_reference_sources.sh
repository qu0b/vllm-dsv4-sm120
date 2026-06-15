#!/bin/bash
# Pull the DSV4 sparse-MLA source from both images into port/ for diffing when
# re-porting to a new vLLM version. port/ is gitignored (vendored third-party).
#
#   port/luc/   = lucifer1004's working SM120 source (the reference impl)
#   port/v023/  = the stock upstream base's source (what we patch onto)
#
# Override the images via env: BASE_IMAGE, LUC_IMAGE.
set -euo pipefail
cd "$(dirname "$0")/.."

BASE_IMAGE="${BASE_IMAGE:-vllm/vllm-openai:v0.23.0}"
LUC_IMAGE="${LUC_IMAGE:-lucifer1004/dsv4-flash-sm120:20260604}"
LUC_SP=/opt/env/lib/python3.12/site-packages/vllm
BASE_SP=/usr/local/lib/python3.12/dist-packages/vllm

FILES=(
  models/deepseek_v4/nvidia/sm120.py
  models/deepseek_v4/nvidia/flashmla.py
  models/deepseek_v4/nvidia/flashinfer_sparse.py
  models/deepseek_v4/nvidia/model.py
  models/deepseek_v4/sparse_mla.py
  v1/attention/backends/mla/flashmla_sparse.py
  v1/attention/backends/mla/sparse_swa.py
  v1/attention/backends/registry.py
)

dump() { # image  src_root  out_dir
  local img="$1" root="$2" out="$3"
  mkdir -p "$out"
  local cid; cid=$(docker create "$img")
  for f in "${FILES[@]}"; do
    docker cp "$cid:$root/$f" "$out/$(echo "$f" | tr '/' '_')" 2>/dev/null \
      && echo "  $img: $f" || echo "  $img MISS: $f"
  done
  docker rm "$cid" >/dev/null
}

echo "== lucifer1004 reference ($LUC_IMAGE) =="; dump "$LUC_IMAGE" "$LUC_SP" port/luc
echo "== base ($BASE_IMAGE) =="; dump "$BASE_IMAGE" "$BASE_SP" port/v023
echo "Done. Diff e.g.: diff port/v023/v1_attention_backends_mla_sparse_swa.py src/sparse_swa.py"
