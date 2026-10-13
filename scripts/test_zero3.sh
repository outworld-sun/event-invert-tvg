#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"
python_bin="${PYTHON_BIN:-python}"
processes="${NPROC_PER_NODE:-2}"
export CUDA_VISIBLE_DEVICES="${GPU_LIST:-0,1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export TOKENIZERS_PARALLELISM=false
export MAX_JOBS="${MAX_JOBS:-2}"
fixture_dir="$(mktemp -d "${TMPDIR:-/tmp}/vtg-zero3.XXXXXX")"
printf 'ZeRO-3 integration artifacts: %s\n' "$fixture_dir"

"$python_bin" tests/zero3_smoke.py fixture --root "$fixture_dir"
launcher=("$python_bin" -m torch.distributed.run --standalone --nproc_per_node="$processes")
"${launcher[@]}" tests/zero3_smoke.py run --root "$fixture_dir" \
  --output "$fixture_dir/resumed" --stop-after 1
"${launcher[@]}" tests/zero3_smoke.py run --root "$fixture_dir" \
  --output "$fixture_dir/resumed" --resume "$fixture_dir/resumed/checkpoint-1"
"${launcher[@]}" tests/zero3_smoke.py run --root "$fixture_dir" \
  --output "$fixture_dir/uninterrupted"
"$python_bin" tests/zero3_smoke.py compare --root "$fixture_dir"
"${launcher[@]}" tests/zero3_smoke.py run --root "$fixture_dir" \
  --output "$fixture_dir/e5" --branch E5
"${launcher[@]}" tests/zero3_smoke.py run --root "$fixture_dir" \
  --output "$fixture_dir/model_only" --model-only
"${launcher[@]}" tests/zero3_smoke.py run --root "$fixture_dir" \
  --output "$fixture_dir/coverage" --coverage-size 1
