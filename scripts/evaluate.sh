#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
python_bin="${PYTHON_BIN:-python}"
exec "$python_bin" -m torch.distributed.run --standalone \
  --nproc_per_node="${NPROC_PER_NODE:-8}" --module vtg.cli.evaluate "$@"
