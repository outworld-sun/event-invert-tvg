#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"
python_bin="${PYTHON_BIN:-python3}"
"$python_bin" -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install torch==2.6.0 torchvision==0.21.0 \
  --index-url "${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu124}"
.venv/bin/python -m pip install -e '.[dev]'
.venv/bin/python -m spacy download en_core_web_sm
if [[ "${INSTALL_FLASH_ATTENTION:-0}" == "1" ]]; then
  .venv/bin/python -m pip install packaging ninja wheel
  .venv/bin/python -m pip install flash-attn==2.7.4.post1 --no-build-isolation
fi
printf '%s\n' 'Installed. Activate with: source .venv/bin/activate'
