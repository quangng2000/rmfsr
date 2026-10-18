#!/usr/bin/env bash
set -euo pipefail
# Verified official template: runpod-torch-v280.
# Image: runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404
cd "$(dirname "$0")/.."
case "${1:-}" in
  ''|--gpu) export RMFSR_BOOTSTRAP_DEVICE=cuda ;;
  --cpu) export RMFSR_BOOTSTRAP_DEVICE=cpu ;;
  *) echo 'Usage: bootstrap-runpod.sh [--cpu|--gpu]' >&2; exit 2 ;;
esac
if [[ ! -d /workspace ]]; then
  echo 'Run this bootstrap inside the Linux pod with persistent storage at /workspace.' >&2
  exit 1
fi
apt-get update
apt-get install -y --no-install-recommends libgsm1 libsndfile1 ffmpeg curl
python -m venv --system-site-packages "${RMFSR_VENV:-/opt/rmfsr-venv}"
source "${RMFSR_VENV:-/opt/rmfsr-venv}/bin/activate"
python - <<'PY'
import os
import torch
assert torch.__version__.split('+')[0] == '2.8.0', torch.__version__
if os.environ['RMFSR_BOOTSTRAP_DEVICE'] == 'cuda':
    assert torch.cuda.is_available(), 'CUDA unavailable: inspect pod/driver before training'
    print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name())
else:
    print(torch.__version__, 'CPU data preparation; no GPU required')
PY
python -m pip install -e .
python -m unittest discover -s tests -v
mkdir -p runs
