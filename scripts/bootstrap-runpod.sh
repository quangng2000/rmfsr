#!/usr/bin/env bash
set -euo pipefail
# Verified official template: runpod-torch-v280.
# Image: runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404
cd "$(dirname "$0")/.."
if [[ ! -d /workspace ]]; then
  echo 'Run this bootstrap inside the Linux pod with persistent storage at /workspace.' >&2
  exit 1
fi
apt-get update
apt-get install -y --no-install-recommends libgsm1 libsndfile1 ffmpeg curl
python -m venv --system-site-packages /workspace/rmfsr-venv
source /workspace/rmfsr-venv/bin/activate
python - <<'PY'
import torch
assert torch.__version__.split('+')[0] == '2.8.0', torch.__version__
assert torch.cuda.is_available(), 'CUDA unavailable: inspect pod/driver before training'
print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name())
PY
python -m pip install -e .
python -m unittest discover -s tests -v
mkdir -p runs
