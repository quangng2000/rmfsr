#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source /workspace/rmfsr-venv/bin/activate
training_config="${1:-configs/runpod-estimate.json}"
training_run="${2:-runs/estimate}"
mkdir -p "$training_run"
resume_args=()
if [[ -f "$training_run/latest.pt" ]]; then
  resume_args=(--resume "$training_run/latest.pt")
fi
python -m rmfsr.train --config "$training_config" --run "$training_run" "${resume_args[@]}" 2>&1 | tee -a "$training_run/console.log"
echo 'Checkpoint saved. The process has exited; the Runpod instance is STILL billed until stopped/terminated.'
