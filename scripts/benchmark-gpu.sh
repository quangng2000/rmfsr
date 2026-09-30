#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source /workspace/rmfsr-venv/bin/activate
benchmark_config="${1:-configs/runpod-estimate-benchmark.json}"
benchmark_dir="runs/gpu-benchmark-$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$benchmark_dir"
nvidia-smi > "$benchmark_dir/nvidia-smi.txt"
python -m pip freeze > "$benchmark_dir/environment.txt"
# Synthetic off-diagonal JVP test works before the full dataset has downloaded.
python -m rmfsr.profile --device cuda --batch 4 --seconds 4 --repeat 20 --output "$benchmark_dir/compute.json"
# Real data test also measures CPU augmentation, codecs, disk and checkpoint costs.
python -m rmfsr.train --config "$benchmark_config" --run "$benchmark_dir/real-data" 2>&1 | tee "$benchmark_dir/train.log"
echo "Benchmark artifacts: $benchmark_dir"
echo 'Training has exited; this script DOES NOT stop or terminate the billed pod.'
