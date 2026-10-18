#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source "${RMFSR_VENV:-/opt/rmfsr-venv}/bin/activate"
benchmark_config="${1:-configs/efficient-benchmark.json}"
python - "$benchmark_config" <<'PY'
import json, sys
from rmfsr.data_smoke import require_prepared_data
cfg = json.load(open(sys.argv[1]))
if not cfg.get('benchmark_only') or not 1 <= cfg['steps'] <= 1000 or not 0 < cfg.get('max_wall_seconds', 0) <= 3600:
    raise SystemExit('Use a benchmark_only config with <=1000 updates and <=3600 seconds.')
require_prepared_data(cfg)
PY
benchmark_dir="runs/gpu-benchmark-$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$benchmark_dir"
nvidia-smi > "$benchmark_dir/nvidia-smi.txt"
python -m pip freeze > "$benchmark_dir/environment.txt"
# Synthetic off-diagonal JVP test works before the full dataset has downloaded.
python -m rmfsr.profile --config "$benchmark_config" --device cuda --repeat 20 --output "$benchmark_dir/compute.json"
# Real data test also measures CPU augmentation, codecs, disk and checkpoint costs.
python -m rmfsr.train --config "$benchmark_config" --run "$benchmark_dir/real-data" 2>&1 | tee "$benchmark_dir/train.log"
python -m rmfsr.benchmark_report --run "$benchmark_dir/real-data" --output "$benchmark_dir/throughput.json"
echo "Benchmark artifacts: $benchmark_dir"
echo 'Training has exited; this script DOES NOT stop or terminate the billed pod.'
