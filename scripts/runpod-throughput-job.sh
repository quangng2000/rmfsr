#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source "${RMFSR_VENV:-/opt/rmfsr-venv}/bin/activate"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1
python - <<'PY'
import json
from rmfsr.data_smoke import require_prepared_data
require_prepared_data(json.load(open('configs/efficient-benchmark.json')))
PY
mkdir -p runs/qualification
job_dir=runs/qualification
children=()
finish() {
  result=$?
  trap - EXIT
  for child in "${children[@]}"; do kill "$child" 2>/dev/null || true; done
  if [[ "$result" -eq 0 ]]; then
    echo complete > "$job_dir/status"
  else
    echo "failed:$result" > "$job_dir/status"
  fi
  exit "$result"
}
trap finish EXIT
trap 'exit 143' TERM
echo profiling_prepared_data > "$job_dir/status"
nvidia-smi > "$job_dir/nvidia-smi.txt"
python -m pip freeze > "$job_dir/environment.txt"
python -m rmfsr.profile --config configs/efficient-benchmark.json --device cuda --repeat 20 --output "$job_dir/compute.json" > "$job_dir/compute.log" 2>&1
echo training_benchmark > "$job_dir/status"
python -m rmfsr.train --config configs/efficient-benchmark.json --run "$job_dir/real-data" > "$job_dir/train.log" 2>&1
python -m rmfsr.benchmark_report --run "$job_dir/real-data" --output "$job_dir/throughput.json"
echo 'Qualification complete. Keep the GPU allocated; release only on explicit user instruction.'
