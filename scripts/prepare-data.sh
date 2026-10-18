#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
# Uses the caller's Python environment. Can run on a CPU data-preparation host.
config="${1:-configs/efficient.json}"
mode="${2:-}"
if [[ -n "$mode" && "$mode" != --checks-only ]]; then
  echo 'Usage: prepare-data.sh [config] [--checks-only]' >&2
  exit 2
fi
job_dir=runs/data-preparation
mkdir -p data "$job_dir"
# Hold the lock for this shell and its children; two hosts must not convert the
# same archive or publish completion manifests concurrently on the shared volume.
exec 9>"$job_dir/preparation.lock"
flock -n 9 || { echo 'A data preparation job is already running.' >&2; exit 2; }
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1
children=()
finish() {
  result=$?
  trap - EXIT
  for child in "${children[@]}"; do kill "$child" 2>/dev/null || true; done
  if [[ "$result" -eq 0 ]]; then
    echo ready > "$job_dir/status"
  else
    echo "failed:$result" > "$job_dir/status"
  fi
  exit "$result"
}
trap finish EXIT
trap 'exit 143' TERM
echo preparing_corpora > "$job_dir/status"
# Invalidate old readiness before changing any data.
python - <<'PY'
from rmfsr.corpus import atomic_json
for path in ('runs/data-readiness.json', 'runs/data-pipeline-check.json'):
    atomic_json(path, {'ready': False, 'status': 'preparing_corpora'})
PY
if [[ "$mode" != --checks-only ]]; then
  for corpus in ears daps dns; do
    python -m rmfsr.corpus --root data --corpus "$corpus" > "$job_dir/data-$corpus.log" 2>&1 &
    children+=("$!")
  done
  for child in "${children[@]}"; do wait "$child"; done
  children=()
fi
echo checking_full_data_and_batches > "$job_dir/status"
python -m rmfsr.data_smoke --config "$config" --batches 8 \
  --readiness-output runs/data-readiness.json --output runs/data-pipeline-check.json \
  > "$job_dir/check.log" 2>&1 &
children+=("$!")
wait "${children[0]}"
children=()
python scripts/preview-data.py --config "$config" --output "$job_dir/previews" --count 8 \
  > "$job_dir/preview.log" 2>&1 &
children+=("$!")
wait "${children[0]}"
children=()
echo 'Full corpus and actual CPU training/validation batches passed. Data is ready for GPU qualification.'
