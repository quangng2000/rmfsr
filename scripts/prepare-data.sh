#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
# Uses the caller's Python environment. Can run on a CPU data-preparation host.
mkdir -p data runs
python -m rmfsr.corpus --root data --corpus all 2>&1 | tee -a runs/data-preparation.log
python -m rmfsr.preflight --config configs/runpod.json --output runs/data-readiness.json
