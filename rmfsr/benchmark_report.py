"""Summarize measured optimizer updates from a bounded real-data run."""
import argparse
import json
from pathlib import Path
import statistics


def summarize(run, warmup=10):
    run = Path(run)
    manifest = json.loads((run / 'run.json').read_text())
    rows = [json.loads(line) for line in (run / 'metrics.jsonl').read_text().splitlines() if line]
    measured = rows[warmup:]
    if not measured:
        raise ValueError('Not enough completed updates after warm-up')
    times = [row['seconds'] for row in measured]
    mean = statistics.mean(times)
    cfg = manifest['config']
    return dict(
        gpu=manifest.get('gpu'), architecture=manifest['architecture'],
        effective_batch_size=manifest['effective_batch_size'], crop_seconds=cfg['seconds'],
        flow_progress=cfg.get('benchmark_flow_progress'),
        completed_updates=len(rows), discarded_warmup_updates=warmup,
        measured_updates=len(times), mean_seconds_per_update=mean,
        median_seconds_per_update=statistics.median(times), optimizer_updates_per_second=1 / mean,
        data_wait_seconds_per_update=statistics.mean(row.get('data_wait_seconds', 0) for row in measured),
        estimated_300k_update_days_excluding_validation_and_checkpoints=mean * 300000 / 86400,
        run_wall_seconds=manifest.get('wall_seconds'),
        peak_allocated_gib=manifest.get('peak_allocated_gib'),
        peak_reserved_gib=manifest.get('peak_reserved_gib'),
        note='Real augmentation, STFT, transfer, accumulated backward passes, optimizer and EMA included. '
             'Update timings exclude validation/checkpointing; run_wall_seconds includes them. '
             'Throughput benchmark at the configured flow progress; not a quality measurement.',
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--warmup', type=int, default=10)
    args = parser.parse_args()
    report = summarize(args.run, args.warmup)
    Path(args.output).write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
