"""Run a fixed training recipe across checkpointed sessions after qualification."""
import argparse
import fcntl
import json
import math
from pathlib import Path
import subprocess
import sys
import time


def qualification_passed(qualification, config):
    qualification = Path(qualification)
    status = (qualification / 'status').read_text().strip()
    if status.startswith('failed:'):
        raise RuntimeError(f'Qualification {status}')
    if status != 'complete':
        return False
    report = json.loads((qualification / 'throughput.json').read_text())
    run = json.loads((qualification / 'real-data/run.json').read_text())
    benchmark = run['config']
    if run['status'] != 'training_complete_not_evaluated' or run['completed_steps'] != benchmark['steps']:
        raise RuntimeError('Qualification did not finish its configured updates')
    if not benchmark.get('benchmark_only') or benchmark.get('pilot'):
        raise RuntimeError('Qualification must be a real-data throughput benchmark')
    for key in ('model_type', 'channels', 'groups', 'attention_rank', 'folded_width',
                'batch_size', 'accumulation_steps', 'seconds', 'sample_rate',
                'augmentation_profile', 'flow_schedule'):
        if benchmark.get(key) != config.get(key):
            raise RuntimeError(f'Qualification recipe differs: {key}')
    duration = report['mean_seconds_per_update']
    if not math.isfinite(duration) or duration <= 0:
        raise RuntimeError('Qualification timing is invalid')
    return True


def write_status(control, phase, **fields):
    temporary = control / 'status.partial'
    temporary.write_text(json.dumps(dict(phase=phase, updated_at=time.time(), **fields), indent=2) + '\n')
    temporary.replace(control / 'status.json')


def run_sessions(config_path, run, control, execute=subprocess.run):
    config = json.loads(Path(config_path).read_text())
    if config.get('pilot') or config.get('benchmark_only') or config['steps'] < 1:
        raise ValueError('Continuous training requires a full-training configuration')
    run, control = Path(run), Path(control)
    run.mkdir(parents=True, exist_ok=True)
    control.mkdir(parents=True, exist_ok=True)
    saved = run / 'training-config.json'
    if saved.exists() and json.loads(saved.read_text()) != config:
        raise ValueError('Continuous training configuration changed; preserve the saved recipe')
    saved.write_text(json.dumps(config, indent=2) + '\n')
    previous = -1
    if (run / 'latest.pt').exists() and (run / 'run.json').exists():
        existing = json.loads((run / 'run.json').read_text())
        if existing.get('status') in ('paused_checkpointed', 'training_complete_not_evaluated'):
            previous = existing.get('completed_steps', -1)
        if previous >= config['steps'] and existing['status'] == 'training_complete_not_evaluated':
            write_status(control, 'complete', completed_steps=previous, target_steps=config['steps'])
            return
    while True:
        write_status(control, 'training', completed_steps=max(previous, 0), target_steps=config['steps'])
        command = [sys.executable, '-u', '-m', 'rmfsr.train', '--config', str(saved), '--run', str(run)]
        if (run / 'latest.pt').exists():
            command += ['--resume', str(run / 'latest.pt')]
        with (run / 'console.log').open('a') as output:
            execute(command, stdout=output, stderr=subprocess.STDOUT, check=True)
        result = json.loads((run / 'run.json').read_text())
        completed = result['completed_steps']
        if completed <= previous or not (run / 'latest.pt').exists():
            raise RuntimeError('Training session exited without advancing a saved checkpoint')
        if completed >= config['steps'] and result['status'] == 'training_complete_not_evaluated':
            write_status(control, 'complete', completed_steps=completed, target_steps=config['steps'])
            return
        if result['status'] != 'paused_checkpointed':
            raise RuntimeError(f'Unexpected training session status: {result["status"]}')
        previous = completed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/efficient.json')
    parser.add_argument('--run', default='runs/efficient')
    parser.add_argument('--qualification', default='runs/qualification')
    parser.add_argument('--control', default='runs/continuous')
    parser.add_argument('--skip-qualification', action='store_true',
                        help='Start the authorized full run directly without a separate benchmark')
    args = parser.parse_args()
    control = Path(args.control)
    control.mkdir(parents=True, exist_ok=True)
    with (control / 'controller.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            config = json.loads(Path(args.config).read_text())
            if not args.skip_qualification:
                write_status(control, 'waiting_for_qualification', target_steps=config['steps'])
                while not qualification_passed(args.qualification, config):
                    time.sleep(30)
            else:
                print('Qualification skipped explicitly; starting full training.', flush=True)
            run_sessions(args.config, args.run, control)
        except Exception as exc:
            write_status(control, 'failed', error=repr(exc))
            raise


if __name__ == '__main__':
    main()
