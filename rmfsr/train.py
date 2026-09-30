import argparse
import copy
import json
import math
import os
from pathlib import Path
import random
import shutil
import signal
import time
import numpy as np
import torch
from .model import RMFSR
from .spectral import Spectral
from .flow import meanflow_loss
from .data import SpeechPairs, assert_disjoint
from .paths import manifest_fingerprint


def device_for(name):
    if name != 'auto':
        return torch.device(name)
    return torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')


def get_ffmpeg(value):
    if value:
        return value
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        return None


def input_fingerprints(cfg):
    result = {key: manifest_fingerprint(cfg[key]) for key in ('train_manifest', 'validation_manifest')}
    if cfg.get('ltas_path'):
        from .download import digest
        result['ltas'] = digest(cfg['ltas_path'])
    if cfg.get('noise_completion'):
        from .download import digest
        result['noise_inventory'] = digest(cfg['noise_completion'])
    return result


def save_checkpoint(path, model, ema, optimizer, step, cfg, pairs, history, best=float('inf')):
    data = dict(model=model.state_dict(), ema=ema.state_dict(), optimizer=optimizer.state_dict(),
                step=step, config=cfg, torch_rng=torch.get_rng_state(),
                numpy_rng=np.random.get_state(), python_rng=random.getstate(),
                data_rng=pairs.rng.bit_generator.state, history=history[-1000:],
                best_validation=best, input_fingerprints=input_fingerprints(cfg))
    if torch.backends.mps.is_available():
        data['mps_rng'] = torch.mps.get_rng_state()
    if torch.cuda.is_available():
        data['cuda_rng'] = torch.cuda.get_rng_state_all()
    temp = path.with_suffix('.partial')
    torch.save(data, temp)
    os.replace(temp, path)


def validate(ema, validation, spectral, cfg, device):
    validation.rng = np.random.default_rng(cfg['seed'] + 1000)
    # Use a separate CPU RNG stream so validation cannot change training's sequence.
    generator = torch.Generator().manual_seed(cfg['seed'] + 2000)
    totals = np.zeros(2)
    count = cfg.get('validation_examples', 1)
    for _ in range(count):
        clean, damaged = validation.batch(1)
        x, y = spectral.encode(clean).to(device), spectral.encode(damaged).to(device)
        t = x.new_full((1,), .5)
        noise = torch.randn(x.shape, generator=generator)
        if cfg.get('validation_pink_noise', False):
            # Same frequency decay as training; normalize the variance per sample.
            weights = torch.arange(x.shape[2]).float().clamp_min(1).rsqrt()[None, None, :, None]
            noise = noise * weights / weights.square().mean().sqrt()
        with torch.no_grad():
            pred = ema(.5 * x + .5 * y + .15 * noise.to(device), y, t, t)
            totals += [(pred - x).square().mean().item(), (y - x).square().mean().item()]
    return dict(validation_dp_mse=totals[0] / count, validation_input_mse=totals[1] / count)


def backward_microbatches(model, batches, spectral, device, progress, count, schedule='legacy'):
    """Accumulate mean gradients; batches have the same configured size.
    Only the caller clips, takes the optimizer step, and updates EMA.
    """
    totals = dict(loss=0., data_wait_seconds=0.)
    for _ in range(count):
        tic = time.perf_counter()
        clean, damaged = next(batches)
        totals['data_wait_seconds'] += time.perf_counter() - tic
        x, y = spectral.encode(clean).to(device), spectral.encode(damaged).to(device)
        loss, diagnostics = meanflow_loss(model, x, y, progress, schedule=schedule)
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite microbatch loss')
        (loss / count).backward()
        totals['loss'] += loss.detach().item() / count
        for key, value in diagnostics.items():
            totals[key] = totals.get(key, 0.) + value / count
    return totals


def train(cfg, run, resume=None):
    from .preflight import check
    readiness = check(cfg)
    if not readiness['ready']:
        raise ValueError('Training inputs incomplete: ' + '; '.join(readiness['problems']))
    torch.set_num_threads(cfg.get('threads', 8))
    seed = cfg['seed']
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    device = device_for(cfg['device'])
    accumulation = cfg.get('accumulation_steps', 1)
    if not isinstance(accumulation, int) or accumulation < 1:
        raise ValueError('accumulation_steps must be a positive integer')
    from .flow import schedule_values
    schedule_values(0, cfg.get('flow_schedule', 'legacy'))
    run = Path(run)
    run.mkdir(parents=True, exist_ok=True)
    if (run / 'latest.pt').exists() and not resume:
        raise ValueError('Run already has a checkpoint; use --resume or a new run directory')
    assert_disjoint(cfg['train_manifest'], cfg['validation_manifest'])
    from .gsm import library_path
    library_path()
    kwargs = dict(seconds=cfg['seconds'], sr=cfg['sample_rate'], noise_dir=cfg.get('noise_dir'),
                  ltas_path=cfg.get('ltas_path'), ffmpeg=get_ffmpeg(cfg.get('ffmpeg')), pilot=cfg['pilot'],
                  augmentation_profile=cfg.get('augmentation_profile','legacy'))
    pairs = SpeechPairs(cfg['train_manifest'], seed=seed, **kwargs)
    val_kwargs = {**kwargs, 'noise_dir': cfg.get('validation_noise_dir', cfg.get('noise_dir'))}
    validation = SpeechPairs(cfg['validation_manifest'], seed=seed + 1000, **val_kwargs)
    spectral = Spectral(cfg['sample_rate'])
    model = RMFSR(channels=cfg['channels']).to(device)
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['learning_rate'], weight_decay=.01)
    first, history, best = 0, [], float('inf')
    if resume:
        # Only load trusted checkpoints from this experiment.
        checkpoint = torch.load(resume, map_location='cpu', weights_only=False)
        for key in ('channels', 'sample_rate', 'pilot', 'seconds', 'batch_size', 'seed',
                    'learning_rate', 'warmup_steps', 'ema_decay', 'gradient_clip'):
            if checkpoint['config'][key] != cfg[key]:
                raise ValueError(f'Resume mismatch: {key}')
        if checkpoint['config'].get('indexed_batches', False) != cfg.get('indexed_batches', False):
            raise ValueError('Cannot switch augmentation sampling scheme on resume')
        if checkpoint['config'].get('augmentation_profile','legacy') != cfg.get('augmentation_profile','legacy'):
            raise ValueError('Cannot switch augmentation profile on resume')
        for key, default in [('accumulation_steps', 1), ('flow_schedule', 'legacy')]:
            if checkpoint['config'].get(key, default) != cfg.get(key, default):
                raise ValueError(f'Resume mismatch: {key}')
        if checkpoint['config'].get('schedule_steps', checkpoint['config']['steps']) != cfg.get('schedule_steps', cfg['steps']):
            raise ValueError('Cannot change the learning schedule on resume')
        if 'input_fingerprints' in checkpoint:
            if checkpoint['input_fingerprints'] != input_fingerprints(cfg):
                raise ValueError('Resume dataset content differs')
        else:
            for key in ('train_manifest', 'validation_manifest'):
                if checkpoint['config'][key] != cfg[key]:
                    raise ValueError(f'Legacy resume mismatch: {key}')
        model.load_state_dict(checkpoint['model'])
        ema.load_state_dict(checkpoint['ema'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        first, history = checkpoint['step'], checkpoint['history']
        best = checkpoint.get('best_validation', float('inf'))
        torch.set_rng_state(checkpoint['torch_rng'])
        np.random.set_state(checkpoint['numpy_rng'])
        random.setstate(checkpoint['python_rng'])
        pairs.rng.bit_generator.state = checkpoint['data_rng']
        if device.type == 'mps' and 'mps_rng' in checkpoint:
            torch.mps.set_rng_state(checkpoint['mps_rng'])
        if device.type == 'cuda' and 'cuda_rng' in checkpoint:
            torch.cuda.set_rng_state_all(checkpoint['cuda_rng'])
        # Preserve a full log from a crashed process but remove uncheckpointed rows
        # from the active log before replaying those steps.
        metrics = run / 'metrics.jsonl'
        if metrics.exists():
            rows = metrics.read_text().splitlines()
            valid = []
            for row in rows:
                try:
                    if json.loads(row)['step'] <= first:
                        valid.append(row)
                except (ValueError, KeyError):
                    pass
            if valid != rows:
                shutil.copy2(metrics, run / f'metrics-before-resume-{time.time_ns()}.jsonl')
                metrics.write_text('\n'.join(valid) + ('\n' if valid else ''))
    if first >= cfg['steps']:
        raise ValueError('Checkpoint has already reached configured steps')
    if cfg.get('indexed_batches', False):
        from .batches import batch_loader
        micro_cfg = {**cfg, 'steps': cfg['steps'] * accumulation}
        batches = iter(batch_loader(cfg['train_manifest'], kwargs, micro_cfg, first * accumulation))
    else:
        batches = (pairs.batch(cfg['batch_size']) for _ in range(first * accumulation, cfg['steps'] * accumulation))
    manifest = dict(status='training', device=str(device), parameters=sum(p.numel() for p in model.parameters()),
                    receptive_frames_per_evaluation=model.receptive_frames,
                    network_context_seconds=(model.receptive_frames - 1) * .01,
                    window_ms=20, paper_parameters=7800000, pilot=cfg['pilot'],
                    quality_validated=False, config=cfg, inputs=input_fingerprints(cfg),
                    torch_version=torch.__version__, cuda_version=torch.version.cuda,
                    effective_batch_size=cfg['batch_size'] * accumulation)
    if device.type == 'cuda':
        manifest['gpu'] = torch.cuda.get_device_name()
        torch.cuda.reset_peak_memory_stats()
    (run / 'run.json').write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest), flush=True)
    stop = [False]
    old_handlers = {}
    def request_stop(signum, frame):
        stop[0] = True
    for sig in (signal.SIGTERM, signal.SIGINT):
        old_handlers[sig] = signal.signal(sig, request_stop)
    wall, completed = time.perf_counter(), first
    try:
        for step in range(first, cfg['steps']):
            tic = time.perf_counter()
            progress = step / cfg.get('schedule_steps', cfg['steps'])
            warm = min(1, (step + 1) / cfg['warmup_steps'])
            lr = cfg['learning_rate'] * warm * (.1 + .9 * .5 * (1 + math.cos(math.pi * min(progress, 1))))
            for group in optimizer.param_groups:
                group['lr'] = lr
            optimizer.zero_grad(set_to_none=True)
            diagnostics = backward_microbatches(model, batches, spectral, device, progress,
                                                accumulation, cfg.get('flow_schedule', 'legacy'))
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['gradient_clip'], error_if_nonfinite=True)
            optimizer.step()
            with torch.no_grad():
                for average, param in zip(ema.parameters(), model.parameters()):
                    average.lerp_(param, 1 - cfg['ema_decay'])
            if device.type == 'mps':
                torch.mps.synchronize()
            elif device.type == 'cuda':
                torch.cuda.synchronize()
            completed = step + 1
            row = dict(step=completed, lr=lr, gradient_norm=float(norm),
                       seconds=time.perf_counter() - tic,
                       examples_seen=completed * cfg['batch_size'] * accumulation, **diagnostics)
            review_due = completed in cfg.get('quality_review_updates', [])
            if review_due:
                row['quality_review_due'] = True
            improved = False
            if completed % cfg['validate_every'] == 0 or completed == cfg['steps']:
                row.update(validate(ema, validation, spectral, cfg, device))
                if row['validation_dp_mse'] < best:
                    best, improved = row['validation_dp_mse'], True
            history.append(row)
            history = history[-1000:]
            with (run / 'metrics.jsonl').open('a') as handle:
                handle.write(json.dumps(row) + '\n')
            print(json.dumps(row), flush=True)
            if cfg.get('max_wall_seconds') and time.perf_counter() - wall >= cfg['max_wall_seconds']:
                stop[0] = True
            if completed % cfg['save_every'] == 0 or completed == cfg['steps'] or stop[0] or improved or review_due:
                save_checkpoint(run / 'latest.pt', model, ema, optimizer, completed, cfg, pairs, history, best)
                if improved:
                    shutil.copy2(run / 'latest.pt', run / 'best-validation.pt')
                if review_due:
                    shutil.copy2(run / 'latest.pt', run / f'review-{completed:08d}.pt')
                if completed % cfg.get('milestone_every', 10000) == 0:
                    shutil.copy2(run / 'latest.pt', run / f'step-{completed:08d}.pt')
                    for old in sorted(run.glob('step-*.pt'))[:-cfg.get('keep_milestones', 3)]:
                        old.unlink()
            if stop[0]:
                break
        status = 'paused_checkpointed' if completed < cfg['steps'] else ('pilot_complete' if cfg['pilot'] else 'training_complete_not_evaluated')
        manifest.update(status=status, completed_steps=completed, wall_seconds=time.perf_counter() - wall)
    except Exception as exc:
        manifest.update(status='failed', completed_steps=completed, error=repr(exc),
                        note='Resume from latest.pt; uncheckpointed steps will be replayed.')
        raise
    finally:
        for sig, handler in old_handlers.items():
            signal.signal(sig, handler)
        if device.type == 'cuda':
            manifest['peak_allocated_gib'] = torch.cuda.max_memory_allocated() / 1024**3
            manifest['peak_reserved_gib'] = torch.cuda.max_memory_reserved() / 1024**3
        (run / 'run.json').write_text(json.dumps(manifest, indent=2))
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--run', required=True)
    parser.add_argument('--resume')
    args = parser.parse_args()
    train(json.loads(Path(args.config).read_text()), args.run, args.resume)
