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
from .model import model_from_config
from .spectral import Spectral
from .flow import meanflow_loss
from .flow_curriculum import config_digest, resolve_transition, training_flow_progress
from .frequency_transition import apply_frequency_transition, publish_best_checkpoint
from .speech_fidelity import build_speech_fidelity, fidelity_config, fidelity_resume_transition
from .stability import (StabilityGuardError, check_stability, hold_training,
                        stability_config, stability_resume_transition)
from .data import SpeechPairs, assert_disjoint


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
    """Verify once when starting a process, never during each checkpoint save."""
    from .preflight import startup_check
    readiness = startup_check(cfg)
    if not readiness['ready']:
        raise ValueError('Training inputs incomplete: ' + '; '.join(readiness['problems']))
    return readiness['input_fingerprints']


def validation_selection_metric(cfg):
    evaluations = cfg.get('inference_evaluations_to_compare', [1, 2, 5])
    if not isinstance(evaluations, (list, tuple)) or not evaluations or any(isinstance(n, bool) or not isinstance(n, int) or n < 1 for n in evaluations):
        raise ValueError('inference_evaluations_to_compare must contain positive integers')
    if len(set(evaluations)) != len(evaluations):
        raise ValueError('inference_evaluations_to_compare must contain unique values')
    allowed = {f'validation_restoration_mse_nfe{n}' for n in evaluations}
    selected = cfg.get('validation_selection_metric', f'validation_restoration_mse_nfe{max(evaluations)}')
    if selected not in allowed:
        raise ValueError('Select a restoration MSE metric from the configured inference evaluations')
    return selected


def alignment_resume_transition(previous, cfg):
    """Allow only the explicitly requested FIR correction across recipe versions."""
    old = previous.get('augmentation_profile', 'legacy')
    new = cfg.get('augmentation_profile', 'legacy')
    if old == new:
        return None
    if (old, new) != ('figure2-v2', 'figure2-v3-aligned') or cfg.get('allow_fir_alignment_fix_on_resume') is not True:
        raise ValueError('Cannot switch augmentation profile on resume without the explicit FIR alignment migration')
    return dict(previous_profile=old, augmentation_profile=new,
                reason='Compensate the symmetric FIR bandlimit delay by 32 samples',
                best_validation_reset=True)


def validation_augmentation_profile(cfg):
    return cfg.get('validation_augmentation_profile', cfg.get('augmentation_profile', 'legacy'))


def augmentation_resume_transition(previous, cfg):
    old = previous.get('augmentation_profile', 'legacy')
    new = cfg.get('augmentation_profile', 'legacy')
    old_val, new_val = validation_augmentation_profile(previous), validation_augmentation_profile(cfg)
    if old == new:
        if old_val != new_val:
            raise ValueError('Cannot change validation augmentation on ordinary resume')
        return None
    if (old, new) == ('figure2-v3-aligned', 'demo-eight-v1'):
        if cfg.get('allow_demo_eight_on_resume') is not True:
            raise ValueError('demo-eight-v1 requires an explicit training-only migration')
        if old_val != new_val or new_val != 'figure2-v3-aligned':
            raise ValueError('Keep validation augmentation unchanged during demo-eight migration')
        return dict(previous_profile=old, augmentation_profile=new,
                    validation_augmentation_profile=new_val, best_validation_reset=False,
                    reason='User approved moderate treatments spanning all eight demo categories')
    if (old, new) == ('demo-eight-v1', 'figure2-v3-aligned'):
        if cfg.get('allow_paper_data_on_resume') is not True:
            raise ValueError('Paper data recipe requires an explicit training-only migration')
        if old_val != new_val or new_val != 'figure2-v3-aligned':
            raise ValueError('Keep validation augmentation unchanged during paper data migration')
        return dict(previous_profile=old, augmentation_profile=new,
                    validation_augmentation_profile=new_val, best_validation_reset=False,
                    reason='User requested the Figure 2 training pipeline at the next session boundary')
    # Preserve the earlier, separately authorized FIR correction behavior.
    transition = alignment_resume_transition(previous, cfg)
    if old_val != old or new_val != new:
        raise ValueError('FIR migration requires matching train/validation profiles')
    return transition


def record_training_augmentation_transition(run, transition, step, previous_best):
    transition = dict(transition, after_step=step,
                      previous_best_validation=previous_best if math.isfinite(previous_best) else None)
    # Retain a full pre-change checkpoint even after normal milestone pruning.
    label = 'figure2' if transition['augmentation_profile'] == 'figure2-v3-aligned' else 'demo-eight'
    for filename in ('latest.pt', 'best-validation.pt'):
        source = run / filename
        archive = run / f'{source.stem}-before-{label}-step-{step:08d}.pt'
        if source.exists() and not archive.exists():
            temporary = archive.with_suffix('.partial')
            shutil.copy2(source, temporary)
            temporary.replace(archive)
        if archive.exists():
            transition['previous_' + source.stem.replace('-', '_') + '_checkpoint'] = archive.name
    path = run / f'augmentation-transition-after-step-{step:08d}.json'
    temporary = path.with_suffix('.partial')
    temporary.write_text(json.dumps(transition, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)
    return transition


def record_alignment_transition(run, transition, step, previous_best):
    """Preserve the old best checkpoint before selection on corrected validation."""
    transition = dict(transition, after_step=step, previous_best_validation=(
        previous_best if math.isfinite(previous_best) else None))
    source = run / 'best-validation.pt'
    archive = run / f'best-validation-before-alignment-step-{step:08d}.pt'
    if source.exists() and not archive.exists():
        temporary = archive.with_suffix('.partial')
        shutil.copy2(source, temporary)
        temporary.replace(archive)
    if archive.exists():
        transition['previous_best_checkpoint'] = archive.name
    path = run / f'augmentation-transition-after-step-{step:08d}.json'
    temporary = path.with_suffix('.partial')
    temporary.write_text(json.dumps(transition, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)
    return transition


def save_checkpoint(path, model, ema, optimizer, step, cfg, pairs, history, best=float('inf'), verified_inputs=None):
    data = dict(model=model.state_dict(), ema=ema.state_dict(), optimizer=optimizer.state_dict(),
                architecture=model.architecture(),
                step=step, config=cfg, torch_rng=torch.get_rng_state(),
                numpy_rng=np.random.get_state(), python_rng=random.getstate(),
                data_rng=pairs.rng.bit_generator.state, history=history[-1000:],
                best_validation=best, validation_selection_metric=validation_selection_metric(cfg),
                input_fingerprints=verified_inputs if verified_inputs is not None else input_fingerprints(cfg))
    frequency = getattr(model, 'frequency_width_transition', None)
    if frequency is not None:
        data['frequency_width_transition'] = copy.deepcopy(frequency)
    conditioning = getattr(model, 'conditioning_architecture_transition', None)
    if conditioning is not None:
        data['conditioning_architecture_transition'] = copy.deepcopy(conditioning)
    if torch.backends.mps.is_available():
        data['mps_rng'] = torch.mps.get_rng_state()
    if torch.cuda.is_available():
        data['cuda_rng'] = torch.cuda.get_rng_state_all()
    temp = path.with_suffix('.partial')
    torch.save(data, temp)
    os.replace(temp, path)


def validate(ema, validation, spectral, cfg, device, output_dir=None, save_audio=False, conditioner=None):
    from .validation import validate_restoration
    return validate_restoration(ema, validation, spectral, cfg, device,
                                output_dir=output_dir, save_audio=save_audio, conditioner=conditioner)


def backward_microbatches(model, batches, spectral, device, progress, count, schedule='legacy', fidelity=None, conditioner=None):
    """Accumulate mean gradients; batches have the same configured size.
    Only the caller clips, takes the optimizer step, and updates EMA.
    """
    totals = dict(loss=0., data_wait_seconds=0.)
    for _ in range(count):
        tic = time.perf_counter()
        clean, damaged = next(batches)
        totals['data_wait_seconds'] += time.perf_counter() - tic
        x, y = spectral.encode(clean).to(device), spectral.encode(damaged).to(device)
        loss_kwargs = {} if fidelity is None else {'fidelity': lambda estimate: fidelity(estimate, clean)}
        if conditioner is not None:
            loss_kwargs['conditioning'] = conditioner(damaged, spectral, y.shape[-1])
        loss, diagnostics = meanflow_loss(model, x, y, progress, schedule=schedule, **loss_kwargs)
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite microbatch loss')
        (loss / count).backward()
        totals['loss'] += loss.detach().item() / count
        for key, value in diagnostics.items():
            totals[key] = totals.get(key, 0.) + value / count
    return totals


def train(cfg, run, resume=None):
    cfg = dict(cfg)
    stability = stability_config(cfg)
    fidelity_config(cfg)
    from .preflight import startup_check
    readiness = startup_check(cfg)
    if not readiness['ready']:
        raise ValueError('Training inputs incomplete: ' + '; '.join(readiness['problems']))
    verified_inputs = readiness['input_fingerprints']
    selection_metric = validation_selection_metric(cfg)
    benchmark_progress = cfg.get('benchmark_flow_progress')
    if benchmark_progress is not None:
        if not cfg.get('benchmark_only') or isinstance(benchmark_progress, bool) or not isinstance(benchmark_progress, (int, float)) or not 0 <= benchmark_progress <= 1:
            raise ValueError('benchmark_flow_progress requires benchmark_only and a value in [0,1]')
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
    schedule_request_path = run / 'flow-schedule-transition.json'
    restore_request_path = run / 'flow-schedule-restore.json'
    frequency_request_path = run / 'frequency-width-transition.json'
    if frequency_request_path.exists() and not resume:
        raise ValueError('A frequency width transition requires a saved checkpoint')
    if not resume and (schedule_request_path.exists() or restore_request_path.exists() or
                       cfg.get('flow_schedule_transition') is not None):
        raise ValueError('A flow schedule transition requires a saved checkpoint')
    if (run / 'latest.pt').exists() and not resume:
        raise ValueError('Run already has a checkpoint; use --resume or a new run directory')
    assert_disjoint(cfg['train_manifest'], cfg['validation_manifest'])
    from .gsm import library_path
    library_path()
    kwargs = dict(seconds=cfg['seconds'], sr=cfg['sample_rate'], noise_dir=cfg.get('noise_dir'),
                  ltas_path=cfg.get('ltas_path'), ffmpeg=get_ffmpeg(cfg.get('ffmpeg')), pilot=cfg['pilot'],
                  augmentation_profile=cfg.get('augmentation_profile','legacy'))
    if cfg.get('input_verification') == 'manifest':
        kwargs.update(resolve_manifest_paths=False, noise_completion=cfg.get('noise_completion'))
    pairs = SpeechPairs(cfg['train_manifest'], seed=seed, **kwargs)
    val_kwargs = {**kwargs, 'noise_dir': cfg.get('validation_noise_dir', cfg.get('noise_dir')),
                  'augmentation_profile': validation_augmentation_profile(cfg)}
    validation = SpeechPairs(cfg['validation_manifest'], seed=seed + 1000, **val_kwargs)
    spectral = Spectral(cfg['sample_rate'])
    fidelity = build_speech_fidelity(cfg, spectral, device)
    model = model_from_config(cfg).to(device)
    from .wavjepa import build_conditioner
    conditioner = build_conditioner(cfg, device)
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['learning_rate'], weight_decay=.01)
    first, history, best = 0, [], float('inf')
    alignment_transition = None
    fidelity_transition = None
    stability_transition = None
    frequency_transition = None
    conditioning_transition = None
    if resume:
        # Only load trusted checkpoints from this experiment.
        checkpoint = torch.load(resume, map_location='cpu', weights_only=False)
        from .conditioning_migration import validate_conditioning_transition
        conditioning_transition = validate_conditioning_transition(checkpoint, cfg)
        fidelity_transition = fidelity_resume_transition(checkpoint['config'], cfg)
        stability_transition = stability_resume_transition(checkpoint['config'], cfg)
        for key in ('channels', 'sample_rate', 'pilot', 'seconds', 'batch_size', 'seed',
                    'learning_rate', 'warmup_steps', 'ema_decay', 'gradient_clip'):
            if checkpoint['config'][key] != cfg[key]:
                raise ValueError(f'Resume mismatch: {key}')
        if checkpoint['config'].get('indexed_batches', False) != cfg.get('indexed_batches', False):
            raise ValueError('Cannot switch augmentation sampling scheme on resume')
        alignment_transition = augmentation_resume_transition(checkpoint['config'], cfg)
        for key, default in [('accumulation_steps', 1), ('flow_schedule', 'legacy'),
                             ('benchmark_flow_progress', None), ('benchmark_only', False)]:
            if checkpoint['config'].get(key, default) != cfg.get(key, default):
                raise ValueError(f'Resume mismatch: {key}')
        if checkpoint['config'].get('schedule_steps', checkpoint['config']['steps']) != cfg.get('schedule_steps', cfg['steps']):
            raise ValueError('Cannot change the learning schedule on resume')
        previous_inputs = checkpoint.get('input_fingerprints', {})
        if previous_inputs.get('integrity_version') != verified_inputs.get('integrity_version'):
            raise ValueError('Checkpoint predates verified-content fingerprints; start a new run')
        if previous_inputs != verified_inputs:
            raise ValueError('Resume dataset content or selected partitions differ')
        if checkpoint.get('validation_selection_metric') != selection_metric:
            raise ValueError('Cannot change the checkpoint selection metric on resume')
        for key, default in [('validation_examples', 1), ('validation_chunk_frames', 10)]:
            if checkpoint['config'].get(key, default) != cfg.get(key, default):
                raise ValueError(f'Resume mismatch: {key}')
        if checkpoint['config'].get('inference_evaluations_to_compare', [1, 2, 5]) != cfg.get('inference_evaluations_to_compare', [1, 2, 5]):
            raise ValueError('Cannot change validation inference evaluations on resume')
        if checkpoint.get('architecture') != model.architecture():
            raise ValueError('Checkpoint uses an older or different model/decoder architecture; start a new run')
        request = json.loads(schedule_request_path.read_text()) if schedule_request_path.exists() else None
        restore_request = json.loads(restore_request_path.read_text()) if restore_request_path.exists() else None
        # The active flow clock is bound to its recipe. Only this explicitly
        # authorized data migration may rebind those two configuration fields;
        # its start step, ramp, target and original request remain unchanged.
        data_changes = ()
        if (alignment_transition is not None and
                alignment_transition['previous_profile'] == 'demo-eight-v1' and
                alignment_transition['augmentation_profile'] == 'figure2-v3-aligned'):
            data_changes = ('augmentation_profile', 'allow_paper_data_on_resume')
        if fidelity_transition is not None:
            data_changes += ('speech_fidelity', 'allow_speech_fidelity_on_resume')
        if stability_transition is not None:
            data_changes += ('stability_guard', 'allow_stability_guard_on_resume')
        flow_transition = resolve_transition(cfg, checkpoint['config'], checkpoint['step'], request,
                                             allowed_config_changes=data_changes,
                                             restore_request=restore_request)
        if flow_transition is not None:
            cfg['flow_schedule_transition'] = flow_transition
        model.load_state_dict(checkpoint['model'])
        ema.load_state_dict(checkpoint['ema'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        if conditioning_transition is not None:
            model.conditioning_architecture_transition = copy.deepcopy(conditioning_transition)
            ema.conditioning_architecture_transition = copy.deepcopy(conditioning_transition)
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
        if flow_transition is not None:
            previous_transition = checkpoint['config'].get('flow_schedule_transition')
            restoring = (flow_transition.get('restoration') is not None and
                         not (previous_transition or {}).get('restoration'))
            if previous_transition is None or restoring:
                label = 'flow-restore' if restoring else 'flow-schedule'
                archive = run / f'before-{label}-step-{first:08d}.pt'
                if not archive.exists():
                    temporary = archive.with_suffix('.partial')
                    shutil.copy2(resume, temporary)
                    temporary.replace(archive)
            active_schedule = run / 'flow-schedule-active.json'
            temporary = active_schedule.with_suffix('.partial')
            temporary.write_text(json.dumps(flow_transition, indent=2, allow_nan=False) + '\n')
            temporary.replace(active_schedule)
        if alignment_transition is not None:
            if alignment_transition['best_validation_reset']:
                alignment_transition = record_alignment_transition(run, alignment_transition, first, best)
                # Only the historical FIR migration changes the validation set.
                best = float('inf')
            else:
                alignment_transition = record_training_augmentation_transition(run, alignment_transition, first, best)
        if fidelity_transition is not None:
            fidelity_transition['after_step'] = first
            archive = run / f'before-speech-fidelity-step-{first:08d}.pt'
            if not archive.exists():
                temporary = archive.with_suffix('.partial')
                shutil.copy2(resume, temporary)
                temporary.replace(archive)
            transition_path = run / 'speech-fidelity-transition.json'
            temporary = transition_path.with_suffix('.partial')
            temporary.write_text(json.dumps(fidelity_transition, indent=2) + '\n')
            temporary.replace(transition_path)
        # This file is read once at process startup, never by the active session.
        # Persist the marker in subsequent checkpoints to prevent repeated scaling.
        frequency_request = (json.loads(frequency_request_path.read_text())
                             if frequency_request_path.exists() else None)
        frequency_recipe = (checkpoint.get('frequency_width_transition') or {}).get('request', frequency_request)
        if (frequency_recipe is not None and
                config_digest(cfg) != frequency_recipe.get('expected_config_sha256')):
            raise ValueError('Frequency width request differs from the current recipe')
        frequency_transition, frequency_applied = apply_frequency_transition(
            model, ema, checkpoint, frequency_request, run, resume)
        if frequency_applied:
            best = float('inf')
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
    manifest = dict(status='training', device=str(device), architecture=model.architecture(), parameters=sum(p.numel() for p in model.parameters()),
                    receptive_frames_per_evaluation=model.receptive_frames,
                    network_context_seconds=(model.receptive_frames - 1) * .01,
                    window_ms=20, paper_parameters=7800000, pilot=cfg['pilot'],
                    quality_validated=False, config=cfg, inputs=verified_inputs,
                    augmentation_transition=alignment_transition,
                    speech_fidelity_transition=fidelity_transition,
                    stability_guard_transition=stability_transition,
                    frequency_width_transition=frequency_transition,
                    conditioning_architecture_transition=conditioning_transition,
                    validation_selection_metric=selection_metric,
                    torch_version=torch.__version__, cuda_version=torch.version.cuda,
                    effective_batch_size=cfg['batch_size'] * accumulation)
    if conditioner is not None:
        manifest.update(conditioning=conditioner.metadata(), streaming_supported=False)
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
            flow_progress = training_flow_progress(step, cfg) if benchmark_progress is None else benchmark_progress
            try:
                diagnostics = backward_microbatches(model, batches, spectral, device, flow_progress,
                                                    accumulation, cfg.get('flow_schedule', 'legacy'), fidelity=fidelity,
                                                    conditioner=conditioner)
            except FloatingPointError as exc:
                if stability is not None:
                    hold_training(stability, run, step + 1,
                                  [dict(kind='nonfinite_microbatch', error=str(exc))])
                raise
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['gradient_clip'],
                                                 error_if_nonfinite=stability is None)
            raw_metrics = diagnostics
            if stability is not None and fidelity is None:
                # Without auxiliary losses the unchanged total is exactly MF.
                raw_metrics = {**diagnostics, 'meanflow_loss': diagnostics['loss']}
            check_stability(stability, raw_metrics, norm, run, step + 1)
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
                       examples_seen=completed * cfg['batch_size'] * accumulation, flow_progress=flow_progress,
                       augmentation_profile=cfg.get('augmentation_profile', 'legacy'), **diagnostics)
            if frequency_transition is not None:
                row['fourier_frequency_width'] = frequency_transition['nominal_frequency_width']
            if cfg.get('conditioning_architecture') is not None:
                row['conditioning_architecture'] = cfg['conditioning_architecture']
            diagonal, gamma = schedule_values(flow_progress, cfg.get('flow_schedule', 'legacy'))
            row.update(flow_diagonal_probability=diagonal, flow_span_gamma=gamma,
                       learning_rate_progress=progress)
            row['validation_augmentation_profile'] = validation_augmentation_profile(cfg)
            review_due = completed in cfg.get('quality_review_updates', [])
            if review_due:
                row['quality_review_due'] = True
            improved = False
            if completed % cfg['validate_every'] == 0 or completed == cfg['steps'] or review_due:
                report_dir = run / 'validation' / f'step-{completed:08d}'
                row.update(validate(ema, validation, spectral, cfg, device,
                                    output_dir=report_dir, save_audio=review_due, conditioner=conditioner))
                score = row[selection_metric]
                if score is None or not math.isfinite(score):
                    raise FloatingPointError('Nonfinite restoration checkpoint score')
                row['validation_selection_metric'] = selection_metric
                if score < best:
                    best, improved = score, True
                    if frequency_transition is not None:
                        frequency_transition['best_selection_pending'] = False
            history.append(row)
            history = history[-1000:]
            with (run / 'metrics.jsonl').open('a') as handle:
                handle.write(json.dumps(row) + '\n')
            print(json.dumps(row), flush=True)
            if cfg.get('max_wall_seconds') and time.perf_counter() - wall >= cfg['max_wall_seconds']:
                stop[0] = True
            conditioning_first_update = (conditioning_transition is not None and
                                         completed == conditioning_transition['after_step'] + 1)
            if completed % cfg['save_every'] == 0 or completed == cfg['steps'] or stop[0] or improved or review_due or conditioning_first_update or completed in cfg.get('initial_checkpoint_updates', []):
                save_checkpoint(run / 'latest.pt', model, ema, optimizer, completed, cfg, pairs, history, best, verified_inputs=verified_inputs)
                if improved:
                    if frequency_transition is not None:
                        publish_best_checkpoint(run / 'latest.pt', run / 'best-validation.pt')
                    else:
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
    except StabilityGuardError as exc:
        manifest.update(status='unstable_training', completed_steps=completed, error=repr(exc),
                        diagnostic=str(exc.path), note='Held before optimizer update; inspect before any explicit resume. Saved checkpoints are unchanged.')
        raise
    except Exception as exc:
        manifest.update(status='failed', completed_steps=completed, error=repr(exc),
                        note='Resume from latest.pt; uncheckpointed steps will be replayed.')
        raise
    finally:
        if frequency_transition is not None:
            manifest['frequency_width_transition'] = copy.deepcopy(frequency_transition)
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
