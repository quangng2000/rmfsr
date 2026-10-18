"""Opt-in pre-step protection for raw training metrics, never a loss modifier.

Bounds must be calibrated for the recipe and effective-batch reduction. This
module only writes a diagnostic and raises: no optimizer, retry or pod control.
"""
import json
import math
import os
from pathlib import Path
import tempfile


METRICS = ('meanflow_loss', 'target_rms', 'gradient_norm')


class StabilityGuardError(FloatingPointError):
    def __init__(self, path, report):
        self.path, self.report = Path(path), report
        super().__init__(f'Unstable training held before update {report["attempted_step"]}; diagnostic: {path}')


def stability_config(config):
    value = config.get('stability_guard')
    if value is None:
        return None
    keys = {'version', 'thresholds', 'required_exceedances', 'single_extreme_factor'}
    if not isinstance(value, dict) or set(value) != keys:
        raise ValueError('stability_guard requires version, thresholds, required_exceedances and single_extreme_factor')
    if type(value['version']) is not int or value['version'] != 1:
        raise ValueError('Unknown stability_guard version')
    thresholds = value['thresholds']
    if not isinstance(thresholds, dict) or set(thresholds) != set(METRICS):
        raise ValueError('stability_guard thresholds must specify all three raw metrics')
    for key, bound in thresholds.items():
        if isinstance(bound, bool) or not isinstance(bound, (int, float)) or not math.isfinite(bound) or bound <= 0:
            raise ValueError(f'stability_guard threshold {key} must be finite and positive')
    count, factor = value['required_exceedances'], value['single_extreme_factor']
    if type(count) is not int or not 1 <= count <= len(METRICS):
        raise ValueError('stability_guard required_exceedances must be an integer in [1,3]')
    if isinstance(factor, bool) or not isinstance(factor, (int, float)) or not math.isfinite(factor) or factor <= 1:
        raise ValueError('stability_guard single_extreme_factor must be finite and greater than one')
    return {**value, 'thresholds': dict(thresholds)}


def stability_resume_transition(previous, current):
    old, new = stability_config(previous), stability_config(current)
    if old == new:
        return None
    if old is not None or new is None or current.get('allow_stability_guard_on_resume') is not True:
        raise ValueError('Changing stability guard on resume requires explicit one-time activation')
    return dict(previous=None, stability_guard=new, optimizer_and_objective_unchanged=True)


def _json_number(value):
    if value is None or math.isfinite(value):
        return value
    if math.isnan(value):
        return 'NaN'
    return '+Infinity' if value > 0 else '-Infinity'


def hold_training(settings, run, attempted_step, reasons, metrics=None):
    """Persist one compact atomic diagnostic, then fail closed before any step."""
    report = dict(status='unstable_training', attempted_step=attempted_step,
                  last_completed_step=attempted_step - 1, optimizer_step_performed=False,
                  reasons=reasons, raw_metrics={k: _json_number(v) for k, v in (metrics or {}).items()},
                  settings=settings, action='HOLD; no automatic retry or rollback. Inspect before resuming.')
    directory = Path(run)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f'stability-hold-before-step-{attempted_step:08d}.json'
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', dir=directory, prefix=path.name + '.', suffix='.partial', delete=False) as handle:
            temporary = Path(handle.name)
            json.dump(report, handle, indent=2, allow_nan=False)
            handle.write('\n')
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    raise StabilityGuardError(path, report)


def check_stability(settings, diagnostics, gradient_norm, run, attempted_step):
    """Check effective-update aggregates and the accumulated preclip gradient."""
    if settings is None:
        return
    metrics = {key: diagnostics.get(key) for key in METRICS[:-1]}
    metrics['gradient_norm'] = float(gradient_norm)
    missing = [key for key, value in metrics.items() if value is None]
    nonfinite = [key for key, value in metrics.items() if value is not None and not math.isfinite(value)]
    exceeded = [key for key, value in metrics.items()
                if value is not None and value > settings['thresholds'][key]]
    extreme = [key for key in exceeded if metrics[key] / settings['thresholds'][key] > settings['single_extreme_factor']]
    reasons = []
    if missing:
        reasons.append(dict(kind='missing_raw_metrics', metrics=missing))
    if nonfinite:
        reasons.append(dict(kind='nonfinite_raw_metrics', metrics=nonfinite))
    if len(exceeded) >= settings['required_exceedances']:
        reasons.append(dict(kind='raw_threshold_count', metrics=exceeded))
    if extreme:
        reasons.append(dict(kind='single_extreme_raw_metric', metrics=extreme))
    if reasons:
        hold_training(settings, run, attempted_step, reasons, metrics)
