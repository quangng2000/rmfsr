"""Verify complete inputs and exercise the production CPU data path before GPU rental."""
import argparse
import json
import time
from pathlib import Path

import torch

from .batches import batch_loader
from .corpus import atomic_json
from .data import SpeechPairs
from .preflight import check
from .spectral import Spectral
from .train import get_ffmpeg


DATA_RECIPE_KEYS = (
    'train_manifest', 'validation_manifest', 'test_manifest', 'noise_dir',
    'validation_noise_dir', 'noise_completion', 'ltas_path', 'sample_rate',
    'seconds', 'batch_size', 'accumulation_steps', 'augmentation_profile',
)


def data_recipe(config):
    return {key: config.get(key) for key in DATA_RECIPE_KEYS}


def validate_batch(clean, damaged, config, spectral):
    expected = (config['batch_size'], round(config['sample_rate'] * config['seconds']))
    for label, wave in [('target', clean), ('input', damaged)]:
        if tuple(wave.shape) != expected or wave.dtype != torch.float32:
            raise ValueError(f'Invalid {label} batch: {tuple(wave.shape)}, {wave.dtype}; expected {expected}, float32')
        if not torch.isfinite(wave).all():
            raise ValueError(f'Nonfinite {label} waveform')
        encoded = spectral.encode(wave)
        if not torch.isfinite(encoded).all():
            raise ValueError(f'Nonfinite {label} training features')
    return {'waveform_shape': list(expected), 'feature_shape': list(encoded.shape)}


def check_pipeline(config, batches=8, readiness_output=None):
    if config.get('pilot'):
        raise ValueError('Data readiness requires the full corpus, not pilot inputs')
    if not isinstance(batches, int) or batches < 1:
        raise ValueError('At least one actual training batch must be checked')
    start = time.monotonic()
    readiness = check(config)
    readiness['data_recipe'] = data_recipe(config)
    if readiness_output:
        atomic_json(readiness_output, readiness)
    if not readiness['ready']:
        raise ValueError('Training inputs incomplete: ' + '; '.join(readiness['problems']))
    torch.set_num_threads(config.get('threads', 2))
    kwargs = dict(seconds=config['seconds'], sr=config['sample_rate'],
                  noise_dir=config.get('noise_dir'), ltas_path=config.get('ltas_path'),
                  ffmpeg=get_ffmpeg(config.get('ffmpeg')), pilot=False,
                  augmentation_profile=config.get('augmentation_profile', 'legacy'))
    # Same indexed worker path as training; only the number of requested batches
    # and the device change. No model, optimizer or training checkpoint is created.
    cpu_config = {**config, 'device': 'cpu', 'steps': batches}
    spectral = Spectral(config['sample_rate'])
    shapes = None
    for clean, damaged in batch_loader(config['train_manifest'], kwargs, cpu_config, first=0):
        shapes = validate_batch(clean, damaged, config, spectral)
    validation = SpeechPairs(config['validation_manifest'], seed=config['seed'] + 1000,
                             **{**kwargs, 'noise_dir': config.get('validation_noise_dir')})
    for _ in range(2):
        validate_batch(*validation.batch(config['batch_size']), config, spectral)
    return dict(ready=True, status='actual_data_pipeline_passed', device='cpu',
                checked_at=time.time(), elapsed_seconds=time.monotonic() - start,
                training_batches=batches, validation_batches=2,
                training_examples=batches * config['batch_size'],
                validation_examples=2 * config['batch_size'],
                data_recipe=data_recipe(config), input_fingerprints=readiness['input_fingerprints'],
                **shapes)


def require_prepared_data(config, readiness_path='runs/data-readiness.json',
                          pipeline_path='runs/data-pipeline-check.json',
                          status_path='runs/data-preparation/status'):
    if Path(status_path).read_text().strip() != 'ready':
        raise ValueError('CPU data preparation has not completed')
    readiness = json.loads(Path(readiness_path).read_text())
    pipeline = json.loads(Path(pipeline_path).read_text())
    if (not readiness.get('ready') or not pipeline.get('ready') or
            pipeline.get('status') != 'actual_data_pipeline_passed'):
        raise ValueError('Finish CPU corpus and actual data-pipeline checks before GPU qualification')
    if readiness.get('pilot') or readiness.get('data_recipe') != data_recipe(config) or pipeline.get('data_recipe') != data_recipe(config):
        raise ValueError('Prepared data does not match the requested training recipe')
    if not readiness.get('input_fingerprints') or readiness['input_fingerprints'] != pipeline.get('input_fingerprints'):
        raise ValueError('CPU readiness and pipeline dataset fingerprints differ')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--batches', type=int, default=8)
    parser.add_argument('--readiness-output', default='runs/data-readiness.json')
    parser.add_argument('--output', default='runs/data-pipeline-check.json')
    args = parser.parse_args()
    try:
        result = check_pipeline(json.loads(Path(args.config).read_text()), args.batches,
                                args.readiness_output)
    except Exception as exc:
        atomic_json(args.output, {'ready': False, 'status': 'failed', 'error': repr(exc)})
        raise
    atomic_json(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
