"""Create a separate conditioned experiment from a trusted RMFSR checkpoint.

Usage: python -m rmfsr.wavjepa_fork --checkpoint old/latest.pt --run new-run
This writes only the new directory. It never starts/stops training or a pod.
The optimizer moments, EMA backbone, RNG state, update counter, data recipe,
loss, and flow/LR clocks survive. New adapter parameters start without moments.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path

import torch

from .flow_curriculum import resolve_transition
from .model import model_from_checkpoint, model_from_config
from .wavjepa import DEFAULT_OPTIONS


def conditioned_checkpoint(checkpoint):
    previous = checkpoint['config']
    if previous.get('model_type') != 'efficient-v1' or previous.get('wavjepa') is not None:
        raise ValueError('Initialize conditioning only from an unconditioned efficient-v1 checkpoint')
    config = {**previous, 'wavjepa': dict(DEFAULT_OPTIONS), 'validation_chunk_frames': 0}
    transition = resolve_transition(config, previous, checkpoint['step'],
                                    allowed_config_changes=('wavjepa', 'validation_chunk_frames'))
    if transition is not None:
        config['flow_schedule_transition'] = transition
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(previous['seed'])
        old = model_from_checkpoint(checkpoint)
        new = model_from_config(config)
    old.load_state_dict(checkpoint['model'], strict=True)
    added = {key: value for key, value in new.state_dict().items() if key.startswith('conditioning.')}
    if set(new.state_dict()) != set(old.state_dict()) | set(added):
        raise ValueError('Conditioning migration unexpectedly changed the backbone')
    payload = dict(checkpoint)
    for field in ('model', 'ema'):
        if set(checkpoint[field]) != set(old.state_dict()):
            raise ValueError(f'Unexpected backbone state in {field}')
        payload[field] = {**checkpoint[field], **{key: value.clone() for key, value in added.items()}}
        new.load_state_dict(payload[field], strict=True)

    # Map optimizer state by parameter name, not guessed numeric indices.
    saved = checkpoint['optimizer']
    if len(saved['param_groups']) != 1:
        raise ValueError('Expected this trainer\'s single AdamW parameter group')
    group = saved['param_groups'][0]
    names = [name for name, _ in old.named_parameters()]
    new_names = [name for name, _ in new.named_parameters()]
    if len(names) != len(group['params']) or set(saved['state']) - set(group['params']):
        raise ValueError('Optimizer state does not match the source model')
    by_name = dict(zip(names, group['params']))
    new_ids = {name: i for i, name in enumerate(new_names)}
    state = {new_ids[name]: saved['state'][old_id]
             for name, old_id in by_name.items() if old_id in saved['state']}
    payload['optimizer'] = dict(state=state, param_groups=[{**group, 'params': list(range(len(new_names)))}])
    payload.update(config=config, architecture=new.architecture(), best_validation=float('inf'))
    payload['conditioning_initialization'] = dict(
        after_step=checkpoint['step'], new_parameters=sum(p.numel() for name, p in new.named_parameters()
                                                          if name.startswith('conditioning.')),
        optimizer_backbone_preserved=True, encoder_frozen=True,
        validation_protocol='offline; best score reset; retain old run for comparisons',
        previous_architecture=copy.deepcopy(checkpoint['architecture']))
    return payload


def fork_checkpoint(source, destination):
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if destination.exists():
        raise ValueError('Use a new, nonexistent run directory; existing runs are never overwritten')
    # Local files from this project only: PyTorch pickle is not safe for strangers' checkpoints.
    with source.open('rb') as handle:
        hasher = hashlib.sha256()
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            hasher.update(block)
        digest = hasher.hexdigest()
        handle.seek(0)
        payload = conditioned_checkpoint(torch.load(handle, map_location='cpu', weights_only=False))
    payload['conditioning_initialization'].update(source=str(source), source_sha256=digest)
    destination.mkdir(parents=True, exist_ok=False)
    temporary = destination / 'latest.partial'
    torch.save(payload, temporary)
    temporary.replace(destination / 'latest.pt')
    (destination / 'config.json').write_text(json.dumps(payload['config'], indent=2) + '\n')
    (destination / 'conditioning-initialization.json').write_text(
        json.dumps(payload['conditioning_initialization'], indent=2) + '\n')
    return payload['conditioning_initialization']


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--run', required=True)
    args = parser.parse_args()
    print(json.dumps(fork_checkpoint(args.checkpoint, args.run), indent=2))
