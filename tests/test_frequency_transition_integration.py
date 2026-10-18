import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from rmfsr.flow_curriculum import config_digest
from rmfsr.frequency_transition import tensor_sha256
from rmfsr.train import train
from tests.test_training_integration import fixture


class TinyModel(torch.nn.Module):
    receptive_frames = 1

    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Module()
        self.embedding.register_buffer('frequencies', torch.arange(32, dtype=torch.float32) / 16)
        self.weight = torch.nn.Parameter(torch.tensor([.5, -.25]))

    def architecture(self):
        return {'test_model': 'frequency-resume'}


def backward(model, *_args, **_kwargs):
    loss = model.weight.square().mean()
    loss.backward()
    return {'loss': float(loss.detach()), 'target_rms': .05}


class FrequencyTransitionIntegrationTests(unittest.TestCase):
    def test_natural_session_boundary_preserves_optimizer_and_never_rescales_twice(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cfg = {**fixture(root), 'steps': 4, 'max_wall_seconds': 1e-9}
            run = root / 'run'
            latest = run / 'latest.pt'
            scores = [.2, .8, .7, .6]
            with patch('rmfsr.train.model_from_config', side_effect=lambda cfg: TinyModel()), \
                 patch('rmfsr.train.backward_microbatches', side_effect=backward), \
                 patch('rmfsr.train.validate', side_effect=[
                     {'validation_restoration_mse_nfe1': score} for score in scores]), \
                 contextlib.redirect_stdout(io.StringIO()):
                first = train(cfg, run)
                self.assertEqual(first['completed_steps'], 1)
                self.assertEqual(first['status'], 'paused_checkpointed')
                source = torch.load(latest, weights_only=False)
                source_bytes = latest.read_bytes()
                frequencies = source['model']['embedding.frequencies'].clone()
                request = dict(version=1, transition_id='width1-to4-integration', from_width=1,
                               to_width=4, minimum_checkpoint_step=1,
                               expected_config_sha256=config_digest(cfg),
                               source_frequency_sha256={kind: tensor_sha256(frequencies)
                                                        for kind in ('model', 'ema')},
                               target_frequency_sha256={kind: tensor_sha256(frequencies * 4)
                                                        for kind in ('model', 'ema')})
                request_path = run / 'frequency-width-transition.json'
                request_path.write_text(json.dumps(request))
                for step in (2, 3, 4):
                    if step == 4:
                        request_path.unlink()
                    manifest = train(cfg, run, latest)
                    saved = torch.load(latest, weights_only=False)
                    self.assertEqual(saved['step'], step)
                    self.assertEqual(saved['config'], source['config'])
                    self.assertEqual(saved['best_validation'], scores[step - 1])
                    for kind in ('model', 'ema'):
                        self.assertTrue(torch.equal(saved[kind]['embedding.frequencies'], frequencies * 4))
                    for state in saved['optimizer']['state'].values():
                        self.assertEqual(int(state['step']), step)
                    marker = saved['frequency_width_transition']
                    self.assertEqual(marker['after_step'], 1)
                    self.assertFalse(marker['best_selection_pending'])
                    self.assertEqual(manifest['frequency_width_transition'], marker)
                    self.assertEqual(saved['history'][-1]['fourier_frequency_width'], 4)
                    self.assertEqual(saved['history'][-1]['learning_rate_progress'], (step - 1) / 20)
                    self.assertEqual((run / marker['archives']['source_checkpoint']['path']).read_bytes(), source_bytes)
                    old_best = torch.load(run / marker['archives']['previous_best_checkpoint']['path'], weights_only=False)
                    self.assertEqual(old_best['step'], 1)
                    self.assertEqual(old_best['best_validation'], .2)
                best = torch.load(run / 'best-validation.pt', weights_only=False)
                self.assertEqual(best['step'], 4)
                self.assertEqual(manifest['status'], 'pilot_complete')
                self.assertEqual(len((run / 'metrics.jsonl').read_text().splitlines()), 4)


if __name__ == '__main__':
    unittest.main()
