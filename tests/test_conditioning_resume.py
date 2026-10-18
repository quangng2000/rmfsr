"""Small CPU integration checks for a saved conditioning-architecture handoff."""
import contextlib
import copy
import hashlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from rmfsr.conditioning import ARCHITECTURE
from rmfsr.conditioning_migration import convert_checkpoint, validate_conditioning_transition
from rmfsr.flow_curriculum import config_digest
from rmfsr.frequency_transition import tensor_sha256
from rmfsr.model import model_from_checkpoint, model_from_config
from rmfsr.preflight import startup_check
from rmfsr.train import backward_microbatches, save_checkpoint, train
from tests.test_training_integration import fixture


def seed(root, max_wall_seconds=1e-9):
    """Trusted production-shaped metadata with a tiny model and two fixture files."""
    cfg = {**fixture(root), 'seconds': .04, 'steps': 9, 'schedule_steps': 20,
           'model_type': 'efficient-v1', 'channels': [4] * 5, 'groups': 4,
           'attention_rank': 4, 'folded_width': 4,
           'encoder_dilations': [1] * 5, 'tcn_dilations': [1] * 4,
           'flow_schedule': 'figure1-cosine-approx', 'max_wall_seconds': max_wall_seconds,
           'save_every': 100, 'validate_every': 100}
    torch.manual_seed(8)
    model = model_from_config(cfg)
    model.embedding.frequencies.mul_(.25)
    with torch.no_grad():
        model.head.conv.bias.fill_(.01)
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    with torch.no_grad():
        ema.head.conv.bias.fill_(.02)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['learning_rate'], weight_decay=.01)
    # A nonempty old optimizer makes accidental moment reuse observable.
    for parameter in model.parameters():
        optimizer.state[parameter] = dict(step=torch.tensor(5.),
            exp_avg=torch.ones_like(parameter) * .01,
            exp_avg_sq=torch.ones_like(parameter) * .001)
    frequencies = {kind: tensor_sha256(module.embedding.frequencies)
                   for kind, module in [('model', model), ('ema', ema)]}
    model.frequency_width_transition = dict(nominal_frequency_width=4,
        request=dict(target_frequency_sha256=frequencies))
    path = root / 'source.pt'
    pairs = SimpleNamespace(rng=np.random.default_rng(3))
    inputs = startup_check(cfg)['input_fingerprints']
    save_checkpoint(path, model, ema, optimizer, 5, cfg, pairs, [], verified_inputs=inputs)
    request = dict(version=1, transition_id='conditioning-test', expected_step=5,
        expected_source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        expected_config_sha256=config_digest(cfg))
    run = root / 'run'
    run.mkdir()
    converted = run / 'latest.pt'
    convert_checkpoint(path, converted, request)
    checkpoint = torch.load(converted, map_location='cpu', weights_only=False)
    return cfg, path, run, checkpoint


class ConditioningResumeTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)

    def test_migrated_checkpoint_resumes_full_jvp_and_preserves_clocks_and_marker(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            root = Path(directory)
            old_cfg, source_path, run, initial = seed(root)
            source_bytes = source_path.read_bytes()
            cfg = initial['config']
            marker = initial['conditioning_architecture_transition']
            self.assertEqual(cfg['conditioning_architecture'], ARCHITECTURE)
            self.assertEqual(initial['step'], 5)
            self.assertEqual(initial['optimizer']['state'], {})
            self.assertNotIn('frequency_width_transition', initial)
            # Raw and EMA are distinct migrated copies, not an EMA reset to raw.
            torch.testing.assert_close(initial['model']['head.conv.bias'], torch.full((2,), .01))
            torch.testing.assert_close(initial['ema']['head.conv.bias'], torch.full((2,), .02))
            latest = run / 'latest.pt'
            for completed in (6, 7):
                # Force a nonzero span so the real trainer must exercise full JVP.
                times = (torch.tensor([.8]), torch.tensor([.2]))
                with patch('rmfsr.flow.sample_times', return_value=times), \
                     patch('torch.func.jvp', wraps=torch.func.jvp) as jvp:
                    result = train(cfg, run, latest)
                self.assertEqual(jvp.call_count, 1)
                primals, tangents = jvp.call_args.args[1:3]
                self.assertTrue(torch.equal(tangents[1], torch.zeros_like(primals[1])))
                self.assertTrue(torch.equal(tangents[2], torch.ones_like(primals[2])))
                self.assertGreater(float(tangents[0].abs().max()), 0)
                saved = torch.load(latest, map_location='cpu', weights_only=False)
                self.assertEqual(result['status'], 'paused_checkpointed')
                self.assertEqual(saved['step'], completed)
                self.assertEqual(saved['config'], cfg)
                self.assertEqual(saved['conditioning_architecture_transition'], marker)
                self.assertEqual(result['conditioning_architecture_transition'], marker)
                self.assertEqual(validate_conditioning_transition(saved, cfg), marker)
                self.assertIsNone(result['frequency_width_transition'])
                row = saved['history'][-1]
                self.assertEqual(row['conditioning_architecture'], ARCHITECTURE)
                self.assertNotIn('fourier_frequency_width', row)
                self.assertEqual(row['learning_rate_progress'], (completed - 1) / 20)
                self.assertEqual(row['flow_progress'], (completed - 1) / 20)
                self.assertTrue(np.isfinite(row['loss']))
                self.assertGreater(len(saved['optimizer']['state']), 0)
                self.assertEqual({int(state['step']) for state in saved['optimizer']['state'].values()},
                                 {completed - 5})
                restored = model_from_checkpoint(saved)
                restored.load_state_dict(saved['model'], strict=True)
                self.assertEqual(restored.architecture(), saved['architecture'])
            self.assertEqual(source_path.read_bytes(), source_bytes)
            self.assertEqual(cfg['flow_schedule'], old_cfg['flow_schedule'])
            self.assertEqual(len((run / 'metrics.jsonl').read_text().splitlines()), 2)

    def test_resume_rejects_replaced_architecture_and_future_transition(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            _, _, run, original = seed(Path(directory))
            cfg = original['config']
            latest = run / 'latest.pt'
            damaged = copy.deepcopy(original)
            damaged['architecture'].pop('conditioning_architecture')
            torch.save(damaged, latest)
            with self.assertRaises(ValueError):
                train(cfg, run, latest)
            self.assertFalse((run / 'metrics.jsonl').exists())
            damaged = copy.deepcopy(original)
            damaged['conditioning_architecture_transition']['after_step'] = original['step'] + 1
            torch.save(damaged, latest)
            with self.assertRaises(ValueError):
                train(cfg, run, latest)
            self.assertFalse((run / 'metrics.jsonl').exists())

    def test_first_new_architecture_update_is_saved_before_normal_checkpoint_interval(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            _, _, run, initial = seed(Path(directory), max_wall_seconds=0)
            count = 0

            def first_update_only(*args, **kwargs):
                nonlocal count
                count += 1
                if count == 2:
                    raise FloatingPointError('fixture ends after one committed update')
                return backward_microbatches(*args, **kwargs)

            with patch('rmfsr.train.backward_microbatches', side_effect=first_update_only):
                with self.assertRaisesRegex(FloatingPointError, 'fixture ends'):
                    train(initial['config'], run, run / 'latest.pt')
            saved = torch.load(run / 'latest.pt', map_location='cpu', weights_only=False)
            self.assertEqual(saved['step'], 6)
            self.assertEqual(saved['conditioning_architecture_transition'],
                             initial['conditioning_architecture_transition'])
            self.assertEqual(json.loads((run / 'run.json').read_text())['completed_steps'], 6)


if __name__ == '__main__':
    unittest.main()
