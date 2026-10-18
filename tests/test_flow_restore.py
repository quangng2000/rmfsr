"""A requested clock restoration must retain training state across sessions."""
import contextlib
import copy
import io
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import torch

from rmfsr.flow import schedule_values
from rmfsr.flow_curriculum import config_digest, resolve_transition, training_flow_progress
from rmfsr.train import backward_microbatches, train
from tests.test_flow_curriculum import request_for
from tests.test_training_integration import fixture


def restore_for(config):
    return dict(version=1, mode='original-gradual-clock',
                expected_config_sha256=config_digest(config))


class FlowRestoreTests(unittest.TestCase):
    def test_restores_original_clock_and_preserves_acceleration_history(self):
        cfg = dict(steps=300000, schedule_steps=300000, flow_schedule='figure1-cosine-approx',
                   speech_fidelity=dict(pesq_weight=.001, sisdr_weight=.0001))
        ramp = resolve_transition(cfg, cfg, 10969, request_for(cfg))
        previous = {**cfg, 'flow_schedule_transition': ramp}
        original = copy.deepcopy(previous)
        restored = resolve_transition(cfg, previous, 18400, restore_request=restore_for(cfg))
        self.assertEqual(previous, original)
        for key in ramp:
            self.assertEqual(restored[key], ramp[key])
        active = {**cfg, 'flow_schedule_transition': restored}
        for step in (18400, 50000, 150000, 240000, 300000):
            self.assertEqual(training_flow_progress(step, active), step / 300000)
        diagonal, gamma = schedule_values(training_flow_progress(18400, active), cfg['flow_schedule'])
        self.assertAlmostEqual(diagonal, .9707, delta=.0001)
        self.assertAlmostEqual(gamma, .0637, delta=.0001)
        self.assertEqual(resolve_transition(cfg, active, 19000), restored)
        self.assertEqual(resolve_transition(cfg, active, 19000, request_for(cfg),
                                           restore_request=restore_for(cfg)), restored)

    def test_restore_rejects_mismatched_recipe_and_cannot_be_reset(self):
        cfg = dict(steps=300000, flow_schedule='figure1-cosine-approx')
        ramp = resolve_transition(cfg, cfg, 10000, request_for(cfg))
        previous = {**cfg, 'flow_schedule_transition': ramp}
        restore = restore_for(cfg)
        with self.assertRaisesRegex(ValueError, 'accelerated checkpoint'):
            resolve_transition(cfg, cfg, 18000, restore_request=restore)
        with self.assertRaisesRegex(ValueError, 'different configuration'):
            resolve_transition({**cfg, 'learning_rate': .1}, previous, 18000,
                               restore_request=restore, allowed_config_changes=('learning_rate',))
        with self.assertRaisesRegex(ValueError, 'different configuration'):
            resolve_transition(cfg, previous, 18000,
                               restore_request={**restore, 'expected_config_sha256': '0'*64})
        restored = resolve_transition(cfg, previous, 18000, restore_request=restore)
        active = {**cfg, 'flow_schedule_transition': restored}
        with self.assertRaisesRegex(ValueError, 'differs from checkpoint'):
            resolve_transition(cfg, active, 19000,
                               restore_request={**restore, 'expected_config_sha256': '0'*64})
        with self.assertRaisesRegex(ValueError, 'precedes flow schedule restoration'):
            training_flow_progress(17999, active)
        with self.assertRaisesRegex(ValueError, 'precedes flow schedule restoration'):
            resolve_transition(cfg, active, 17999)

    def test_real_resume_retains_state_and_future_augmentation_migration(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cfg = {**fixture(root), 'steps': 6, 'max_wall_seconds': .0001,
                   'flow_schedule': 'figure1-cosine-approx', 'input_verification': 'manifest',
                   'augmentation_profile': 'demo-eight-v1',
                   'validation_augmentation_profile': 'figure2-v3-aligned'}
            with patch('rmfsr.preflight.check', side_effect=AssertionError('Full scan forbidden')), \
                    patch('rmfsr.train.validate', return_value={'validation_restoration_mse_nfe1': .2}), \
                    contextlib.redirect_stdout(io.StringIO()):
                seed = root/'seed'
                train(cfg, seed)
                (seed/'flow-schedule-transition.json').write_text(json.dumps(request_for(cfg, ramp=3)))
                train(cfg, seed, seed/'latest.pt')
                previous = torch.load(seed/'latest.pt', weights_only=False)
                self.assertEqual(previous['step'], 2)
                before = (seed/'latest.pt').read_bytes()
                (seed/'flow-schedule-restore.json').write_text(json.dumps(restore_for(cfg)))
                for name in ('straight', 'split'):
                    shutil.copytree(seed, root/name)

                def loaded_model(model, *args, **kwargs):
                    for key, value in previous['model'].items():
                        torch.testing.assert_close(model.state_dict()[key], value, rtol=0, atol=0)
                    return backward_microbatches(model, *args, **kwargs)

                with patch('rmfsr.train.backward_microbatches', side_effect=loaded_model):
                    train(cfg, root/'split', root/'split/latest.pt')
                first = torch.load(root/'split/latest.pt', weights_only=False)
                self.assertEqual(first['step'], 3)
                self.assertEqual(first['history'][-1]['flow_progress'], 2/20)
                for key, state in previous['optimizer']['state'].items():
                    self.assertEqual(first['optimizer']['state'][key]['step'], state['step'] + 1)
                for key in previous['ema']:
                    expected = (previous['ema'][key] if key == 'embedding.frequencies' else
                                previous['ema'][key].lerp(first['model'][key], 1-cfg['ema_decay']))
                    torch.testing.assert_close(first['ema'][key], expected, rtol=0, atol=0)
                train(cfg, root/'straight', root/'straight/latest.pt')
                new = {**cfg, 'augmentation_profile': 'figure2-v3-aligned',
                       'allow_paper_data_on_resume': True}
                for _ in range(3):
                    train(new, root/'split', root/'split/latest.pt')
                with patch('rmfsr.train.time.perf_counter', return_value=0.):
                    train(new, root/'straight', root/'straight/latest.pt')
            a = torch.load(root/'straight/latest.pt', weights_only=False)
            b = torch.load(root/'split/latest.pt', weights_only=False)
            self.assertEqual(a['step'], b['step'])
            self.assertEqual(b['step'], 6)
            self.assertEqual(a['history'][-1]['loss'], b['history'][-1]['loss'])
            self.assertEqual(b['history'][-1]['flow_progress'], 5/20)
            self.assertEqual(b['history'][-1]['learning_rate_progress'], 5/20)
            self.assertEqual(a['config']['flow_schedule_transition'], b['config']['flow_schedule_transition'])
            self.assertEqual(b['config']['flow_schedule_transition']['restoration']['after_step'], 2)
            for field in ('model', 'ema'):
                for key in a[field]:
                    torch.testing.assert_close(a[field][key], b[field][key], rtol=0, atol=0)
            for key, state in a['optimizer']['state'].items():
                for name, value in state.items():
                    torch.testing.assert_close(value, b['optimizer']['state'][key][name], rtol=0, atol=0)
            torch.testing.assert_close(a['torch_rng'], b['torch_rng'], rtol=0, atol=0)
            self.assertEqual(a['data_rng'], b['data_rng'])
            self.assertEqual((root/'split/before-flow-restore-step-00000002.pt').read_bytes(), before)
            self.assertEqual(json.loads((root/'split/flow-schedule-active.json').read_text()),
                             b['config']['flow_schedule_transition'])


if __name__ == '__main__':
    unittest.main()
