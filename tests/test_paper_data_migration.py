"""Session-boundary data changes must preserve the already activated flow clock."""
import contextlib
import copy
import io
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from rmfsr.continuous_train import run_sessions
from rmfsr.flow_curriculum import resolve_transition, training_flow_progress
from rmfsr.train import augmentation_resume_transition, backward_microbatches, train
from tests.test_flow_curriculum import request_for
from tests.test_training_integration import fixture


class PaperDataMigrationTests(unittest.TestCase):
    def test_requires_explicit_migration_and_identical_validation(self):
        old = dict(augmentation_profile='demo-eight-v1',
                   validation_augmentation_profile='figure2-v3-aligned')
        new = {**old, 'augmentation_profile': 'figure2-v3-aligned',
               'allow_paper_data_on_resume': True}
        transition = augmentation_resume_transition(old, new)
        self.assertFalse(transition['best_validation_reset'])
        self.assertIsNone(augmentation_resume_transition(new, new))
        for flag in (None, False, 'true', 1):
            with self.assertRaises(ValueError):
                augmentation_resume_transition(old, {**new, 'allow_paper_data_on_resume': flag})
        with self.assertRaises(ValueError):
            augmentation_resume_transition(old, {**new, 'validation_augmentation_profile': 'demo-eight-v1'})

    def test_existing_controller_reads_new_recipe_only_on_following_session(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            old = dict(steps=2, pilot=False, augmentation_profile='demo-eight-v1',
                       validation_augmentation_profile='figure2-v3-aligned',
                       save_every=1000, validate_every=5000)
            config = root/'config.json'
            config.write_text(json.dumps(old))
            run = root/'run'
            observed = []

            def execute(command, **kwargs):
                path = Path(command[command.index('--config') + 1])
                in_memory = json.loads(path.read_text())
                observed.append(in_memory)
                step = len(observed)
                if step == 1:
                    queued = {**old, 'augmentation_profile': 'figure2-v3-aligned',
                              'allow_paper_data_on_resume': True}
                    for p in (config, path):
                        p.write_text(json.dumps(queued))
                    self.assertEqual(in_memory['augmentation_profile'], 'demo-eight-v1')
                else:
                    self.assertIn('--resume', command)
                (run/'latest.pt').write_bytes(b'controller fixture')
                (run/'run.json').write_text(json.dumps(dict(completed_steps=step,
                    status='paused_checkpointed' if step == 1 else 'training_complete_not_evaluated')))

            run_sessions(config, run, root/'control', execute)
            self.assertEqual([c['augmentation_profile'] for c in observed],
                             ['demo-eight-v1', 'figure2-v3-aligned'])
            self.assertEqual(observed[1]['save_every'], 1000)
            self.assertEqual(observed[1]['validate_every'], 5000)

    def test_flow_rebinding_cannot_change_other_fields_or_restart_ramp(self):
        old = dict(steps=300000, flow_schedule='figure1-cosine-approx',
                   augmentation_profile='demo-eight-v1',
                   validation_augmentation_profile='figure2-v3-aligned')
        request = request_for(old)
        policy = resolve_transition(old, old, 10969, request)
        previous = {**old, 'flow_schedule_transition': policy}
        original = copy.deepcopy(previous)
        new = {**old, 'augmentation_profile': 'figure2-v3-aligned',
               'allow_paper_data_on_resume': True}
        allowed = ('augmentation_profile', 'allow_paper_data_on_resume')
        with self.assertRaises(ValueError):
            resolve_transition(new, previous, 15000, request)
        migrated = resolve_transition(new, previous, 15000, request, allowed_config_changes=allowed)
        self.assertEqual(previous, original)
        for key in ('start_step', 'start_progress', 'ramp_updates', 'target_progress', 'request'):
            self.assertEqual(policy[key], migrated[key])
        for step in (15000, 20969, 300000):
            self.assertEqual(training_flow_progress(step, previous),
                             training_flow_progress(step, {**new, 'flow_schedule_transition': migrated}))
        self.assertEqual(resolve_transition(new, {**new, 'flow_schedule_transition': migrated},
                                            20000, request), migrated)
        for key, value in [('steps', 300001), ('learning_rate', .1),
                           ('validation_augmentation_profile', 'demo-eight-v1'), ('groups', 16)]:
            with self.assertRaises(ValueError):
                resolve_transition({**new, key: value}, previous, 15000, request,
                                   allowed_config_changes=allowed)

    def test_actual_resume_preserves_state_and_later_resume_is_exact(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cfg = {**fixture(root), 'steps': 6, 'max_wall_seconds': .0001,
                   'flow_schedule': 'figure1-cosine-approx',
                   'augmentation_profile': 'demo-eight-v1',
                   'validation_augmentation_profile': 'figure2-v3-aligned',
                   'input_verification': 'manifest'}
            seed = root/'seed'
            validation_pairs = []

            def validate(ema, pairs, spectral, config, device, **kwargs):
                pairs.rng = np.random.default_rng(config['seed'] + 1000)
                validation_pairs.append(pairs.one())
                return {'validation_restoration_mse_nfe1':
                        .2 if config['augmentation_profile'] == 'demo-eight-v1' else .4}

            with patch('rmfsr.preflight.check', side_effect=AssertionError('Full corpus scan forbidden')), \
                    patch('rmfsr.train.validate', side_effect=validate), \
                    contextlib.redirect_stdout(io.StringIO()):
                train(cfg, seed)
                request = request_for(cfg, ramp=3)
                (seed/'flow-schedule-transition.json').write_text(json.dumps(request))
                train(cfg, seed, seed/'latest.pt')
                previous = torch.load(seed/'latest.pt', weights_only=False)
                self.assertEqual(previous['step'], 2)
                latest_bytes = (seed/'latest.pt').read_bytes()
                best_bytes = (seed/'best-validation.pt').read_bytes()
                for name in ('straight', 'split'):
                    shutil.copytree(seed, root/name)
                new = {**cfg, 'augmentation_profile': 'figure2-v3-aligned',
                       'allow_paper_data_on_resume': True}

                def loaded_model(model, *args, **kwargs):
                    for key, tensor in previous['model'].items():
                        torch.testing.assert_close(model.state_dict()[key], tensor, rtol=0, atol=0)
                    return backward_microbatches(model, *args, **kwargs)

                with patch('rmfsr.train.backward_microbatches', side_effect=loaded_model):
                    manifest = train(new, root/'split', root/'split/latest.pt')
                first = torch.load(root/'split/latest.pt', weights_only=False)
                self.assertEqual(first['step'], 3)
                self.assertEqual(first['best_validation'], .2)
                for key, state in previous['optimizer']['state'].items():
                    self.assertEqual(first['optimizer']['state'][key]['step'], state['step'] + 1)
                for key in previous['ema']:
                    expected = (previous['ema'][key] if key == 'embedding.frequencies' else
                                previous['ema'][key].lerp(first['model'][key], 1-cfg['ema_decay']))
                    torch.testing.assert_close(first['ema'][key], expected, rtol=0, atol=0)
                self.assertFalse(manifest['augmentation_transition']['best_validation_reset'])
                for _ in range(3):
                    train(new, root/'split', root/'split/latest.pt')
                with patch('rmfsr.train.time.perf_counter', return_value=0.):
                    train(new, root/'straight', root/'straight/latest.pt')

            a = torch.load(root/'straight/latest.pt', weights_only=False)
            b = torch.load(root/'split/latest.pt', weights_only=False)
            self.assertEqual(a['step'], 6)
            self.assertEqual(b['step'], 6)
            self.assertEqual(a['history'][-1]['loss'], b['history'][-1]['loss'])
            self.assertEqual(a['history'][-1]['lr'], b['history'][-1]['lr'])
            self.assertEqual(b['history'][-1]['flow_progress'], .8)
            self.assertEqual(b['config']['flow_schedule_transition']['start_step'], 1)
            for field in ('model', 'ema'):
                for key in a[field]:
                    torch.testing.assert_close(a[field][key], b[field][key], rtol=0, atol=0)
            for key, state in a['optimizer']['state'].items():
                for name, value in state.items():
                    torch.testing.assert_close(value, b['optimizer']['state'][key][name], rtol=0, atol=0)
            self.assertEqual((root/'split/latest-before-figure2-step-00000002.pt').read_bytes(), latest_bytes)
            self.assertEqual((root/'split/best-validation-before-figure2-step-00000002.pt').read_bytes(), best_bytes)
            self.assertEqual((root/'split/best-validation.pt').read_bytes(), best_bytes)
            self.assertEqual(len(list((root/'split').glob('augmentation-transition-*.json'))), 1)
            for before, after in zip(validation_pairs[0][:3], validation_pairs[-1][:3]):
                np.testing.assert_array_equal(before, after)


if __name__ == '__main__':
    unittest.main()
