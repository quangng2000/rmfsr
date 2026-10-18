import copy
from pathlib import Path
import pickle
import random
import tempfile
import unittest
import numpy as np
import torch
from rmfsr.conditioning_migration import convert_checkpoint, convert_payload, validate_conditioning_transition
from rmfsr.conditioning_migration_io import file_sha256
from rmfsr.flow_curriculum import config_digest, resolve_transition, training_flow_progress
from rmfsr.frequency_transition import tensor_sha256
from rmfsr.model import model_from_config, model_from_checkpoint


def fixture(directory):
    config = dict(channels=[4], encoder_dilations=[1], tcn_dilations=[1], steps=300000,
                  schedule_steps=300000, flow_schedule='figure1-cosine-approx')
    transition = dict(version=1, expected_config_sha256=config_digest(config), ramp_updates=500, target_progress=.8)
    config['flow_schedule_transition'] = resolve_transition(config, config, 100, transition)
    model = model_from_config(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.0002, betas=(.8, .95), weight_decay=.03)
    sum(parameter.square().sum() for parameter in model.parameters()).backward()
    optimizer.step()
    checkpoint = dict(model=copy.deepcopy(model.state_dict()), ema=copy.deepcopy(model.state_dict()),
        architecture=model.architecture(), optimizer=optimizer.state_dict(), step=200, config=config,
        torch_rng=torch.get_rng_state(), numpy_rng=np.random.get_state(), python_rng=random.getstate(),
        data_rng=np.random.default_rng(5).bit_generator.state, input_fingerprints={'dataset': 'same'},
        history=[{'step': 200, 'loss': .01}], best_validation=.02, validation_selection_metric='mse')
    checkpoint['frequency_width_transition'] = dict(nominal_frequency_width=4, request=dict(
        target_frequency_sha256={kind: tensor_sha256(checkpoint[kind]['embedding.frequencies']) for kind in ('model', 'ema')}))
    source = Path(directory) / 'source.pt'
    torch.save(checkpoint, source)
    request = dict(version=1, transition_id='approved-scm', expected_source_sha256=file_sha256(source),
                   expected_config_sha256=config_digest(config), expected_step=200)
    return source, checkpoint, request


class ConditioningMigrationTests(unittest.TestCase):
    def test_preserves_training_position_weights_rng_and_group_policy(self):
        with tempfile.TemporaryDirectory() as directory:
            source, original, request = fixture(directory)
            rng = torch.get_rng_state().clone()
            destination = Path(directory) / 'converted.pt'
            marker = convert_checkpoint(source, destination, request)
            actual = torch.load(destination, weights_only=False)
            self.assertEqual(file_sha256(source), request['expected_source_sha256'])
            self.assertTrue(torch.equal(torch.get_rng_state(), rng))
            for key in ('step', 'history', 'data_rng', 'input_fingerprints', 'numpy_rng', 'python_rng'):
                self.assertEqual(pickle.dumps(actual[key]), pickle.dumps(original[key]))
            self.assertTrue(torch.equal(actual['torch_rng'], original['torch_rng']))
            for kind in ('model', 'ema'):
                for key, value in original[kind].items():
                    if key != 'embedding.frequencies':
                        self.assertTrue(torch.equal(value, actual[kind][key]), key)
            for key, value in original['optimizer']['param_groups'][0].items():
                if key != 'params':
                    self.assertEqual(value, actual['optimizer']['param_groups'][0][key])
            self.assertEqual(actual['optimizer']['state'], {})
            self.assertEqual(actual['best_validation'], float('inf'))
            self.assertNotIn('frequency_width_transition', actual)
            self.assertEqual(marker['source_frequency_transition'], original['frequency_width_transition'])
            self.assertEqual(training_flow_progress(200, actual['config']), training_flow_progress(200, original['config']))
            self.assertEqual(len(actual['config']['flow_schedule_transition']['config_migrations']), 1)
            model_from_checkpoint(actual).load_state_dict(actual['model'], strict=True)
            self.assertFalse(list(Path(directory).glob('*.partial')))

    def test_source_guards_fail_without_output(self):
        with tempfile.TemporaryDirectory() as directory:
            source, original, request = fixture(directory)
            destination = Path(directory) / 'converted.pt'
            for changed in ({'expected_source_sha256': '0' * 64}, {'expected_step': 201},
                            {'expected_config_sha256': '0' * 64}, {'version': True}):
                with self.assertRaises(ValueError):
                    convert_checkpoint(source, destination, {**request, **changed})
                self.assertFalse(destination.exists())
            with self.assertRaises(ValueError):
                convert_checkpoint(source, source, request)
            convert_checkpoint(source, destination, request)
            with self.assertRaises(ValueError):
                convert_checkpoint(source, destination, request)

    def test_normal_resume_validates_marker_without_resetting_optimizer(self):
        with tempfile.TemporaryDirectory() as directory:
            source, original, request = fixture(directory)
            actual = convert_payload(original, request, source)
            actual['step'] += 5
            actual['optimizer']['state'] = {0: {'step': torch.tensor(5.)}}
            self.assertEqual(validate_conditioning_transition(actual, actual['config'])['after_step'], 200)
            with self.assertRaises(ValueError):
                validate_conditioning_transition(actual, {**actual['config'], 'steps': 400000})
            with self.assertRaises(ValueError):
                validate_conditioning_transition({**actual, 'step': 199}, actual['config'])
            with self.assertRaises(ValueError):
                convert_payload(actual, {**request, 'expected_step': 205}, source)
