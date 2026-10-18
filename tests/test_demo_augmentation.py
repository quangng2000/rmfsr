"""Approved treatment coverage and continuation without changing held-out audio."""
import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from scipy import signal

from rmfsr.batches import batch_loader
from rmfsr.demo_augment import FAMILIES, apply_treatment, mp3_roundtrip
from rmfsr.train import (augmentation_resume_transition, backward_microbatches,
                        get_ffmpeg, train)
from tests.test_pipeline import fixture as pair_fixture
from tests.test_training_integration import fixture


class TreatmentTests(unittest.TestCase):
    def test_all_families_are_finite_reproducible_and_keep_source_untouched(self):
        t = np.arange(16000) / 16000
        clean = (.07 * signal.chirp(t, 130, 1, 3400) * np.sin(np.pi * t) ** 2).astype(np.float32)
        noise = np.random.default_rng(7).normal(size=len(clean))
        source = clean.copy()
        for family in FAMILIES:
            with self.subTest(family=family):
                left, info = apply_treatment(clean, noise, np.random.default_rng(17),
                                             ffmpeg=get_ffmpeg(None), family=family)
                right, _ = apply_treatment(clean, noise, np.random.default_rng(17),
                                           ffmpeg=get_ffmpeg(None), family=family)
                np.testing.assert_array_equal(left, right)
                np.testing.assert_array_equal(clean, source)
                self.assertEqual(left.shape, clean.shape)
                self.assertTrue(np.isfinite(left).all())
                self.assertLessEqual(float(np.max(abs(left))), .950001)
                self.assertGreater(float(np.std(left)), 1e-5)
                self.assertEqual(info['family'], family)

    def test_mp3_delay_compensation_keeps_nonperiodic_signal_aligned(self):
        t = np.arange(16000) / 16000
        x = .1 * signal.chirp(t, 200, 1, 3000) * np.sin(np.pi * t) ** 2
        y, delay = mp3_roundtrip(x, 16000, get_ffmpeg(None), 24)
        lags = signal.correlation_lags(len(x), len(y))
        self.assertEqual(int(lags[np.argmax(signal.correlate(y, x, method='fft'))]), 0)
        self.assertGreater(delay, 0)
        self.assertEqual(y.shape, x.shape)

    def test_indexed_resume_is_independent_of_prefetch_workers(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest, _ = pair_fixture(Path(directory))
            kwargs = dict(seconds=.2, sr=16000, pilot=True,
                          ffmpeg=get_ffmpeg(None), augmentation_profile='demo-eight-v1')
            cfg = dict(seed=7, batch_size=1, steps=4, device='cpu', num_workers=0)
            serial = list(batch_loader(manifest, kwargs, cfg, 0))
            resumed = list(batch_loader(manifest, kwargs, {**cfg, 'num_workers': 2}, 2))
            for a, b in zip(serial[2:], resumed):
                for left, right in zip(a, b):
                    torch.testing.assert_close(left, right, rtol=0, atol=0)


class MigrationTests(unittest.TestCase):
    def test_only_explicit_training_change_with_frozen_validation_is_allowed(self):
        old = dict(augmentation_profile='figure2-v3-aligned')
        new = dict(augmentation_profile='demo-eight-v1',
                   validation_augmentation_profile='figure2-v3-aligned',
                   allow_demo_eight_on_resume=True)
        self.assertFalse(augmentation_resume_transition(old, new)['best_validation_reset'])
        self.assertIsNone(augmentation_resume_transition(new, new))
        for bad in [{**new, 'allow_demo_eight_on_resume': False},
                    {**new, 'validation_augmentation_profile': 'demo-eight-v1'}]:
            with self.assertRaises(ValueError):
                augmentation_resume_transition(old, bad)
        with self.assertRaises(ValueError):
            augmentation_resume_transition(new, old)
        with self.assertRaises(ValueError):
            augmentation_resume_transition(new, {**new, 'validation_augmentation_profile': 'legacy'})

    def test_resume_preserves_state_validation_and_pre_change_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cfg = {**fixture(root), 'augmentation_profile': 'figure2-v3-aligned',
                   'input_verification': 'manifest'}
            run = root / 'run'
            validation_pairs = []

            def validation(ema, pairs, spectral, config, device, **kwargs):
                pairs.rng = np.random.default_rng(config['seed'] + 1000)
                validation_pairs.append(pairs.one())
                return {'validation_restoration_mse_nfe1': .2 if len(validation_pairs) <= 2 else .4}

            with patch('rmfsr.preflight.check', side_effect=AssertionError('Full scan forbidden')), \
                    patch('rmfsr.train.validate', side_effect=validation), \
                    contextlib.redirect_stdout(io.StringIO()):
                train(cfg, run)
            previous = torch.load(run / 'latest.pt', weights_only=False)
            previous_bytes = (run / 'latest.pt').read_bytes()
            best_bytes = (run / 'best-validation.pt').read_bytes()
            following = {**cfg, 'steps': 3, 'augmentation_profile': 'demo-eight-v1',
                         'validation_augmentation_profile': 'figure2-v3-aligned',
                         'allow_demo_eight_on_resume': True}

            def loaded_model(model, *args, **kwargs):
                for name, value in previous['model'].items():
                    torch.testing.assert_close(model.state_dict()[name], value, rtol=0, atol=0)
                return backward_microbatches(model, *args, **kwargs)

            with patch('rmfsr.preflight.check', side_effect=AssertionError('Full scan forbidden')), \
                    patch('rmfsr.train.validate', side_effect=validation), \
                    patch('rmfsr.train.backward_microbatches', side_effect=loaded_model), \
                    contextlib.redirect_stdout(io.StringIO()):
                manifest = train(following, run, run / 'latest.pt')
            current = torch.load(run / 'latest.pt', weights_only=False)
            self.assertEqual(current['step'], 3)
            self.assertEqual(current['best_validation'], .2)
            self.assertEqual((run / 'best-validation.pt').read_bytes(), best_bytes)
            self.assertEqual((run / 'latest-before-demo-eight-step-00000002.pt').read_bytes(), previous_bytes)
            self.assertFalse(manifest['augmentation_transition']['best_validation_reset'])
            self.assertEqual(current['history'][-1]['augmentation_profile'], 'demo-eight-v1')
            self.assertEqual(current['history'][-1]['validation_augmentation_profile'], 'figure2-v3-aligned')
            expected_lr = cfg['learning_rate'] * (.1 + .45 * (1 + np.cos(np.pi * 2 / cfg['schedule_steps'])))
            self.assertAlmostEqual(current['history'][-1]['lr'], expected_lr)
            for before, after in zip(validation_pairs[0][:3], validation_pairs[-1][:3]):
                np.testing.assert_array_equal(before, after)
            for key, state in previous['optimizer']['state'].items():
                self.assertEqual(current['optimizer']['state'][key]['step'], state['step'] + 1)
            for key in previous['ema']:
                expected = (previous['ema'][key] if key == 'embedding.frequencies' else
                            previous['ema'][key].lerp(current['model'][key], 1 - cfg['ema_decay']))
                torch.testing.assert_close(current['ema'][key], expected, rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
