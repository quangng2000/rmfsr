import copy
import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import soundfile as sf
import torch

from rmfsr.validation import validate_restoration


class IdentitySpectral:
    sr = 16000

    def encode(self, wave):
        wave = wave.reshape(1, -1)
        return torch.stack([wave, torch.zeros_like(wave)], 1).unsqueeze(2)

    def decode(self, spectrum, length):
        return spectrum[:, 0, 0, :length]


class FixedPairs:
    def __init__(self, clean=None, damaged=None, mask=None, randomize=False):
        self.clean = np.ones(64, dtype=np.float32) if clean is None else clean
        self.damaged = np.zeros_like(self.clean) if damaged is None else damaged
        self.mask = np.ones_like(self.clean, dtype=bool) if mask is None else mask
        self.randomize = randomize
        self.rng = np.random.default_rng(123)

    def one(self):
        clean = self.clean.copy()
        damaged = self.damaged.copy()
        if self.randomize:
            clean *= self.rng.uniform(.5, 1.)
        return clean, damaged, self.mask.copy(), ['packet-gaps']


class EchoIntermediate(torch.nn.Module):
    def forward(self, x, y, t, r, cache=None):
        return x


class EchoDamaged(torch.nn.Module):
    def forward(self, x, y, t, r, cache=None):
        return y


class BadOutput(torch.nn.Module):
    def forward(self, x, y, t, r, cache=None):
        return torch.full_like(x, float('nan'))


class ValidationTests(unittest.TestCase):
    def setUp(self):
        self.spec = IdentitySpectral()
        self.cfg = dict(seed=17, sample_rate=16000, validation_examples=2,
                        validation_chunk_frames=10, inference_evaluations_to_compare=[1, 2, 5])

    def validate(self, model, pairs=None, cfg=None, **kwargs):
        return validate_restoration(model, pairs or FixedPairs(), self.spec,
                                    self.cfg if cfg is None else cfg, torch.device('cpu'), **kwargs)

    def test_teacher_input_diagnostic_can_favor_worse_restoration(self):
        intermediate = self.validate(EchoIntermediate())
        damaged = self.validate(EchoDamaged())
        self.assertLess(intermediate['validation_dp_mse'], damaged['validation_dp_mse'])
        for nfe in (1, 2, 5):
            self.assertGreater(intermediate[f'validation_restoration_mse_nfe{nfe}'],
                               damaged[f'validation_restoration_mse_nfe{nfe}'])

    def test_fixed_examples_do_not_consume_training_or_validation_rng(self):
        torch.manual_seed(61)
        np.random.seed(62)
        random.seed(63)
        pairs = FixedPairs(randomize=True)
        model = EchoIntermediate().train()
        state = (torch.get_rng_state().clone(), np.random.get_state(), random.getstate(),
                 copy.deepcopy(pairs.rng.bit_generator.state))
        first = self.validate(model, pairs)
        second = self.validate(model, pairs)
        self.assertEqual(first, second)
        torch.testing.assert_close(state[0], torch.get_rng_state(), rtol=0, atol=0)
        np.testing.assert_equal(state[1], np.random.get_state())
        self.assertEqual(state[2], random.getstate())
        self.assertEqual(state[3], pairs.rng.bit_generator.state)
        self.assertTrue(model.training)

    def test_same_noise_and_only_damaged_input_for_each_nfe(self):
        captures = []
        def capture(model, y, steps, noise, chunk_frames):
            captures.append((steps, y.clone(), noise.clone(), chunk_frames))
            return y
        with patch('rmfsr.validation.sample', side_effect=capture):
            self.validate(EchoDamaged())
        self.assertEqual([item[0] for item in captures], [1, 2, 5, 1, 2, 5])
        for offset in (0, 3):
            for item in captures[offset:offset + 3]:
                torch.testing.assert_close(item[1], torch.zeros_like(item[1]), rtol=0, atol=0)
                torch.testing.assert_close(item[2], captures[offset][2], rtol=0, atol=0)
                self.assertEqual(item[3], 10)
        self.assertFalse(torch.equal(captures[0][2], captures[3][2]))

    def test_gap_and_intact_errors_are_sample_weighted_with_empty_masks_explicit(self):
        clean = np.ones(300, np.float32)
        mask = np.arange(300) < 100
        damaged = np.where(mask, 0., .5).astype(np.float32)
        metrics = self.validate(EchoDamaged(), FixedPairs(clean, damaged, mask))
        self.assertEqual(metrics['validation_gap_samples'], 200)
        self.assertEqual(metrics['validation_intact_samples'], 400)
        for nfe in (1, 2, 5):
            self.assertEqual(metrics[f'validation_gap_mse_nfe{nfe}'], 1.)
            self.assertEqual(metrics[f'validation_intact_mse_nfe{nfe}'], .25)
            self.assertEqual(metrics[f'validation_restoration_mse_nfe{nfe}'], .5)
            self.assertIsNone(metrics[f'validation_stoi_nfe{nfe}'])
        no_gaps = self.validate(EchoDamaged(), FixedPairs(mask=np.zeros(64, bool)))
        self.assertEqual(no_gaps['validation_gap_samples'], 0)
        self.assertIsNone(no_gaps['validation_gap_mse_nfe5'])
        all_gaps = self.validate(EchoDamaged())
        self.assertIsNone(all_gaps['validation_intact_mse_nfe5'])

    def test_review_audio_is_capped_and_uses_one_gain(self):
        cfg = {**self.cfg, 'validation_examples': 3, 'validation_audio_examples': 2,
               'inference_evaluations_to_compare': [1, 5]}
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            metrics = self.validate(EchoDamaged(), cfg=cfg, output_dir=root, save_audio=True)
            report = json.loads((root / 'report.json').read_text())
            self.assertEqual(report['metrics'], metrics)
            self.assertEqual(report['audio_examples'], 2)
            self.assertEqual(len(list(root.rglob('*.wav'))), 8)
            self.assertEqual(report['playback_gain'], .95)
            wave, sr = sf.read(root / 'example-000/clean.wav')
            self.assertEqual(sr, 16000)
            np.testing.assert_allclose(wave, .95, atol=1 / 32768)
            other = root / 'no-audio'
            self.validate(EchoDamaged(), cfg=cfg, output_dir=other)
            self.assertTrue((other / 'report.json').is_file())
            self.assertFalse(list(other.rglob('*.wav')))

    def test_nonfinite_restore_and_invalid_configuration_reject(self):
        with self.assertRaises(FloatingPointError):
            self.validate(BadOutput())
        with patch('rmfsr.validation.sample', return_value=torch.full((1, 2, 1, 64), float('nan'))):
            with self.assertRaises(FloatingPointError):
                self.validate(EchoDamaged())
        for key, value in [('validation_examples', 0), ('validation_examples', True),
                           ('inference_evaluations_to_compare', []),
                           ('inference_evaluations_to_compare', [0]),
                           ('inference_evaluations_to_compare', [1, 1]),
                           ('inference_evaluations_to_compare', [1.5]),
                           ('validation_chunk_frames', -1)]:
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                self.validate(EchoDamaged(), cfg={**self.cfg, key: value})
        with self.assertRaises(ValueError):
            self.validate(EchoDamaged(), FixedPairs(clean=np.empty(0, np.float32)))
        with self.assertRaises(ValueError):
            self.validate(EchoDamaged(), save_audio=True)

    def test_stoi_reports_valid_long_clip_and_skips_silence(self):
        clean = np.sin(np.arange(16000) * .1).astype(np.float32)
        cfg = {**self.cfg, 'validation_examples': 1, 'validation_chunk_frames': 0,
               'inference_evaluations_to_compare': [1]}
        metrics = self.validate(EchoDamaged(), FixedPairs(clean, clean, np.zeros(16000, bool)), cfg)
        self.assertEqual(metrics['validation_stoi_examples_nfe1'], 1)
        self.assertAlmostEqual(metrics['validation_stoi_nfe1'], 1., places=6)
        silent = np.zeros(16000, np.float32)
        metrics = self.validate(EchoDamaged(), FixedPairs(silent, silent, np.zeros(16000, bool)), cfg)
        self.assertEqual(metrics['validation_stoi_examples_nfe1'], 0)
        self.assertIsNone(metrics['validation_stoi_nfe1'])


if __name__ == '__main__':
    unittest.main()
