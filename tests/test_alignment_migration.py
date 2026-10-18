"""FIR timing and checkpoint continuity across the explicitly approved fix."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from scipy import signal
import torch

from rmfsr.data import SpeechPairs
from rmfsr.continuous_train import run_sessions
from rmfsr.degradations import bandlimit
from rmfsr.train import alignment_resume_transition, backward_microbatches, train
from tests.test_pipeline import fixture
from tests.test_training_integration import fixture as training_fixture


class ChosenFilter:
    def __init__(self, kind, seed=8):
        self.kind = kind
        self.rng = np.random.default_rng(seed)

    def uniform(self, *args, **kwargs):
        return self.rng.uniform(*args, **kwargs)

    def integers(self, *args, **kwargs):
        return self.rng.integers(*args, **kwargs)

    def choice(self, options):
        self.rng.choice(options)  # Preserve the original number of RNG draws.
        return self.kind


class AlignmentTests(unittest.TestCase):
    def test_fir_removes_delay_and_preserves_both_crop_boundaries(self):
        taps = signal.firwin(65, [400, 3500], pass_zero=False, fs=16000)
        for length, position in [(1, 0), (17, 0), (129, 0), (129, 64), (129, 128)]:
            wave = np.zeros(length)
            wave[position] = 1
            with patch('rmfsr.degradations.signal.firwin', return_value=taps):
                actual = bandlimit(wave, ChosenFilter('fir'), 16000, align_fir=True)
            # Independently trim a full direct convolution by the known delay.
            expected = np.convolve(wave, taps, mode='full')[32:32 + length]
            np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-15)
            self.assertEqual(len(actual), length)
            self.assertEqual(int(np.argmax(abs(actual))), position)

    def test_old_recipe_keeps_exact_filter_output_and_rng_state(self):
        wave = np.random.default_rng(9).normal(size=257)
        rng = ChosenFilter('fir')
        lo, hi = rng.uniform(100, 800), rng.uniform(1500, .49 * 16000)
        rng.choice(['butter', 'cheby1', 'cheby2', 'ellip', 'bessel', 'fir'])
        expected = signal.lfilter(signal.firwin(65, [lo, hi], pass_zero=False, fs=16000), [1], wave)
        old_rng, new_rng = ChosenFilter('fir'), ChosenFilter('fir')
        np.testing.assert_array_equal(bandlimit(wave, old_rng, 16000), expected)
        bandlimit(wave, new_rng, 16000, align_fir=True)
        self.assertEqual(rng.rng.bit_generator.state, old_rng.rng.bit_generator.state)
        self.assertEqual(old_rng.rng.bit_generator.state, new_rng.rng.bit_generator.state)

    def test_other_filter_branches_are_unchanged(self):
        wave = np.random.default_rng(9).normal(size=257)
        for kind in ('butter', 'cheby1', 'cheby2', 'ellip', 'bessel'):
            expected = bandlimit(wave, ChosenFilter(kind), 16000)
            actual = bandlimit(wave, ChosenFilter(kind), 16000, align_fir=True)
            np.testing.assert_array_equal(actual, expected)

    def test_recipe_changes_only_damaged_fir_alignment(self):
        def fir_only(wave, rng, sr, *args, align_fir=False, **kwargs):
            wave = np.asarray(wave, np.float64)  # corrupt() uses this working precision.
            return (bandlimit(wave, ChosenFilter('fir'), sr, align_fir=align_fir).astype(np.float32),
                    np.zeros(len(wave), bool), ['bandlimit'])
        with tempfile.TemporaryDirectory() as directory:
            manifest, _ = fixture(Path(directory))
            with patch('rmfsr.data.corrupt', side_effect=fir_only):
                old = SpeechPairs(manifest, seconds=.2, pilot=True, seed=7,
                                  augmentation_profile='figure2-v2').one()
                new = SpeechPairs(manifest, seconds=.2, pilot=True, seed=7,
                                  augmentation_profile='figure2-v3-aligned').one()
            np.testing.assert_array_equal(old[0], new[0])
            np.testing.assert_array_equal(old[2], new[2])
            self.assertEqual(old[3], new[3])
            np.testing.assert_allclose(old[1][32:], new[1][:-32], rtol=1e-6, atol=1e-9)
            self.assertGreater(float(np.max(abs(new[1][-32:]))), 0.)


class AlignmentResumeTests(unittest.TestCase):
    def test_existing_controller_reads_queued_recipe_at_next_session(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / 'config.json'
            old = dict(steps=2, pilot=False, augmentation_profile='figure2-v2',
                       validate_every=1000, save_every=1000)
            config.write_text(json.dumps(old))
            run = root / 'run'
            observed = []

            def execute(command, **kwargs):
                saved = Path(command[command.index('--config') + 1])
                observed.append(json.loads(saved.read_text()))
                step = len(observed)
                (run / 'latest.pt').write_bytes(b'checkpoint fixture')
                (run / 'run.json').write_text(json.dumps(dict(completed_steps=step,
                    status='paused_checkpointed' if step == 1 else 'training_complete_not_evaluated')))
                if step == 1:
                    following = {**old, 'augmentation_profile': 'figure2-v3-aligned',
                                 'allow_fir_alignment_fix_on_resume': True, 'validate_every': 5000}
                    config.write_text(json.dumps(following))
                    saved.write_text(json.dumps(following))
                else:
                    self.assertIn('--resume', command)

            run_sessions(config, run, root / 'control', execute)
            self.assertEqual([item['augmentation_profile'] for item in observed],
                             ['figure2-v2', 'figure2-v3-aligned'])
            self.assertEqual(observed[1]['validate_every'], 5000)
            self.assertEqual(observed[1]['save_every'], 1000)

    def test_only_explicit_forward_migration_is_allowed(self):
        old = {'augmentation_profile': 'figure2-v2'}
        new = {'augmentation_profile': 'figure2-v3-aligned'}
        with self.assertRaisesRegex(ValueError, 'explicit FIR'):
            alignment_resume_transition(old, new)
        approved = {**new, 'allow_fir_alignment_fix_on_resume': True}
        self.assertTrue(alignment_resume_transition(old, approved)['best_validation_reset'])
        self.assertIsNone(alignment_resume_transition(approved, approved))
        for previous, following in [
                ({}, approved), (approved, {**old, 'allow_fir_alignment_fix_on_resume': True}),
                (old, {**new, 'allow_fir_alignment_fix_on_resume': 'true'})]:
            with self.assertRaisesRegex(ValueError, 'explicit FIR'):
                alignment_resume_transition(previous, following)

    def test_resume_retains_weights_ema_optimizer_steps_and_archives_old_best(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cfg = {**training_fixture(root), 'augmentation_profile': 'figure2-v2'}
            run = root / 'run'
            with patch('rmfsr.train.validate', return_value={'validation_restoration_mse_nfe1': .2}), \
                    contextlib.redirect_stdout(io.StringIO()):
                train(cfg, run)
            previous = torch.load(run / 'latest.pt', weights_only=False)
            previous_best = (run / 'best-validation.pt').read_bytes()
            expected_lr = cfg['learning_rate'] * (.1 + .45 * (1 + np.cos(np.pi * 2 / cfg['schedule_steps'])))

            def inspect_loaded_model(model, *args, **kwargs):
                for key, value in previous['model'].items():
                    torch.testing.assert_close(model.state_dict()[key], value, rtol=0, atol=0)
                return backward_microbatches(model, *args, **kwargs)

            following = {**cfg, 'steps': 3, 'augmentation_profile': 'figure2-v3-aligned',
                         'allow_fir_alignment_fix_on_resume': True}
            with patch('rmfsr.train.backward_microbatches', side_effect=inspect_loaded_model), \
                    patch('rmfsr.train.validate', return_value={'validation_restoration_mse_nfe1': .4}), \
                    contextlib.redirect_stdout(io.StringIO()):
                manifest = train(following, run, run / 'latest.pt')
            current = torch.load(run / 'latest.pt', weights_only=False)
            self.assertEqual(current['step'], 3)
            self.assertAlmostEqual(current['history'][-1]['lr'], expected_lr)
            self.assertEqual(current['config']['augmentation_profile'], 'figure2-v3-aligned')
            self.assertEqual(current['history'][-1]['augmentation_profile'], 'figure2-v3-aligned')
            self.assertEqual(manifest['augmentation_transition']['after_step'], 2)
            self.assertEqual(current['best_validation'], .4)  # Old score .2 is a different recipe.
            self.assertEqual((run / 'best-validation-before-alignment-step-00000002.pt').read_bytes(), previous_best)
            self.assertEqual(torch.load(run / 'best-validation.pt', weights_only=False)['step'], 3)
            for key, state in previous['optimizer']['state'].items():
                self.assertEqual(current['optimizer']['state'][key]['step'], state['step'] + 1)
            for key in previous['ema']:
                if key == 'embedding.frequencies':
                    torch.testing.assert_close(current['ema'][key], previous['ema'][key], rtol=0, atol=0)
                else:
                    expected = previous['ema'][key].lerp(current['model'][key], 1 - cfg['ema_decay'])
                    torch.testing.assert_close(current['ema'][key], expected, rtol=0, atol=0)

            # A later ordinary resume must retain the new recipe's best threshold.
            with patch('rmfsr.train.validate', return_value={'validation_restoration_mse_nfe1': .5}), \
                    contextlib.redirect_stdout(io.StringIO()):
                train({**following, 'steps': 4}, run, run / 'latest.pt')
            self.assertEqual(torch.load(run / 'latest.pt', weights_only=False)['best_validation'], .4)
            self.assertEqual(len(list(run.glob('augmentation-transition-*.json'))), 1)


if __name__ == '__main__':
    unittest.main()
