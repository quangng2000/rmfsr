import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import soundfile as sf
import torch
from rmfsr.train import train, validation_selection_metric
from rmfsr.preflight import check


def fixture(root):
    paths = []
    for split, speaker in [('train', 'p001'), ('validation', 'p002')]:
        audio = root / (split + '.wav')
        sf.write(audio, np.sin(np.arange(16000) * .1) * .1, 16000)
        path = root / (split + '.json')
        path.write_text(json.dumps([dict(path=audio.name, speaker=speaker, split=split,
                                         frames=16000, seconds=1, sr=16000)]))
        paths.append(str(path))
    return dict(name='integration', pilot=True, train_manifest=paths[0], validation_manifest=paths[1],
                sample_rate=16000, seconds=.2, batch_size=1, channels=[8,8,16,16,16],
                steps=2, schedule_steps=20, seed=3, device='cpu', threads=2,
                indexed_batches=True, num_workers=0, noise_dir=None, ltas_path=None,
                ffmpeg=None, learning_rate=.0001, warmup_steps=2, ema_decay=.95,
                gradient_clip=1, save_every=1, validate_every=1, validation_examples=1,
                inference_evaluations_to_compare=[1], validation_audio_examples=0)


class TrainingIntegrationTests(unittest.TestCase):
    def test_best_checkpoint_uses_restoration_and_verifies_inputs_once(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); cfg = fixture(root)
            scores = [{'validation_dp_mse':.1, 'validation_restoration_mse_nfe1':.4},
                      {'validation_dp_mse':.3, 'validation_restoration_mse_nfe1':.2}]
            with patch('rmfsr.preflight.check', wraps=check) as preflight, \
                 patch('rmfsr.train.input_fingerprints', side_effect=AssertionError('checkpoint rehashed data')), \
                 patch('rmfsr.train.validate', side_effect=scores), contextlib.redirect_stdout(io.StringIO()):
                train(cfg, root / 'run')
            self.assertEqual(preflight.call_count, 1)
            checkpoint = torch.load(root/'run/best-validation.pt', weights_only=False)
            self.assertEqual(checkpoint['step'], 2)
            self.assertEqual(checkpoint['architecture']['decoder_layout'], 'mirror-v2')
            self.assertEqual(checkpoint['best_validation'], .2)
            self.assertEqual(checkpoint['validation_selection_metric'], 'validation_restoration_mse_nfe1')
            self.assertEqual(checkpoint['input_fingerprints']['integrity_version'], 'verified-content-v1')

    def test_old_fingerprint_checkpoint_requires_new_run(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); cfg = fixture(root)
            with patch('rmfsr.train.validate', return_value={'validation_restoration_mse_nfe1':.2}), contextlib.redirect_stdout(io.StringIO()):
                train(cfg, root/'run')
            path = root/'run/latest.pt'
            checkpoint = torch.load(path, weights_only=False)
            checkpoint['input_fingerprints'].pop('integrity_version')
            torch.save(checkpoint,path)
            with self.assertRaisesRegex(ValueError, 'predates verified-content'):
                train({**cfg, 'steps':3}, root/'run', path)

    def test_old_decoder_checkpoint_cannot_resume_new_training(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); cfg = fixture(root)
            with patch('rmfsr.train.validate', return_value={'validation_restoration_mse_nfe1':.2}), contextlib.redirect_stdout(io.StringIO()):
                train(cfg, root/'run')
            path = root/'run/latest.pt'
            checkpoint = torch.load(path, weights_only=False)
            checkpoint.pop('architecture')
            torch.save(checkpoint, path)
            with self.assertRaisesRegex(ValueError, 'decoder architecture; start a new run'):
                train({**cfg, 'steps':3}, root/'run', path)

    def test_benchmark_progress_cannot_override_normal_training(self):
        with tempfile.TemporaryDirectory() as temp:
            cfg = fixture(Path(temp))
            with self.assertRaisesRegex(ValueError, 'benchmark_only'):
                train({**cfg, 'benchmark_flow_progress':.9}, Path(temp)/'run')

    def test_selection_metric_must_have_inference_evaluation(self):
        with self.assertRaisesRegex(ValueError, 'configured inference'):
            validation_selection_metric({'inference_evaluations_to_compare':[1],
                                         'validation_selection_metric':'validation_restoration_mse_nfe5'})
