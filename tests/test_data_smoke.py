import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import soundfile as sf
import torch

from rmfsr.data_smoke import check_pipeline, data_recipe, require_prepared_data, validate_batch
from rmfsr.spectral import Spectral


class DataPreparationTests(unittest.TestCase):
    def test_cpu_worker_exercises_full_mode_augmentation_and_features(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifests = {}
            for split in ('train', 'validation'):
                audio = root / (split + '.wav')
                sf.write(audio, .1 * np.sin(np.arange(16000) * .1), 16000)
                manifest = root / (split + '.json')
                manifest.write_text(json.dumps([{'path': audio.name, 'frames': 16000}]))
                manifests[split + '_manifest'] = str(manifest)
            noise = root / 'noise'
            noise.mkdir()
            sf.write(noise / 'noise.flac', np.random.default_rng(1).normal(0, .01, 16000), 16000)
            ltas = root / 'ltas.npy'
            np.save(ltas, np.ones(161))
            cfg = dict(**manifests, pilot=False, sample_rate=16000, seconds=.2,
                       batch_size=2, seed=42, noise_dir=str(noise),
                       validation_noise_dir=str(noise), ltas_path=str(ltas),
                       augmentation_profile='figure2-v2', num_workers=1, threads=1)
            # Small fixture replaces corpus integrity only; actual worker loading,
            # full-mode augmentation, codecs and feature transforms still run.
            with patch('rmfsr.data_smoke.check', return_value={
                    'ready': True, 'problems': [], 'input_fingerprints': {'fixture': 'fixture'}}):
                result = check_pipeline(cfg, batches=2, readiness_output=root / 'readiness.json')
            self.assertTrue(result['ready'])
            self.assertEqual(result['training_examples'], 4)
            self.assertEqual(result['validation_examples'], 4)
            self.assertEqual(result['feature_shape'][:3], [2, 2, 161])

    def test_real_spectral_path_rejects_nonfinite_and_wrong_length_audio(self):
        cfg = {'batch_size': 2, 'sample_rate': 16000, 'seconds': .04}
        wave = torch.randn(2, 640)
        info = validate_batch(wave, wave, cfg, Spectral())
        self.assertEqual(info['waveform_shape'], [2, 640])
        with self.assertRaisesRegex(ValueError, 'Invalid input batch'):
            validate_batch(wave, wave[:, :-1], cfg, Spectral())
        broken = wave.clone()
        broken[0, 0] = float('nan')
        with self.assertRaisesRegex(ValueError, 'Nonfinite target'):
            validate_batch(broken, wave, cfg, Spectral())

    def test_gate_requires_complete_full_data_and_matching_fingerprints(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cfg = {'train_manifest': 'data/train.json', 'batch_size': 4}
            readiness = {'ready': True, 'pilot': False, 'data_recipe': data_recipe(cfg),
                         'input_fingerprints': {'train_manifest': 'actual-content-fingerprint'}}
            pipeline = {'ready': True, 'status': 'actual_data_pipeline_passed',
                        'data_recipe': data_recipe(cfg), 'input_fingerprints': readiness['input_fingerprints']}
            r, p, s = root / 'readiness.json', root / 'pipeline.json', root / 'status'
            def write():
                r.write_text(json.dumps(readiness))
                p.write_text(json.dumps(pipeline))
            s.write_text('ready')
            write()
            require_prepared_data(cfg, r, p, s)
            with self.assertRaisesRegex(ValueError, 'does not match'):
                require_prepared_data({**cfg, 'batch_size': 8}, r, p, s)
            readiness['pilot'] = True
            write()
            with self.assertRaisesRegex(ValueError, 'does not match'):
                require_prepared_data(cfg, r, p, s)
            readiness['pilot'] = False
            pipeline['input_fingerprints'] = {'train_manifest': 'another-dataset'}
            write()
            with self.assertRaisesRegex(ValueError, 'fingerprints differ'):
                require_prepared_data(cfg, r, p, s)
            s.write_text('preparing_corpora')
            with self.assertRaisesRegex(ValueError, 'has not completed'):
                require_prepared_data(cfg, r, p, s)


if __name__ == '__main__':
    unittest.main()
