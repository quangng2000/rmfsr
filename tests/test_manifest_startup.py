import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from rmfsr.paths import indexed_noise_paths
from rmfsr.preflight import startup_check
from rmfsr.train import train
from test_training_integration import fixture


class ManifestStartupTests(unittest.TestCase):
    def test_real_updates_and_resume_without_full_audio_scan(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cfg = {**fixture(root), 'input_verification': 'manifest',
                   'save_every': 1000, 'initial_checkpoint_updates': [1]}
            with patch('rmfsr.preflight.check', side_effect=AssertionError('full scan called')), \
                 patch('rmfsr.paths.FileVerifier.audio', side_effect=AssertionError('audio scan called')), \
                 patch('rmfsr.train.validate', return_value={'validation_restoration_mse_nfe1': .2}), \
                 contextlib.redirect_stdout(io.StringIO()):
                train(cfg, root / 'run')
                checkpoint = torch.load(root / 'run/latest.pt', weights_only=False)
                self.assertEqual(checkpoint['step'], 2)
                self.assertFalse(checkpoint['input_fingerprints']['full_audio_verified'])
                train({**cfg, 'steps': 3}, root / 'run', root / 'run/latest.pt')
                before = startup_check(cfg)['input_fingerprints']
                path = Path(cfg['train_manifest'])
                rows = json.loads(path.read_text())
                rows[0]['source'] = 'changed metadata'
                path.write_text(json.dumps(rows))
                self.assertNotEqual(before, startup_check(cfg)['input_fingerprints'])
                with self.assertRaisesRegex(ValueError, 'dataset content'):
                    train({**cfg, 'steps': 4}, root / 'run', root / 'run/latest.pt')

    def test_noise_selection_uses_indexes_without_walking_audio(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'complete.json').write_text(json.dumps({'shards': ['one']}))
            (root / 'one.json').write_text(json.dumps([
                {'path': 'train/b.wav'}, {'path': 'validation/a.wav'}, {'path': 'train/a.wav'}]))
            with patch.object(Path, 'rglob', side_effect=AssertionError('corpus walk called')):
                paths = indexed_noise_paths(root / 'complete.json', root / 'train')
            self.assertEqual(paths, [root.resolve() / 'train/a.wav', root.resolve() / 'train/b.wav'])


if __name__ == '__main__':
    unittest.main()
