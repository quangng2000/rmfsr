import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import soundfile as sf

from rmfsr.paths import (FileVerifier, manifest_fingerprint, verify_manifest,
                         noise_partition_fingerprints, ltas_fingerprint)
from rmfsr.preflight import check


def audio_row(root, name, phase=0, split='train', speaker='p001', recorded=True):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, np.sin(np.arange(320) * .1 + phase) * .1, 16000)
    row = dict(path=name, speaker=speaker, split=split, sr=16000,
               frames=320, seconds=.02, source='test')
    if recorded:
        row['audio_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
    return row


def manifest(root, name, row):
    path = root / name
    path.write_text(json.dumps([row]))
    return path


class IntegrityTests(unittest.TestCase):
    def test_legacy_audio_mutation_changes_fingerprint_including_shared_cache(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            row = audio_row(root, 'a.wav', recorded=False)
            path = manifest(root, 'train.json', row)
            verifier = FileVerifier()
            before = manifest_fingerprint(path, verifier=verifier)
            audio_row(root, 'a.wav', phase=.5, recorded=False)
            self.assertNotEqual(before, manifest_fingerprint(path, verifier=verifier))
            self.assertEqual(verify_manifest(path)['computed_legacy_hashes'], 1)

    def test_stale_recorded_hash_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            path = manifest(root, 'train.json', audio_row(root, 'a.wav'))
            audio_row(root, 'a.wav', phase=.5)
            with self.assertRaisesRegex(ValueError, 'SHA-256 mismatch'):
                manifest_fingerprint(path)

    def test_full_requires_hashes_and_header_metadata_matches_actual_audio(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            row = audio_row(root, 'a.wav', recorded=False)
            path = manifest(root, 'train.json', row)
            with self.assertRaisesRegex(ValueError, 'Missing recorded audio_sha256'):
                manifest_fingerprint(path, require_hashes=True)
            row['frames'] += 1
            manifest(root, 'train.json', row)
            with self.assertRaisesRegex(ValueError, 'frame count mismatch'):
                manifest_fingerprint(path)
            row['frames'] -= 1
            row['sr'] = 8000
            manifest(root, 'train.json', row)
            with self.assertRaisesRegex(ValueError, 'sample rate mismatch'):
                manifest_fingerprint(path)

    def test_relocation_preserves_verified_content_identity(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            original = root / 'original'
            original.mkdir()
            original_path = manifest(original, 'train.json', audio_row(original, 'a.wav'))
            shutil.copytree(original, root / 'moved')
            self.assertEqual(manifest_fingerprint(original_path),
                             manifest_fingerprint(root / 'moved/train.json'))

    def test_selected_noise_partition_swap_changes_fingerprint(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            audio_row(root, 'train/a.wav')
            audio_row(root, 'validation/a.wav', phase=.5)
            before = noise_partition_fingerprints(root / 'train', root / 'validation')
            swapped = noise_partition_fingerprints(root / 'validation', root / 'train')
            self.assertNotEqual(before['noise_train'], swapped['noise_train'])
            self.assertNotEqual(before['noise_validation'], swapped['noise_validation'])
            shutil.copytree(root / 'train', root / 'moved/train')
            shutil.copytree(root / 'validation', root / 'moved/validation')
            self.assertEqual(before, noise_partition_fingerprints(root / 'moved/train', root / 'moved/validation'))

    def test_identical_noise_content_different_paths_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            audio_row(root, 'train/a.wav')
            (root / 'validation').mkdir()
            shutil.copyfile(root / 'train/a.wav', root / 'validation/different-name.wav')
            with self.assertRaisesRegex(ValueError, 'Noise content leakage'):
                noise_partition_fingerprints(root / 'train', root / 'validation')

    def test_noise_selection_must_match_verified_partition_provenance(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            rows = [audio_row(root, 'train/a.wav'),
                    audio_row(root, 'validation/b.wav', phase=.5, split='validation')]
            for row in rows:
                row['path'] = str(root / row['path'])
            noise_partition_fingerprints(root / 'train', root / 'validation',
                                         provenance_rows=rows, require_hashes=True)
            with self.assertRaisesRegex(ValueError, 'partition mismatch'):
                noise_partition_fingerprints(root / 'validation', root / 'train',
                                             provenance_rows=rows, require_hashes=True)
            audio_row(root, 'train/unrecorded.wav', phase=.7)
            with self.assertRaisesRegex(ValueError, 'missing from verified provenance'):
                noise_partition_fingerprints(root / 'train', root / 'validation',
                                             provenance_rows=rows, require_hashes=True)

    def test_ltas_validates_values_provenance_and_recorded_digest(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'ltas.npy'
            np.save(path, np.ones(161))
            metadata = {'source': 'DAPS produced', 'sample_rate': 16000,
                        'normalization_dbfs': -25,
                        'files': [{'path': 'a.flac', 'sha256': 'a' * 64}],
                        'ltas_sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
            Path(str(path) + '.json').write_text(json.dumps(metadata))
            ltas_fingerprint(path, require_provenance=True)
            np.save(path, np.ones(161) * 2)
            with self.assertRaisesRegex(ValueError, 'LTAS SHA-256 mismatch'):
                ltas_fingerprint(path)
            np.save(path, np.full(161, np.nan))
            with self.assertRaisesRegex(ValueError, 'finite, positive'):
                ltas_fingerprint(path)
            np.save(path, np.ones(162))
            with self.assertRaisesRegex(ValueError, '161'):
                ltas_fingerprint(path)

    def test_full_preflight_verifies_noise_audio_beyond_manifest_inventory(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            cfg = dict(pilot=False, sample_rate=16000,
                       noise_dir=str(root / 'noise/train'),
                       validation_noise_dir=str(root / 'noise/validation'),
                       noise_completion=str(root / 'noise/complete.json'),
                       ltas_path=str(root / 'missing-ltas.npy'))
            for key, split, speaker, phase in [('train_manifest', 'train', 'p001', 0),
                                                ('validation_manifest', 'validation', 'p100', .5),
                                                ('test_manifest', 'test', 'p104', .9)]:
                cfg[key] = str(manifest(root, split + '.json', audio_row(root, split + '.wav',
                                       split=split, speaker=speaker, phase=phase)))
            noise_root = root / 'noise'
            rows = [audio_row(noise_root, 'train/a.wav'),
                    audio_row(noise_root, 'validation/b.wav', phase=.5, split='validation')]
            shard = noise_root / 'fixture-shard.json'
            shard.write_text(json.dumps(rows))
            (noise_root / 'complete.json').write_text(json.dumps(dict(
                shards=['fixture-shard'], sample_rate=16000,
                manifest_sha256={'fixture-shard': hashlib.sha256(shard.read_bytes()).hexdigest()})))
            audio_row(noise_root, 'train/a.wav', phase=.7)
            with patch('rmfsr.corpus.SHARDS', ['fixture-shard']), \
                    patch('rmfsr.preflight.library_path', return_value='fixture'), \
                    patch('rmfsr.preflight.get_ffmpeg', return_value=None):
                result = check(cfg)
            self.assertFalse(result['ready'])
            self.assertTrue(any('DNS shard manifest/audio invalid' in problem and 'SHA-256 mismatch' in problem
                                for problem in result['problems']))
            self.assertEqual(result['input_fingerprints']['integrity_version'], 'verified-content-v1')


if __name__ == '__main__':
    unittest.main()
