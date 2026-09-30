import contextlib
import io
import json
from pathlib import Path
import shutil
import tarfile
import tempfile
import unittest
import numpy as np
import soundfile as sf
import torch
from rmfsr.batches import batch_loader
from rmfsr.corpus import convert_tar, noise_split
from rmfsr.paths import load_manifest, manifest_fingerprint
from rmfsr.train import train
from rmfsr.download import download
from rmfsr.data import SpeechPairs
from rmfsr.degradations import spectral_augment


def fixture(root):
    manifests = []
    for split, speaker in [('train', 'p001'), ('validation', 'p002')]:
        path = root / (split + '.wav')
        sf.write(path, np.sin(np.arange(16000) * .1).astype('float32') * .1, 16000)
        manifest = root / (split + '.json')
        manifest.write_text(json.dumps([dict(path=path.name, speaker=speaker, split=split,
                                             frames=16000, seconds=1, sr=16000, source=split)]))
        manifests.append(str(manifest))
    return manifests


class PipelineTests(unittest.TestCase):
    def test_finished_partial_download_publishes_without_network(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'fixture.zip'
            path.with_suffix('.zip.download').write_bytes(b'abc')
            download('https://not-needed.invalid',path,expected_size=3,
                     checksum='sha256:ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad',reserve_gib=0)
            self.assertEqual(path.read_bytes(),b'abc')

    def test_figure2_pairs_and_eq_preserve_length_and_time_alignment(self):
        pulse=np.zeros(1600);pulse[800]=1
        shaped=spectral_augment(pulse,np.random.default_rng(4),16000)
        self.assertEqual(int(np.argmax(abs(shaped))),800)
        with tempfile.TemporaryDirectory() as temp:
            manifest,_=fixture(Path(temp))
            pairs=SpeechPairs(manifest,seconds=.2,pilot=True,augmentation_profile='figure2-v2')
            for _ in range(3):
                clean,damaged,mask,_=pairs.one()
                self.assertEqual(clean.shape,damaged.shape)
                self.assertTrue(np.isfinite(clean).all() and np.isfinite(damaged).all())
                self.assertTrue((damaged[mask]==0).all())

    def test_relocated_manifest_and_fingerprint(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            original = root / 'original'
            original.mkdir()
            manifests = fixture(original)
            shutil.copytree(original, root / 'relocated')
            moved = root / 'relocated' / 'train.json'
            self.assertEqual(manifest_fingerprint(manifests[0]), manifest_fingerprint(moved))
            self.assertEqual(Path(load_manifest(moved)[0]['path']), (root / 'relocated' / 'train.wav').resolve())

    def test_prefetch_and_worker_count_do_not_change_resumed_batch(self):
        with tempfile.TemporaryDirectory() as temp:
            manifest, _ = fixture(Path(temp))
            kwargs = dict(seconds=.2, pilot=True, sr=16000, ffmpeg=None)
            cfg = dict(seed=7, batch_size=1, steps=3, device='cpu', num_workers=0)
            serial = list(batch_loader(manifest, kwargs, cfg, 0))
            parallel = list(batch_loader(manifest, kwargs, {**cfg, 'num_workers': 2}, 1))
            for left, right in zip(serial[1:], parallel):
                for a, b in zip(left, right):
                    torch.testing.assert_close(a, b, rtol=0, atol=0)

    def test_archive_paths_cannot_escape_and_only_produced_daps_selected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            audio = io.BytesIO()
            sf.write(audio, np.zeros(1600), 16000, format='WAV')
            archive = root / 'test.tar.gz'
            with tarfile.open(archive, 'w:gz') as handle:
                for name in ('daps/produced/../../outside.wav', 'daps/cleanraw/a.wav'):
                    entry = tarfile.TarInfo(name)
                    entry.size = len(audio.getvalue())
                    handle.addfile(entry, io.BytesIO(audio.getvalue()))
            rows = convert_tar(archive, root, 'fixture', 'daps', 'fixture')
            self.assertEqual(len(rows), 1)
            self.assertFalse((root.parent / 'outside.wav').exists())
            self.assertEqual(Path(rows[0]['path']).parts[0], 'produced')
            self.assertTrue((root / rows[0]['path']).is_file())
            self.assertEqual(noise_split('a'), noise_split('a'))

    def test_checkpoint_resume_matches_uninterrupted_optimizer_steps(self):
        self.check_resume(1,'legacy')

    def test_accumulated_figure_schedule_checkpoint_resume(self):
        self.check_resume(2,'figure1-cosine-approx')

    def check_resume(self,accumulation,schedule):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            tr, va = fixture(root)
            cfg = dict(name='unit', pilot=True, train_manifest=tr, validation_manifest=va,
                       sample_rate=16000, seconds=.2, batch_size=1, channels=[8, 8, 16, 16, 16],
                       steps=4, schedule_steps=20, seed=3, device='cpu', threads=2,
                       accumulation_steps=accumulation,flow_schedule=schedule,
                       indexed_batches=True, num_workers=0, noise_dir=None, ltas_path=None,
                       ffmpeg=None, learning_rate=.0001, warmup_steps=2, ema_decay=.95,
                       gradient_clip=1, save_every=2, validate_every=2,
                       quality_review_updates=[2],milestone_every=1,keep_milestones=1)
            with contextlib.redirect_stdout(io.StringIO()):
                train(cfg, root / 'continuous')
                paused = train({**cfg, 'max_wall_seconds': .0001}, root / 'resumed')
                self.assertEqual(paused['status'], 'paused_checkpointed')
                train(cfg, root / 'resumed', root / 'resumed/latest.pt')
            a = torch.load(root / 'continuous/latest.pt', weights_only=False)
            b = torch.load(root / 'resumed/latest.pt', weights_only=False)
            for key in a['model']:
                torch.testing.assert_close(a['model'][key], b['model'][key], rtol=0, atol=0)
            self.assertEqual(a['history'][-1]['loss'], b['history'][-1]['loss'])
            self.assertTrue((root / 'resumed/review-00000002.pt').is_file())
            self.assertEqual(len(list((root / 'resumed').glob('step-*.pt'))),1)


if __name__ == '__main__':
    unittest.main()
