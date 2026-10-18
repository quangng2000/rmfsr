import io
from pathlib import Path
import tarfile
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import soundfile as sf

from rmfsr.corpus import convert_tar


class CorpusArchiveTests(unittest.TestCase):
    def setUp(self):
        # Archive fixtures must not depend on available space on the test host.
        disk = patch('rmfsr.corpus.shutil.disk_usage',
                     return_value=SimpleNamespace(free=16 * 1024**3))
        disk.start()
        self.addCleanup(disk.stop)

    def archive(self, path, members):
        with tarfile.open(path, 'w:gz') as archive:
            for name, data in members:
                entry = tarfile.TarInfo(name)
                entry.size = len(data)
                archive.addfile(entry, io.BytesIO(data))

    def test_daps_metadata_is_ignored_and_produced_audio_is_converted(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            wave = io.BytesIO()
            sf.write(wave, np.zeros(1600, dtype=np.float32), 16000, format='WAV')
            archive = root / 'daps.tar.gz'
            self.archive(archive, [
                ('daps/produced/._speech.wav', b'\x00\x05\x16\x07metadata'),
                ('__MACOSX/daps/produced/speech.wav', b'metadata'),
                ('daps/clean/speech.wav', wave.getvalue()),
                ('daps/produced/speech.wav', wave.getvalue()),
            ])
            rows = convert_tar(archive, root, 'fixture', 'daps', 'fixture-digest')
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]['source_member'], 'daps/produced/speech.wav')
            self.assertEqual(rows[0]['frames'], 1600)
            self.assertTrue((root / rows[0]['path']).is_file())

    def test_invalid_actual_audio_still_fails_with_member_name(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / 'daps.tar.gz'
            self.archive(archive, [('daps/produced/broken.wav', b'not audio')])
            with self.assertRaisesRegex(ValueError, 'daps/produced/broken.wav'):
                convert_tar(archive, root, 'fixture', 'daps', 'fixture-digest')

    def test_low_disk_space_preserves_conversion_guard(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / 'daps.tar.gz'
            self.archive(archive, [('daps/produced/speech.wav', b'not decoded')])
            with patch('rmfsr.corpus.shutil.disk_usage',
                       return_value=SimpleNamespace(free=7 * 1024**3)):
                with self.assertRaisesRegex(RuntimeError, 'preserve 8 GiB'):
                    convert_tar(archive, root, 'fixture', 'daps', 'fixture-digest')


if __name__ == '__main__':
    unittest.main()
