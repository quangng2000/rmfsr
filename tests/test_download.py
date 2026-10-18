import hashlib
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from rmfsr.download import download, digest


class DownloadTests(unittest.TestCase):
    def test_truncated_transfer_resumes_before_verified_publication(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / 'archive.tar'
            partial = target.with_name(target.name + '.download')
            calls = []

            def transfer(command, **kwargs):
                calls.append(command)
                self.assertFalse(target.exists())
                if len(calls) == 1:
                    partial.write_bytes(b'abc')
                    raise subprocess.CalledProcessError(18, command)
                self.assertEqual(partial.read_bytes(), b'abc')
                self.assertEqual(command[command.index('--continue-at') + 1], '-')
                with partial.open('ab') as handle:
                    handle.write(b'def')

            checksum = 'sha256:' + hashlib.sha256(b'abcdef').hexdigest()
            with patch('rmfsr.download.subprocess.run', side_effect=transfer), \
                    patch('rmfsr.download.time.sleep'), \
                    patch('rmfsr.download.digest', wraps=digest) as hash_file:
                result = download('https://example.org/archive', target, 6, checksum, reserve_gib=0)
            self.assertEqual(hash_file.call_count, 1)
            self.assertEqual(result.read_bytes(), b'abcdef')
            self.assertFalse(partial.exists())
            self.assertEqual(len(calls), 2)

    def test_retries_are_bounded_and_partial_is_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / 'archive.tar'
            partial = target.with_name(target.name + '.download')
            partial.write_bytes(b'abc')
            with patch('rmfsr.download.subprocess.run',
                       side_effect=subprocess.CalledProcessError(18, 'curl')) as transfer, \
                    patch('rmfsr.download.time.sleep'):
                with self.assertRaises(subprocess.CalledProcessError):
                    download('https://example.org/archive', target, 6, reserve_gib=0)
            self.assertEqual(transfer.call_count, 6)
            self.assertEqual(partial.read_bytes(), b'abc')
            self.assertFalse(target.exists())

    def test_cached_archive_checksum_is_still_verified(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / 'archive.tar'
            target.write_bytes(b'bad')
            with self.assertRaisesRegex(ValueError, 'Cached archive checksum mismatch'):
                download('https://example.org/archive', target, 3,
                         'sha256:' + hashlib.sha256(b'yes').hexdigest(), reserve_gib=0)

    def test_checksum_failure_never_publishes(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / 'archive.tar'
            partial = target.with_name(target.name + '.download')
            partial.write_bytes(b'bad')
            with self.assertRaisesRegex(ValueError, 'checksum mismatch'):
                download('https://example.org/archive', target, 3,
                         'sha256:' + hashlib.sha256(b'yes').hexdigest(), reserve_gib=0)
            self.assertFalse(target.exists())


if __name__ == '__main__':
    unittest.main()
