import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

from rmfsr.paper_evaluate import DNSMOSSIG, aggregate, read_wave, word_counts


class PaperEvaluationTests(unittest.TestCase):
    def test_wer_uses_total_words_not_mean_clip_wer(self):
        rows = [{'variant': 'nfe5', 'asr': {'reference_words': 1, 'substitutions': 1,
                 'deletions': 0, 'insertions': 0}},
                {'variant': 'nfe5', 'asr': {'reference_words': 9, 'substitutions': 0,
                 'deletions': 0, 'insertions': 0}}]
        self.assertAlmostEqual(aggregate(rows)['nfe5']['wer'], 0.1)

    def test_transcript_normalization_and_insertions(self):
        result = word_counts('Hello, world!', 'hello new world')
        self.assertEqual(result['reference_words'], 2)
        self.assertEqual(result['insertions'], 1)
        self.assertAlmostEqual(result['wer'], .5)

    def test_multilingual_reference_preserves_diacritics_and_removes_unknown_tag(self):
        result = word_counts('<UNKNOWN/> Für den Profi.', 'für den profi')
        self.assertEqual(result['reference_normalized'].strip(), 'für den profi')
        self.assertEqual(result['wer'], 0)

    def test_dnsmos_short_clip_repeats_and_calibrates_sig(self):
        class Session:
            def run(self, outputs, inputs):
                segment = inputs['input_1']
                self.seen = segment
                return [np.array([[3., 4., 2.]], dtype=np.float32)]
        score = DNSMOSSIG.__new__(DNSMOSSIG)
        score.session = Session()
        x = np.tile(np.arange(10, dtype=np.float32) / 10, 1600)
        expected = -.08397278 * 9 + 1.22083953 * 3 + .0052439
        self.assertAlmostEqual(score(x), expected)
        self.assertEqual(score.session.seen.shape, (1, 144160))
        np.testing.assert_array_equal(score.session.seen[0, :20], x[:20])

    def test_resample_preserves_duration_and_rejects_empty_audio(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'tone.wav'
            x = .2 * np.sin(2 * np.pi * 1000 * np.arange(48000) / 48000)
            sf.write(path, x, 48000, subtype='FLOAT')
            y = read_wave(path)
            self.assertEqual(len(y), 16000)
            self.assertAlmostEqual(float(np.sqrt(np.mean(y*y))), .2/np.sqrt(2), places=3)
            sf.write(path, np.zeros(0), 16000)
            with self.assertRaises(ValueError):
                read_wave(path)


if __name__ == '__main__':
    unittest.main()
