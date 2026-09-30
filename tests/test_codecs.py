import unittest
import numpy as np
from rmfsr.degradations import codec_roundtrip
from rmfsr.train import get_ffmpeg
from rmfsr.gsm import library_path

class CodecTests(unittest.TestCase):
    def test_gsm_mp3_length_and_energy(self):
        library_path();ffmpeg=get_ffmpeg(None)
        x=(.2*np.sin(2*np.pi*440*np.arange(16000)/16000)).astype(np.float32)
        # Fixed seeds exercise both codec choices.
        for seed in (1,2):
            y=codec_roundtrip(x,np.random.default_rng(seed),16000,ffmpeg)
            self.assertEqual(len(y),len(x));self.assertTrue(np.isfinite(y).all())
            self.assertGreater(float(np.sqrt(np.mean(y**2))),.05)
if __name__=='__main__':unittest.main()
