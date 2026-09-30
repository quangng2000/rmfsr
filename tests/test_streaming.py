import unittest
import torch
from rmfsr.streaming import StreamingSTFT,StreamingISTFT,StreamingRestorer
from rmfsr.spectral import Spectral
from rmfsr.model import RMFSR
from rmfsr.flow import pink_noise_like,sample

class StreamTests(unittest.TestCase):
    def test_streaming_stft_and_overlap_add_boundaries(self):
        torch.manual_seed(3);wave=torch.randn(1823)*.1;wave[0]=.8;wave[-1]=-.9
        analysis=StreamingSTFT();synthesis=StreamingISTFT();spectra=[];audio=[]
        for i in range(0,len(wave),113):
            z=analysis.push(wave[i:i+113]);spectra.append(z);audio.append(synthesis.push(z))
        z=analysis.finish();spectra.append(z);audio.append(synthesis.push(z))
        torch.testing.assert_close(torch.cat(spectra,-1),Spectral().encode(wave))
        torch.testing.assert_close(torch.cat(audio)[:len(wave)],wave,rtol=1e-5,atol=1e-6)
    def test_streaming_restorer_matches_identical_noise_offline(self):
        torch.manual_seed(2);model=RMFSR(channels=(8,8,16,16,16)).eval()
        torch.nn.init.normal_(model.head.conv.weight,std=.01)
        wave=torch.randn(1023)*.05;spec=Spectral();y=spec.encode(wave);generator=torch.Generator().manual_seed(903)
        noise=torch.cat([pink_noise_like(frame,generator) for frame in y.split(1,dim=-1)],-1)
        expected=spec.decode(sample(model,y,2,noise=noise),len(wave))[0]
        stream=StreamingRestorer(model,steps=2);chunks=[stream.push(wave[i:i+211]) for i in range(0,len(wave),211)]
        chunks.append(stream.finish());actual=torch.cat(chunks)
        torch.testing.assert_close(actual,expected,rtol=5e-4,atol=1e-5)
        with self.assertRaises(RuntimeError):stream.push(wave)

if __name__=='__main__':unittest.main()
