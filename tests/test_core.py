import json,tempfile,unittest
from pathlib import Path
import numpy as np
import torch
from rmfsr.model import RMFSR
from rmfsr.spectral import Spectral
from rmfsr.flow import sample,meanflow_loss,sample_times,expand
from rmfsr.data import assert_disjoint
from rmfsr.degradations import image_rir,corrupt

torch.set_num_threads(2)
class ModelTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(1)
        self.m=RMFSR(channels=(8,8,16,16,16)).eval()
        torch.nn.init.normal_(self.m.head.conv.weight,std=.01)
        self.x=torch.randn(1,2,33,31);self.y=torch.randn_like(self.x)
        self.t=torch.tensor([.7]);self.r=torch.tensor([.3])
    def test_future_does_not_affect_past(self):
        a=self.m(self.x,self.y,self.t,self.r)
        x=self.x.clone();y=self.y.clone();x[...,17:]+=20;y[...,17:]-=30
        b=self.m(x,y,self.t,self.r)
        torch.testing.assert_close(a[...,:17],b[...,:17],rtol=0,atol=1e-6)
    def test_streaming_cache_matches_full(self):
        a=self.m(self.x,self.y,self.t,self.r);cache={}
        b=torch.cat([self.m(self.x[...,i:i+7],self.y[...,i:i+7],self.t,self.r,cache) for i in range(0,31,7)],-1)
        torch.testing.assert_close(a,b,rtol=2e-4,atol=2e-5)
    def test_nfe_specific_streaming_caches(self):
        e=torch.randn_like(self.y)
        for nfe in (1,2,5):
            a=sample(self.m,self.y,nfe,noise=e)
            b=sample(self.m,self.y,nfe,noise=e,chunk_frames=6)
            torch.testing.assert_close(a,b,rtol=5e-4,atol=5e-5)
    def test_backward_finite_off_diagonal(self):
        loss,_=meanflow_loss(self.m,self.x,self.y,t=self.t,r=self.r)
        loss.backward()
        self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in self.m.parameters()))
    def test_diagonal_reduces_to_data_prediction(self):
        e=torch.randn_like(self.x);tt=expand(self.t)
        loss,_=meanflow_loss(self.m,self.x,self.y,t=self.t,r=self.t,noise=e,sigma_min=0)
        xt=(1-tt)*self.x+tt*self.y+tt*.3*e
        expected=(self.m(xt,self.y,self.t,self.t)-self.x).square().mean()
        torch.testing.assert_close(loss,expected)

class LinearModel(torch.nn.Module):
    def forward(self,x,y,t,r):return .2*x+.3*expand(t)+.4*expand(r)

class MathTests(unittest.TestCase):
    def test_jvp_sign_and_model_tangent(self):
        torch.manual_seed(4);x=torch.randn(2,2,4,3);y=torch.randn_like(x);e=torch.randn_like(x)
        t=torch.tensor([.7,.4]);r=torch.tensor([.2,.1]);tt,rr=expand(t),expand(r)
        z=(1-tt)*x+tt*y+tt*.3*e
        vc=y-x+.3*e
        # u=(.8*z-.3*t-.4*r)/t. Direction uses diagonal model velocity.
        jvp=-.2*.8*z/tt.square()-.8*(.3+.4)/tt+.4*rr/tt.square()
        target=z-tt*vc+tt*(tt-rr)*jvp
        expected=(LinearModel()(z,y,t,r)-target).square().mean()
        actual,_=meanflow_loss(LinearModel(),x,y,t=t,r=r,noise=e,sigma_min=0)
        torch.testing.assert_close(actual,expected,rtol=1e-5,atol=1e-6)
    def test_time_bounds(self):
        for progress in (0,.5,1):
            t,r=sample_times(2000,progress,'cpu')
            self.assertTrue(((r>=0)&(r<=t)&(t>0)&(t<1)).all())

class AudioTests(unittest.TestCase):
    def test_roundtrip_boundaries(self):
        transform=Spectral()
        for n in (1,159,160,321,41600):
            x=torch.randn(2,n)*.1;x[:,0]=.9;x[:,-1]=-.8
            z=transform.encode(x);y=transform.decode(z,n)
            torch.testing.assert_close(x,y,rtol=1e-5,atol=2e-6)
    def test_stft_causal_frames(self):
        transform=Spectral();a=torch.randn(1,1600);b=a.clone();b[:,800:]+=10
        za=transform.encode(a);zb=transform.encode(b)
        # Frame index 4 sees original samples 480..799 (zero-based).
        torch.testing.assert_close(za[...,:5],zb[...,:5])
    def test_rir_direct_path_alignment(self):
        h,d,*_=image_rir(np.random.default_rng(2),16000)
        first=np.flatnonzero(abs(d)>0)[0]
        np.testing.assert_allclose(h[:first+2],d[:first+2],atol=1e-7)
    def test_degradations_finite(self):
        for seed in range(15):
            y,mask,kinds=corrupt(np.sin(np.arange(8000)*.1),np.random.default_rng(seed),16000,codec_enabled=False)
            self.assertEqual(y.shape,(8000,));self.assertTrue(np.isfinite(y).all());self.assertTrue((y[mask]==0).all())
    def test_split_leakage_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            paths=[Path(root)/'train.json',Path(root)/'val.json']
            for p in paths:p.write_text(json.dumps([{'speaker':'p001'}]))
            with self.assertRaises(ValueError):assert_disjoint(*paths)

if __name__=='__main__':unittest.main()
