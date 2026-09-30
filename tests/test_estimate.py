import copy
import unittest
from unittest.mock import patch
import torch
from rmfsr.flow import meanflow_loss, schedule_values, sample_times
from rmfsr.model import RMFSR
from rmfsr.train import backward_microbatches

class IdentitySpectrum:
    def encode(self,x):
        return x

class EstimateTests(unittest.TestCase):
    def test_accumulation_matches_full_batch_meanflow_gradient(self):
        torch.manual_seed(7)
        model=RMFSR(channels=(8,8,16,16,16))
        torch.nn.init.normal_(model.head.conv.weight,std=.01)
        reference=copy.deepcopy(model)
        x=torch.randn(2,2,33,11);y=torch.randn_like(x)
        def fixed_loss(network,clean,damaged,progress=0,**kwargs):
            t=clean.new_full((len(clean),),.7);r=clean.new_full((len(clean),),.2)
            return meanflow_loss(network,clean,damaged,t=t,r=r,noise=torch.zeros_like(clean))
        expected,_=fixed_loss(reference,x,y)
        expected.backward()
        batches=iter([(x[:1],y[:1]),(x[1:],y[1:])])
        with patch('rmfsr.train.meanflow_loss',side_effect=fixed_loss):
            actual=backward_microbatches(model,batches,IdentitySpectrum(),torch.device('cpu'),.4,2)
        self.assertAlmostEqual(expected.item(),actual['loss'],places=5)
        for param,wanted in zip(model.parameters(),reference.parameters()):
            if wanted.grad is not None:
                torch.testing.assert_close(param.grad,wanted.grad,rtol=2e-4,atol=2e-6)

    def test_figure_schedule_endpoints_monotonicity_and_sampling(self):
        name='figure1-cosine-approx'
        self.assertEqual(schedule_values(0,name),(1.,.05))
        self.assertAlmostEqual(schedule_values(.5,name)[0],.2)
        self.assertEqual(schedule_values(.8,name),(.2,1.))
        values=[schedule_values(i/100,name) for i in range(101)]
        self.assertTrue(all(a[0]>=b[0] and a[1]<=b[1] for a,b in zip(values,values[1:])))
        torch.manual_seed(42)
        t,r=sample_times(10000,.9,'cpu',name)
        self.assertTrue(((r>=0)&(r<=t)).all())
        self.assertLess(abs((r==t).float().mean().item()-.2),.02)

if __name__=='__main__':
    unittest.main()
