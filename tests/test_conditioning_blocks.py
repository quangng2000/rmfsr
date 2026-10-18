"""Analytic normalization checks, including axes and denominator derivatives."""
import unittest
import torch
from torch.nn import functional as F
from rmfsr.conditioning_blocks import FrameGroupNorm, DoubleNormalizedConv, pixel_normalize
from rmfsr.conditioning_blocks import PIXEL_EPSILON, GROUP_EPSILON, group_count
from test_conditioning import ToyLayer


class BlockTests(unittest.TestCase):
    def test_pixel_normalization_and_full_jvp(self):
        value = torch.tensor([[.2,-.4,.6,.1],[0.,0.,0.,0.]], dtype=torch.double)
        direction = torch.tensor([[.3,.2,-.1,.5],[.1,.2,.3,.4]], dtype=torch.double)
        size = (value.square().mean(1,keepdim=True)+PIXEL_EPSILON).sqrt()
        expected = direction/size-value*(value*direction).mean(1,keepdim=True)/size.pow(3)
        output, actual = torch.func.jvp(pixel_normalize,(value,),(direction,))
        torch.testing.assert_close(output,value/size)
        torch.testing.assert_close(actual,expected)
        self.assertFalse(torch.allclose(actual[0],(direction/size)[0]))

    def test_group_norm_matches_torch_per_frame_not_across_time(self):
        value = torch.randn(2,16,3,5,dtype=torch.double)
        module = FrameGroupNorm(16)
        expected = torch.stack([F.group_norm(value[...,index],module.groups,eps=GROUP_EPSILON)
                                for index in range(value.shape[-1])],-1)
        torch.testing.assert_close(module(value),expected)
        future = value.clone(); future[...,3:] *= 9
        torch.testing.assert_close(module(value)[...,:3],module(future)[...,:3])

    def test_single_frequency_groups_do_not_collapse_each_channel(self):
        for channels in (4,16,64,448):
            value = torch.arange(channels,dtype=torch.double).reshape(1,channels,1,1)
            groups = group_count(channels)
            self.assertEqual(channels%groups,0)
            self.assertGreaterEqual(channels//groups,4)
            self.assertGreater(float(FrameGroupNorm(channels)(value).square().mean()),.9)

    def test_double_normalization_matches_paper_formula(self):
        layer = DoubleNormalizedConv(ToyLayer().double())
        value = torch.randn(2,4,3,5,dtype=torch.double)
        embedding = torch.randn(2,128,dtype=torch.double)
        with torch.no_grad(): layer.scale.weight.normal_(0,.01)
        scale, bias = pixel_normalize(layer.scale(embedding)), pixel_normalize(layer.condition(embedding))
        expected = layer.norm(value)*scale[:,:,None,None]+bias[:,:,None,None]
        torch.testing.assert_close(layer.modulate(value,embedding),expected)
        changed = value.clone(); changed[1] *= 100
        torch.testing.assert_close(layer.modulate(value,embedding)[0],layer.modulate(changed,embedding)[0])

    def test_full_backward_derivative_includes_pixel_denominator(self):
        value = torch.tensor([[.2,-.4,.6,.1]],dtype=torch.double,requires_grad=True)
        upstream = torch.tensor([[.3,.2,-.1,.5]],dtype=torch.double)
        size = (value.square().mean(1,keepdim=True)+PIXEL_EPSILON).sqrt()
        expected = upstream/size-value*(value*upstream).mean(1,keepdim=True)/size.pow(3)
        actual, = torch.autograd.grad((pixel_normalize(value)*upstream).sum(),value)
        torch.testing.assert_close(actual,expected)


if __name__ == '__main__': unittest.main()
