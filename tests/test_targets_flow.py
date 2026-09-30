"""Regression checks for level-independent targets and diagonal MeanFlow loss."""
import copy
import unittest
from unittest.mock import patch

import numpy as np
from scipy import signal
import torch

from rmfsr.degradations import level, rms, target_process
from rmfsr.flow import expand, meanflow_loss


class TimeAwareModel(torch.nn.Module):
    """Small differentiable model with nontrivial state/time dependencies."""
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Conv2d(7, 6, 1), torch.nn.Tanh(), torch.nn.Conv2d(6, 2, 1))

    def forward(self, x, y, t, r):
        times = torch.stack((t, r, t*r), dim=1)[:, :, None, None]
        times = times.expand(-1, -1, x.shape[2], x.shape[3])
        return self.layers(torch.cat((x, y, times), dim=1))


def original_loss(model, clean, degraded, t, r, noise, sigma_max, sigma_min):
    """Pre-optimization algebra, including its unconditional JVP."""
    tt, rr = expand(t), expand(r)
    xt = (1-tt)*clean + tt*degraded + ((1-tt)*sigma_min+tt*sigma_max)*noise
    conditional_velocity = degraded-clean + (sigma_max-sigma_min)*noise
    with torch.no_grad():
        instantaneous = (xt-model(xt, degraded, t, t))/tt
        def velocity(z, end, start):
            return (z-model(z, degraded, start, end))/expand(start)
        _, jvp = torch.func.jvp(velocity, (xt, r, t),
                               (instantaneous, torch.zeros_like(r), torch.ones_like(t)))
        target = xt-tt*conditional_velocity+tt*(tt-rr)*jvp
    return (model(xt, degraded, t, r)-target.detach()).square().mean()


class TargetEQTests(unittest.TestCase):
    def test_colored_reference_eq_is_independent_of_recording_volume(self):
        rng = np.random.default_rng(17)
        sr = 16000
        source = signal.lfilter([1, -.4], [1, -.8], rng.standard_normal(sr*2))
        reference = signal.lfilter([1, .6], [1, -.3], rng.standard_normal(sr*2))
        reference = level(reference, -25)
        _, _, z = signal.stft(reference, sr, nperseg=320, noverlap=160)
        reference_ltas = (np.abs(z)**2).mean(axis=1)+1e-10
        wanted = target_process(source, sr, reference_ltas)
        for scale in (.1, 1., 10.):
            actual = target_process(scale*source, sr, reference_ltas)
            np.testing.assert_allclose(actual, wanted, rtol=2e-6, atol=2e-7)
            self.assertAlmostEqual(rms(actual), 10**(-25/20), places=6)
        # Exercise actual spectral shaping rather than only output normalization.
        unshaped = target_process(source, sr)
        self.assertGreater(rms(wanted-unshaped), .002)

    def test_silent_target_with_reference_stays_finite_and_silent(self):
        actual = target_process(np.zeros(16000), 16000, np.full(161, 1e-5))
        self.assertTrue(np.isfinite(actual).all())
        np.testing.assert_array_equal(actual, 0)


class DiagonalFlowTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(19)
        self.model = TimeAwareModel().double()
        self.clean = torch.randn(3, 2, 5, 4, dtype=torch.float64)
        self.degraded = torch.randn_like(self.clean)
        self.noise = torch.randn_like(self.clean)
        self.t = torch.tensor([1e-4, .45, .9999], dtype=torch.float64)

    def assert_original_loss_and_all_gradients_match(self, r, sigma_min):
        reference = copy.deepcopy(self.model)
        expected = original_loss(reference, self.clean, self.degraded,
                                 self.t, r, self.noise, .3, sigma_min)
        expected.backward()
        actual, _ = meanflow_loss(self.model, self.clean, self.degraded,
                                 t=self.t, r=r, noise=self.noise,
                                 sigma_max=.3, sigma_min=sigma_min)
        actual.backward()
        torch.testing.assert_close(actual, expected, rtol=2e-11, atol=2e-12)
        for (name, param), (_, wanted) in zip(self.model.named_parameters(),
                                             reference.named_parameters()):
            self.assertIsNotNone(param.grad, name)
            self.assertIsNotNone(wanted.grad, name)
            self.assertTrue(torch.isfinite(param.grad).all(), name)
            torch.testing.assert_close(param.grad, wanted.grad, rtol=2e-11, atol=2e-12,
                                       msg=lambda msg: name+': '+msg)

    def test_diagonal_loss_and_gradients_match_original_with_noise_floor(self):
        self.assert_original_loss_and_all_gradients_match(self.t, .015)

    def test_diagonal_loss_and_gradients_match_original_with_zero_floor(self):
        self.assert_original_loss_and_all_gradients_match(self.t, 0.)

    def test_diagonal_skips_jvp_and_extra_model_forwards(self):
        with patch('torch.func.jvp', side_effect=AssertionError('diagonal JVP')), \
             patch.object(self.model, 'forward', wraps=self.model.forward) as forward:
            loss, _ = meanflow_loss(self.model, self.clean, self.degraded,
                                   t=self.t, r=self.t, noise=self.noise)
            loss.backward()
        self.assertEqual(forward.call_count, 1)

    def test_mixed_batch_preserves_original_loss_and_gradients(self):
        r = self.t.clone()
        r[1] = .2
        with patch('torch.func.jvp', wraps=torch.func.jvp) as jvp:
            self.assert_original_loss_and_all_gradients_match(r, .015)
        # Both original reference and production path still need a JVP.
        self.assertEqual(jvp.call_count, 2)

    def test_diagonal_target_avoids_large_term_cancellation(self):
        # Large damaged values can erase the clean endpoint in xt-t*v_cond.
        clean = torch.full((1, 2, 5, 4), .125)
        degraded = torch.full_like(clean, 1e8)
        noise = torch.full_like(clean, .5)
        t = torch.tensor([.9999])
        sigma_min = .02
        class ZeroModel(torch.nn.Module):
            def forward(self, x, y, t, r):
                return torch.zeros_like(x)
        loss, stats = meanflow_loss(ZeroModel(), clean, degraded, t=t, r=t,
                                   noise=noise, sigma_min=sigma_min)
        wanted = clean+sigma_min*noise
        torch.testing.assert_close(loss, wanted.square().mean())
        self.assertAlmostEqual(stats['target_rms'], wanted.square().mean().sqrt().item())


if __name__ == '__main__':
    unittest.main()
