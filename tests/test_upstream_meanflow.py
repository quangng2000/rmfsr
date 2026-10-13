"""Numerical contracts between the pinned MeanFlow source and speech adapter.

The original upstream class is evaluated directly from its archived source, so
the oracle cannot silently change alongside the extracted runtime helper. No
network, image model, checkpoint, or upstream training dependencies are needed.
"""
import ast
import copy
import hashlib
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from rmfsr import flow
from rmfsr._vendor.meanflow import core


SOURCE_SHA256 = '59ba2e10401386958a404aaa55e9cf71bdcebd129489a3142a6e57d8608b0e17'
SOURCE = Path(core.__file__).parent / 'upstream' / 'meanflow.py.txt'


def original_meanflow_class():
    """Load the untouched class while omitting image-only import dependencies."""
    tree = ast.parse(SOURCE.read_text())
    original_class = next(node for node in tree.body
                          if isinstance(node, ast.ClassDef) and node.name == 'MeanFlow')
    module = ast.Module(body=[original_class], type_ignores=[])
    namespace = {'torch': torch, 'nn': torch.nn}
    exec(compile(module, str(SOURCE), 'exec'), namespace)
    return namespace['MeanFlow'], namespace


class PolynomialSpeechModel(torch.nn.Module):
    """A data predictor whose state/time derivatives have a closed form."""
    def __init__(self):
        super().__init__()
        self.weights = torch.nn.Parameter(torch.tensor(
            [.17, -.031, .13, .23, -.11, .07], dtype=torch.float64))

    @staticmethod
    def features(z, y, t, r):
        one = torch.ones_like(z)
        return torch.stack((z, z.square(), y, one*flow.expand(t),
                            one*flow.expand(r), one*flow.expand(t*r)))

    def forward(self, z, y, t, r):
        return torch.einsum('p,pbcft->bcft', self.weights, self.features(z, y, t, r))


class UpstreamVelocityModel(PolynomialSpeechModel):
    def forward(self, z, times, aug_cond):
        t, span = times
        return super().forward(z, aug_cond, t, span)


class UpstreamParityTests(unittest.TestCase):
    def test_archived_reference_has_the_pinned_upstream_bytes(self):
        self.assertEqual(hashlib.sha256(SOURCE.read_bytes()).hexdigest(), SOURCE_SHA256)

    def test_target_helper_matches_original_loss_and_parameter_gradients(self):
        torch.manual_seed(39)
        clean = torch.randn(3, 2, 4, 5, dtype=torch.float64)
        noise = torch.randn_like(clean)
        condition = torch.randn_like(clean)
        t = torch.tensor([.07, .43, .97], dtype=torch.float64)
        r = torch.tensor([.07, .13, .02], dtype=torch.float64)
        model = UpstreamVelocityModel()
        reference_model = copy.deepcopy(model)

        original, namespace = original_meanflow_class()
        namespace['sample_two_timesteps'] = lambda *args, **kwargs: (t, r)
        # Bypass only construction of image architecture and EMA; run the
        # actual, unmodified upstream forward_with_loss method below.
        reference = original.__new__(original)
        torch.nn.Module.__init__(reference)
        reference.net = reference_model
        reference.args = SimpleNamespace(norm_eps=.03, norm_p=.7)
        with patch('torch.randn_like', return_value=noise):
            expected = reference.forward_with_loss(clean, condition)
        expected.backward()

        tt, rr = flow.expand(t), flow.expand(r)
        z = (1-tt)*clean + tt*noise
        v = noise-clean
        def velocity(state, time, start):
            return model(state, (time.flatten(), (time-start).flatten()), condition)
        prediction, target = core.velocity_target(velocity, z, tt, rr, v)
        self.assertFalse(target.requires_grad, 'JVP target must remain stop-gradient')
        squared_l2 = (prediction-target).square().sum(dim=(1, 2, 3))
        actual = (squared_l2/(squared_l2.detach()+.03).pow(.7)).mean()
        actual.backward()
        torch.testing.assert_close(actual, expected, rtol=1e-13, atol=1e-13)
        torch.testing.assert_close(model.weights.grad, reference_model.weights.grad,
                                   rtol=1e-13, atol=1e-13)
        self.assertGreater(model.weights.grad.abs().max().item(), .01)

    def test_explicit_speech_tangent_changes_only_the_total_derivative(self):
        torch.manual_seed(51)
        z = torch.randn(2, 2, 3, 4, dtype=torch.float64, requires_grad=True)
        t = flow.expand(torch.tensor([.2, .8], dtype=torch.float64))
        r = flow.expand(torch.tensor([.1, .3], dtype=torch.float64))
        conditional_velocity = torch.randn_like(z)
        model_velocity = torch.randn_like(z)
        def velocity(state, time, start):
            return state.square() + .4*time + .3*start
        prediction, target = core.velocity_target(
            velocity, z, t, r, conditional_velocity, tangent=model_velocity)
        wanted = conditional_velocity-(t-r)*(2*z*model_velocity+.4)
        torch.testing.assert_close(prediction, z.square()+.4*t+.3*r)
        torch.testing.assert_close(target, wanted.detach(), rtol=1e-13, atol=1e-13)
        self.assertFalse(target.requires_grad)


class SpeechAdapterTests(unittest.TestCase):
    def test_speech_times_use_official_sampler_and_retain_floor_and_span_schedule(self):
        normal = torch.tensor([-100., 0., .2])
        span_uniform = torch.tensor([0., .5, .999])
        diagonal_uniform = torch.tensor([.99, .01, .99])
        with patch('torch.randn', return_value=normal), \
             patch('torch.rand', side_effect=[span_uniform, diagonal_uniform]), \
             patch.object(flow.time_sampler, 'logit_normal_timestep_sample',
                          wraps=flow.time_sampler.logit_normal_timestep_sample) as sampler:
            t, r = flow.sample_times(3, progress=0., device='cpu')
        sampler.assert_called_once_with(.4, 1., 3, 'cpu')
        # At progress 0 the speech span exponent is .05. The first sample
        # exercises t's floor and r=0; the second is explicitly diagonal.
        wanted_t = torch.tensor([1e-4, .598687660112452, .645656306225795])
        wanted_r = torch.tensor([0., .598687660112452,
                                 .645656306225795 * .999**.05])
        torch.testing.assert_close(t, wanted_t, rtol=1e-6, atol=1e-7)
        torch.testing.assert_close(r, wanted_r, rtol=1e-6, atol=1e-7)

    def test_shared_restoration_step_preserves_partial_and_final_intervals(self):
        class FixedDataModel(torch.nn.Module):
            def forward(self, x, y, t, r, cache=None):
                return y
        x = torch.full((2, 2, 3, 4), 8., dtype=torch.float64)
        degraded = torch.full_like(x, 2.)
        t = torch.tensor([.8, .2], dtype=torch.float64)
        r = torch.tensor([.2, 0.], dtype=torch.float64)
        actual = flow.restoration_step(FixedDataModel(), x, degraded, t, r)
        # Retain r/t of the initial residual; r=0 lands on the data output.
        torch.testing.assert_close(actual[0], torch.full_like(actual[0], 3.5))
        torch.testing.assert_close(actual[1], degraded[1])

    def test_closed_form_speech_target_and_gradients_at_small_and_mixed_times(self):
        torch.manual_seed(18)
        clean = torch.randn(3, 2, 3, 4, dtype=torch.float64)
        damaged = torch.randn_like(clean)
        noise = torch.randn_like(clean)
        t = torch.tensor([1e-6, .43, .9999], dtype=torch.float64)
        r = torch.tensor([0., .12, .9999], dtype=torch.float64)
        tt, rr = flow.expand(t), flow.expand(r)

        for sigma_min in (0., .015):
            with self.subTest(sigma_min=sigma_min):
                model = PolynomialSpeechModel()
                sigma_max = .3
                z = ((1-tt)*clean + tt*damaged +
                     ((1-tt)*sigma_min+tt*sigma_max)*noise)
                with torch.no_grad():
                    a, b, c, d, e, f = model.weights
                    data_prediction = model(z, damaged, t, r)
                    diagonal_prediction = model(z, damaged, t, t)
                    instantaneous = (z-diagonal_prediction)/tt
                    # Hand derivative of u=(z-D(z,y,t,r))/t along
                    # dz/dt=instantaneous, dt/dt=1, dr/dt=0. No autograd/JVP
                    # participates in this oracle.
                    dudt = (((1-a)-2*b*z)*instantaneous/tt -
                            (d+f*rr)/tt - (z-data_prediction)/tt.square())
                    target = clean + sigma_min*noise + tt*(tt-rr)*dudt
                    residual = data_prediction-target
                    expected_loss = residual.square().mean()
                    features = model.features(z, damaged, t, r)
                    expected_gradient = (2*residual[None]*features).flatten(1).mean(1)

                with patch.object(core, 'velocity_target', wraps=core.velocity_target) as call:
                    actual, stats = flow.meanflow_loss(
                        model, clean, damaged, t=t, r=r, noise=noise,
                        sigma_max=sigma_max, sigma_min=sigma_min)
                    actual.backward()
                self.assertEqual(call.call_count, 1, 'Speech loss must execute the upstream core')
                self.assertTrue(torch.isfinite(actual))
                self.assertTrue(torch.isfinite(model.weights.grad).all())
                torch.testing.assert_close(actual, expected_loss, rtol=2e-11, atol=2e-12)
                torch.testing.assert_close(model.weights.grad, expected_gradient,
                                           rtol=2e-11, atol=2e-12)
                self.assertAlmostEqual(stats['target_rms'], target.square().mean().sqrt().item(),
                                       places=10)


if __name__ == '__main__':
    unittest.main()
