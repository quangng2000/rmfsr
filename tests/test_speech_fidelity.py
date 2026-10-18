"""CPU gradient, numerical parity and resume checks for opt-in paper-1 losses."""
import contextlib
import copy
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from rmfsr.flow import expand, meanflow_loss
from rmfsr.flow_curriculum import config_digest, training_flow_progress
from rmfsr.spectral import Spectral
from rmfsr.speech_fidelity import (SpeechFidelityLoss, fidelity_config,
                                   fidelity_resume_transition, negative_si_sdr)
from rmfsr.train import train
from test_targets_flow import TimeAwareModel
from test_training_integration import fixture

SETTINGS = dict(version='paper1-pesq-sisdr-v1', pesq_weight=1e-3, sisdr_weight=1e-4)
HAS_PESQ = importlib.util.find_spec('torch_pesq') is not None


class SpectralGradientTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(29)
        self.spectral = Spectral()

    def test_inverse_matches_existing_decoder_at_boundaries(self):
        for length in (1, 159, 160, 321, 9600):
            with self.subTest(length=length):
                wave = torch.randn(2, length) * .1
                coefficients = self.spectral.encode(wave).requires_grad_()
                expected = self.spectral.decode(coefficients, length)
                actual = self.spectral.decode_differentiable(coefficients, length)
                torch.testing.assert_close(actual, expected, rtol=1e-5, atol=2e-6)
                torch.testing.assert_close(actual, wave, rtol=1e-5, atol=2e-6)
                actual.square().mean().backward()
                self.assertTrue(torch.isfinite(coefficients.grad).all())
                self.assertGreater(coefficients.grad.norm().item(), 0)

    def test_decoder_gradient_matches_finite_difference_and_handles_zero(self):
        z = (.1 * torch.randn(1, 2, 161, 3, dtype=torch.float64)).requires_grad_()
        self.assertTrue(torch.autograd.gradcheck(
            lambda value: self.spectral.decode_differentiable(value, 160), (z,), fast_mode=True))
        zero = torch.zeros_like(z, requires_grad=True)
        self.spectral.decode_differentiable(zero, 160).sum().backward()
        self.assertTrue(torch.isfinite(zero.grad).all())


class FidelityMathTests(unittest.TestCase):
    def test_si_sdr_matches_projection_formula_and_is_gain_invariant(self):
        torch.manual_seed(31)
        ref = torch.randn(2, 9000, dtype=torch.float64)
        pred = ref + .1 * torch.randn_like(ref)
        projection = (pred * ref).sum(-1, keepdim=True) / ref.square().sum(-1, keepdim=True) * ref
        expected = -10 * torch.log10(projection.square().sum(-1) / (pred-projection).square().sum(-1))
        torch.testing.assert_close(negative_si_sdr(pred, ref), expected, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(negative_si_sdr(.5*pred, ref), negative_si_sdr(pred, ref), atol=1e-4, rtol=1e-5)
        for value in (ref, torch.zeros_like(ref)):
            self.assertTrue(torch.isfinite(negative_si_sdr(value, ref)).all())

    def test_auxiliary_uses_diagonal_estimate_even_for_meanflow_jumps(self):
        torch.manual_seed(37)
        model = TimeAwareModel().double()
        reference_model = copy.deepcopy(model)
        clean = .1 * torch.randn(2, 2, 5, 4, dtype=torch.float64)
        degraded = torch.randn_like(clean)
        noise = torch.randn_like(clean)
        t = torch.tensor([.4, .8], dtype=torch.float64)
        r = torch.tensor([.4, .1], dtype=torch.float64)
        seen = []
        def auxiliary(estimate):
            seen.append(estimate.detach().clone())
            return .2 * (estimate-clean).abs().mean(), {}
        actual, _ = meanflow_loss(model, clean, degraded, t=t, r=r, noise=noise, fidelity=auxiliary)
        actual.backward()
        base, _ = meanflow_loss(reference_model, clean, degraded, t=t, r=r, noise=noise)
        tt = expand(t)
        xt = (1-tt)*clean+tt*degraded+((1-tt)*1e-8+tt*.3)*noise
        diagonal = reference_model(xt, degraded, t, t)
        expected = base + .2 * (diagonal-clean).abs().mean()
        expected.backward()
        torch.testing.assert_close(seen[0], diagonal.detach(), rtol=0, atol=0)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        for p, q in zip(model.parameters(), reference_model.parameters()):
            torch.testing.assert_close(p.grad, q.grad, rtol=1e-12, atol=1e-12)

    def test_diagonal_auxiliary_reuses_model_forward(self):
        model = TimeAwareModel()
        clean = torch.randn(1, 2, 5, 4)
        t = torch.tensor([.6])
        with patch.object(model, 'forward', wraps=model.forward) as forward:
            loss, _ = meanflow_loss(model, clean, clean, t=t, r=t,
                                   fidelity=lambda x: (x.square().mean(), {}))
            loss.backward()
        self.assertEqual(forward.call_count, 1)

    def test_requires_explicit_enable_and_rejects_silent_recipe_changes(self):
        self.assertIsNone(fidelity_config({}))
        with self.assertRaises(ValueError):
            fidelity_resume_transition({}, {'speech_fidelity': SETTINGS})
        cfg = dict(speech_fidelity=SETTINGS, allow_speech_fidelity_on_resume=True)
        self.assertIsNotNone(fidelity_resume_transition({}, cfg))
        self.assertIsNone(fidelity_resume_transition(cfg, cfg))
        with self.assertRaises(ValueError):
            fidelity_resume_transition(cfg, {**cfg, 'speech_fidelity': {**SETTINGS, 'pesq_weight': .1}})
        for value in (-1, True, float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                fidelity_config({'speech_fidelity': {**SETTINGS, 'sisdr_weight': value}})


@unittest.skipUnless(HAS_PESQ, 'Install optional fidelity dependencies to test torch-pesq')
class RealLossIntegrationTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(41)
        self.spectral = Spectral()
        self.loss = SpeechFidelityLoss(self.spectral, SETTINGS)

    def test_actual_pesq_and_sisdr_backpropagate_and_log_components(self):
        t = torch.arange(9600) / 16000
        clean = (.1*torch.sin(2*torch.pi*220*t) + .03*torch.sin(2*torch.pi*700*t))[None]
        coefficients = self.spectral.encode(clean + .005*torch.randn_like(clean)).requires_grad_()
        total, stats = self.loss(coefficients, clean)
        total.backward()
        self.assertTrue(torch.isfinite(coefficients.grad).all())
        self.assertGreater(coefficients.grad.norm().item(), 0)
        self.assertAlmostEqual(total.item(), stats['weighted_pesq_loss']+stats['weighted_si_sdr_loss'])
        self.assertEqual(stats['fidelity_pesq_skipped'], 0)
        self.assertFalse(any(p.requires_grad for p in self.loss.parameters()))

    def test_silent_reference_and_prediction_do_not_produce_nans(self):
        clean = .02 * torch.randn(2, 9600)
        clean[0] = 0
        coefficients = torch.zeros_like(self.spectral.encode(clean), requires_grad=True)
        value, stats = self.loss(coefficients, clean)
        value.backward()
        self.assertTrue(torch.isfinite(value))
        self.assertTrue(torch.isfinite(coefficients.grad).all())
        self.assertEqual(stats['fidelity_silent_references'], 1)
        self.assertEqual(stats['fidelity_pesq_skipped'], 2)

    def test_resume_preserves_flow_clock_and_checkpoints_loss_recipe(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cfg = {**fixture(root), 'seconds': .6, 'steps': 4,
                   'flow_schedule': 'figure1-cosine-approx', 'max_wall_seconds': .0001}
            run = root/'run'
            with patch('rmfsr.train.validate', return_value={'validation_restoration_mse_nfe1': .2}), contextlib.redirect_stdout(io.StringIO()):
                train(cfg, run)
                request = dict(version=1, expected_config_sha256=config_digest(cfg),
                               ramp_updates=3, target_progress=.8)
                (run/'flow-schedule-transition.json').write_text(json.dumps(request))
                train(cfg, run, run/'latest.pt')
                before = torch.load(run/'latest.pt', weights_only=False)
                before_bytes = (run/'latest.pt').read_bytes()
                enabled = {**cfg, 'speech_fidelity': SETTINGS, 'allow_speech_fidelity_on_resume': True}
                train(enabled, run, run/'latest.pt')
                after = torch.load(run/'latest.pt', weights_only=False)
                self.assertEqual((run/'before-speech-fidelity-step-00000002.pt').read_bytes(), before_bytes)
                self.assertEqual(after['step'], 3)
                self.assertEqual(after['config']['speech_fidelity'], SETTINGS)
                self.assertEqual(after['history'][-1]['flow_progress'], training_flow_progress(2, before['config']))
                self.assertEqual(after['history'][-1]['learning_rate_progress'], 2/20)
                self.assertEqual(after['config']['flow_schedule_transition']['start_step'], 1)
                self.assertEqual(after['config']['flow_schedule_transition']['ramp_updates'], 3)
                self.assertIn('pesq_loss', after['history'][-1])
                for state in after['optimizer']['state'].values():
                    self.assertEqual(state['step'].item(), 3)
                train(enabled, run, run/'latest.pt')
                last = torch.load(run/'latest.pt', weights_only=False)
                self.assertEqual(last['step'], 4)
                self.assertEqual(last['config']['speech_fidelity'], SETTINGS)
                self.assertIsNone(json.loads((run/'run.json').read_text())['speech_fidelity_transition'])


if __name__ == '__main__':
    unittest.main()
