"""Efficient architecture integration checks; no restoration-quality training runs."""
import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import soundfile as sf
import torch

from rmfsr.efficient import EfficientRMFSR
from rmfsr.flow import meanflow_loss, sample
from rmfsr.model import RMFSR, model_from_checkpoint, model_from_config
from rmfsr.profile import model_inventory


TINY_CONFIG = dict(model_type='efficient-v1', channels=[8, 8, 16, 16, 16],
                   groups=4, attention_rank=4, folded_width=16)


def tiny_model():
    model = model_from_config(copy.deepcopy(TINY_CONFIG)).eval()
    # Otherwise the identity-initialized output would hide internal defects.
    torch.nn.init.normal_(model.head.conv.weight, std=.01)
    return model


def training_fixture(root):
    manifests = {}
    for split, speaker, omega in [('train', 'p001', .1), ('validation', 'p002', .13)]:
        path = root / f'{split}.wav'
        sf.write(path, .1 * np.sin(np.arange(16000) * omega), 16000)
        manifest = root / f'{split}.json'
        manifest.write_text(json.dumps([dict(path=path.name, speaker=speaker, split=split,
                                             frames=16000, seconds=1, sr=16000)]))
        manifests[f'{split}_manifest'] = str(manifest)
    return dict(**TINY_CONFIG, **manifests, name='efficient-integration', pilot=True,
                sample_rate=16000, seconds=.1, batch_size=1, steps=1, schedule_steps=20,
                seed=7, device='cpu', threads=2, indexed_batches=True, num_workers=0,
                noise_dir=None, ltas_path=None, ffmpeg=None, learning_rate=.0001,
                warmup_steps=1, ema_decay=.95, gradient_clip=1, save_every=1,
                validate_every=1, validation_examples=1, validation_audio_examples=0,
                inference_evaluations_to_compare=[1])


class EfficientFactoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_dispatch_and_recommended_parameter_count(self):
        model = model_from_config({'model_type': 'efficient-v1'})
        self.assertIsInstance(model, EfficientRMFSR)
        self.assertEqual(sum(p.numel() for p in model.parameters()), 7742598)
        architecture = model.architecture()
        self.assertEqual(architecture['model_type'], 'efficient-v1')
        self.assertEqual(architecture['groups'], 16)
        self.assertEqual(architecture['attention_rank'], 16)
        self.assertEqual(architecture['folded_width'], 448)
        self.assertEqual(architecture['channels'], [64, 64, 128, 256, 256])
        self.assertTrue(architecture['decoder_before_upsample'])
        self.assertTrue(architecture['final_depthwise_frequency_refinement'])
        self.assertIsInstance(model_from_config({'channels': [8, 8, 16, 16, 16]}), RMFSR)
        self.assertNotIsInstance(model_from_config({'channels': [8, 8, 16, 16, 16]}), EfficientRMFSR)

    def test_unknown_type_and_inconsistent_checkpoint_config_are_rejected(self):
        with self.assertRaises(ValueError):
            model_from_config({**TINY_CONFIG, 'model_type': 'unknown-model'})
        model = tiny_model()
        checkpoint = dict(config=copy.deepcopy(TINY_CONFIG), architecture=model.architecture(),
                          ema=model.state_dict())
        checkpoint['config']['groups'] = 8
        with self.assertRaises(ValueError):
            model_from_checkpoint(checkpoint)
        checkpoint['config'] = copy.deepcopy(TINY_CONFIG)
        checkpoint['architecture'] = {**model.architecture(), 'model_type': 'unknown-model'}
        with self.assertRaises(ValueError):
            model_from_checkpoint(checkpoint)

    def test_serialized_checkpoint_restores_nonzero_outputs_and_buffers(self):
        torch.manual_seed(37)
        model = tiny_model()
        x = torch.randn(1, 2, 161, 7) * .05
        y = torch.randn_like(x) * .05
        t, r = torch.tensor([.7]), torch.tensor([.2])
        with torch.no_grad():
            expected = model(x, y, t, r)
        self.assertGreater(float((expected - y).abs().max()), 1e-5)
        payload = io.BytesIO()
        torch.save(dict(config=copy.deepcopy(TINY_CONFIG), architecture=model.architecture(),
                        ema=model.state_dict()), payload)
        payload.seek(0)
        checkpoint = torch.load(payload, map_location='cpu', weights_only=False)
        restored = model_from_checkpoint(checkpoint).eval()
        restored.load_state_dict(checkpoint['ema'], strict=True)
        self.assertIsInstance(restored, EfficientRMFSR)
        self.assertEqual(restored.architecture(), model.architecture())
        torch.testing.assert_close(restored.embedding.frequencies, model.embedding.frequencies,
                                   rtol=0, atol=0)
        with torch.no_grad():
            torch.testing.assert_close(restored(x, y, t, r), expected, rtol=0, atol=0)

    def test_historical_checkpoint_still_dispatches_to_original_legacy_model(self):
        original = RMFSR(channels=TINY_CONFIG['channels'], decoder_layout='legacy-v1')
        checkpoint = dict(config={'channels': TINY_CONFIG['channels']}, ema=original.state_dict())
        restored = model_from_checkpoint(checkpoint)
        self.assertNotIsInstance(restored, EfficientRMFSR)
        self.assertEqual(restored.decoder_layout, 'legacy-v1')
        restored.load_state_dict(checkpoint['ema'], strict=True)

    def test_compact_attention_mac_uses_projection_rank_for_both_matmuls(self):
        model = model_from_config(TINY_CONFIG)
        seconds = .02  # Two frames, enough to exercise the real forward hooks.
        actual = model_inventory(model, seconds=seconds)['gmac_by_stage']['enc_attention']
        # Each stage: QKV/output dense projections + QK^T + attention*V.
        # Rank=4 even where feature channels=16; using feature width here is wrong.
        frames, rank = 2, 4
        expected_macs = 0
        for channels, bins in [(8, 81), (8, 41), (16, 21), (16, 11), (16, 6)]:
            projections = frames * bins * (channels * 3 * rank + rank * channels)
            two_attention_products = 2 * frames * bins * bins * rank
            conditioning = 128 * channels + 128 * rank
            expected_macs += projections + two_attention_products + conditioning
        self.assertAlmostEqual(actual, expected_macs / seconds / 1e9, places=12)


class EfficientNumericalTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(23)
        self.model = tiny_model()
        self.x = torch.randn(1, 2, 161, 12) * .05
        self.y = torch.randn_like(self.x) * .05
        self.t, self.r = torch.tensor([.6]), torch.tensor([.2])

    def test_causal_cache_parity_and_frequency_specific_corrections(self):
        with torch.no_grad():
            full = self.model(self.x, self.y, self.t, self.r)
            changed_x, changed_y = self.x.clone(), self.y.clone()
            changed_x[..., 7:] += 10
            changed_y[..., 7:] -= 10
            changed = self.model(changed_x, changed_y, self.t, self.r)
            cache = {}
            chunked = torch.cat([self.model(self.x[..., i:i+5], self.y[..., i:i+5],
                                             self.t, self.r, cache=cache)
                                 for i in range(0, 12, 5)], dim=-1)
        torch.testing.assert_close(full[..., :7], changed[..., :7], rtol=0, atol=1e-6)
        torch.testing.assert_close(full, chunked, rtol=2e-4, atol=2e-5)
        correction = full - self.y
        pair_difference = correction[:, :, :160:2] - correction[:, :, 1:160:2]
        self.assertGreater(float(pair_difference.abs().max()), 1e-6)

    def test_ode_steps_use_independent_caches(self):
        noise = torch.randn_like(self.y)
        for steps in (1, 2, 5):
            with self.subTest(steps=steps):
                whole = sample(self.model, self.y, steps, noise=noise)
                chunked = sample(self.model, self.y, steps, noise=noise, chunk_frames=5)
                torch.testing.assert_close(whole, chunked, rtol=5e-4, atol=5e-5)

    def test_offdiagonal_jvp_backprop_reaches_folded_bottleneck(self):
        x, y = self.x[..., :3], self.y[..., :3]
        loss, _ = meanflow_loss(self.model, x, y, t=self.t, r=self.r, noise=torch.zeros_like(x))
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        gradients = [p.grad for p in self.model.parameters() if p.grad is not None]
        self.assertTrue(gradients)
        self.assertTrue(all(torch.isfinite(g).all() for g in gradients))
        folded = [p.grad for name, p in self.model.named_parameters()
                  if name.startswith('folded.') and p.grad is not None]
        self.assertTrue(folded)
        self.assertTrue(any(torch.count_nonzero(g).item() for g in folded))

    def test_invalid_spectral_width_is_rejected(self):
        with self.assertRaises(ValueError):
            self.model(self.x[:, :, :160], self.y[:, :, :160], self.t, self.r)


class EfficientTrainingIntegrationTests(unittest.TestCase):
    def test_two_update_save_resume_and_evaluation_dispatch(self):
        from rmfsr.evaluate import evaluate
        from rmfsr.train import train

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            cfg = training_fixture(root)
            run = root / 'run'
            # Only two tiny optimizer updates. Validation quality is outside this
            # wiring test; evaluate() below exercises real checkpoint inference.
            with patch('rmfsr.train.validate', return_value={'validation_restoration_mse_nfe1': .5}), \
                 patch('torch.backends.mps.is_available', return_value=False), \
                 contextlib.redirect_stdout(io.StringIO()):
                train(cfg, run)
                first = torch.load(run / 'latest.pt', map_location='cpu', weights_only=False)
                self.assertEqual(first['step'], 1)
                self.assertEqual(first['architecture']['model_type'], 'efficient-v1')
                self.assertEqual(first['architecture']['folded_width'], 16)
                train({**cfg, 'steps': 2}, run, run / 'latest.pt')
            checkpoint = torch.load(run / 'latest.pt', map_location='cpu', weights_only=False)
            self.assertEqual(checkpoint['step'], 2)
            self.assertEqual(checkpoint['architecture'], model_from_config(cfg).architecture())
            self.assertTrue(checkpoint['optimizer']['state'])
            self.assertTrue(any(not torch.equal(first['model'][k], checkpoint['model'][k])
                                for k in first['model']))
            with patch('rmfsr.train.validate', return_value={'validation_restoration_mse_nfe1': .5}), \
                 contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaises(ValueError):
                    train({**cfg, 'steps': 3, 'groups': 8}, run, run / 'latest.pt')

            case = root / 'baselines' / 'opus_20ms'
            case.mkdir(parents=True)
            reference = (.1 * np.sin(np.arange(1600) * .1)).astype(np.float32)
            damaged = reference * .5
            for name in ('reference', 'zero_filled', 'opus_classic', 'opus_deep', 'tplc_s', 'tplc_l'):
                sf.write(case / f'{name}.wav', reference if name == 'reference' else damaged,
                         16000, subtype='FLOAT')
            np.save(case / 'missing_mask.npy', np.zeros(len(reference), dtype=bool))
            # Short synthetic fixtures do not have enough speech for valid STOI;
            # this test targets evaluation dispatch/alignment, not that metric.
            with patch('rmfsr.evaluate.stoi', return_value=.5), contextlib.redirect_stdout(io.StringIO()):
                report = evaluate(run / 'latest.pt', root / 'baselines', root / 'evaluation',
                                  device='cpu', steps=(1,), chunk_frames=3)
            self.assertEqual(report['architecture']['model_type'], 'efficient-v1')
            self.assertEqual(report['training_steps'], 2)
            self.assertEqual(len(report['results']), 1)
            result_path = root / 'evaluation' / 'opus_20ms' / 'rmfsr_nfe1.wav'
            waveform, sr = sf.read(result_path)
            self.assertEqual(sr, 16000)
            self.assertEqual(waveform.shape, reference.shape)
            self.assertTrue(np.isfinite(waveform).all())


if __name__ == '__main__':
    unittest.main()
