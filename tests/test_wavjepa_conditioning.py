"""CPU integration checks; no pretrained download or corpus scan in unit tests."""
import copy
import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from rmfsr.flow import meanflow_loss, sample
from rmfsr.flow_curriculum import config_digest, training_flow_progress
from rmfsr.model import model_from_config, model_from_checkpoint
from rmfsr.spectral import Spectral
from rmfsr.train import backward_microbatches, train
from rmfsr.validation import validate_restoration
from rmfsr.wavjepa import DEFAULT_OPTIONS, FrozenWavJEPA, align_features
from rmfsr.wavjepa_fork import conditioned_checkpoint, fork_checkpoint
from test_training_integration import fixture


def config():
    return dict(model_type='efficient-v1', channels=[8, 8, 16, 16, 16],
                groups=1, attention_rank=4, folded_width=0, sample_rate=16000,
                seed=42, steps=300000, learning_rate=1e-4,
                flow_schedule='figure1-cosine-approx', validation_chunk_frames=10)


class FakeEncoder:
    def __init__(self):
        self.inputs = []

    def __call__(self, damaged, spectral, frames):
        self.inputs.append(damaged.clone())
        return torch.arange(768, dtype=torch.float32)[None, :, None].expand(damaged.shape[0], -1, frames).clone()

    def metadata(self):
        return {'test_fixture': True}


class WavJEPAConditioningTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(71)
        self.cfg = config()
        self.old = model_from_config(self.cfg)
        # Avoid the initial identity head masking broken bottleneck integration.
        nn.init.normal_(self.old.head.conv.weight, std=.01)
        self.optimizer = torch.optim.AdamW(self.old.parameters(), lr=1e-4)
        self.y = torch.randn(1, 2, 161, 4) * .03
        self.clean, self.noise = self.y * 1.2, torch.randn_like(self.y)
        self.t, self.r = torch.tensor([.7]), torch.tensor([.2])
        loss, _ = meanflow_loss(self.old, self.clean, self.y, t=self.t, r=self.t, noise=self.noise)
        loss.backward()
        self.optimizer.step()
        self.checkpoint = dict(config=self.cfg, architecture=self.old.architecture(), step=65000,
                               model=copy.deepcopy(self.old.state_dict()), ema=copy.deepcopy(self.old.state_dict()),
                               optimizer=copy.deepcopy(self.optimizer.state_dict()), best_validation=.02)
        self.payload = conditioned_checkpoint(self.checkpoint)
        self.new = model_from_checkpoint(self.payload)
        self.new.load_state_dict(self.payload['model'])
        self.c = torch.randn(1, 768, 4)

    def test_zero_fusion_preserves_full_model_and_meanflow_target(self):
        for end in (self.t, self.r):
            a, _ = meanflow_loss(self.old, self.clean, self.y, t=self.t, r=end, noise=self.noise)
            b, _ = meanflow_loss(self.new, self.clean, self.y, t=self.t, r=end, noise=self.noise, conditioning=self.c)
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        for nfe in (1, 5):
            a = sample(self.old, self.y, nfe, noise=self.noise)
            b = sample(self.new, self.y, nfe, noise=self.noise, conditioning=self.c)
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        self.assertFalse(self.new.architecture()['conditioning_is_causal'])

    def test_optimizer_resume_adapter_learns_and_checkpoint_roundtrips(self):
        optimizer = torch.optim.AdamW(self.new.parameters(), lr=1e-4)
        optimizer.load_state_dict(self.payload['optimizer'])
        for name, parameter in self.old.named_parameters():
            other = dict(self.new.named_parameters())[name]
            for key, value in self.optimizer.state[parameter].items():
                torch.testing.assert_close(value, optimizer.state[other][key], rtol=0, atol=0)
        for _ in range(2):
            optimizer.zero_grad(set_to_none=True)
            loss, _ = meanflow_loss(self.new, self.clean, self.y, t=self.t, r=self.r,
                                    noise=self.noise, conditioning=self.c)
            loss.backward()
            for p in self.new.parameters():
                if p.grad is not None:
                    self.assertTrue(bool(torch.isfinite(p.grad).all()))
            optimizer.step()
        self.assertGreater(self.new.conditioning.project[1].weight.grad.abs().sum().item(), 0)
        a = sample(self.new, self.y, 5, noise=self.noise, conditioning=self.c)
        b = sample(self.new, self.y, 5, noise=self.noise, conditioning=-self.c)
        self.assertGreater((a-b).abs().max().item(), 0)
        restored = model_from_checkpoint(self.payload)
        restored.load_state_dict(self.new.state_dict(), strict=True)
        c = sample(restored, self.y, 5, noise=self.noise, conditioning=self.c)
        torch.testing.assert_close(a, c, rtol=0, atol=0)

    def test_rejects_missing_features_streaming_and_unpinned_model(self):
        with self.assertRaisesRegex(ValueError, 'requires features'):
            sample(self.new, self.y)
        with self.assertRaisesRegex(ValueError, 'offline'):
            sample(self.new, self.y, conditioning=self.c, chunk_frames=2)
        with self.assertRaisesRegex(ValueError, 'time-aligned'):
            sample(self.new, self.y, conditioning=self.c[..., :-1])
        for bad in ({**self.cfg, 'wavjepa': {}},
                    {**self.payload['config'], 'wavjepa': {'revision': 'main'}}):
            with self.assertRaises(ValueError):
                model_from_config(bad)

    def test_checkpoint_fork_preserves_original_flow_clock(self):
        previous = copy.deepcopy(self.checkpoint)
        cfg = previous['config']
        digest = config_digest(cfg)
        cfg['flow_schedule_transition'] = dict(
            version=1, start_step=10000, ramp_updates=1000, start_progress=10000/300000,
            target_progress=.8, request=dict(expected_config_sha256=digest),
            restoration=dict(after_step=18491, request=dict(version=1, mode='original-gradual-clock',
                                                            expected_config_sha256=digest)))
        migrated = conditioned_checkpoint(previous)
        self.assertEqual(migrated['step'], 65000)
        self.assertEqual(migrated['config']['steps'], 300000)
        for step in (65000, 100000):
            self.assertEqual(training_flow_progress(step, cfg), training_flow_progress(step, migrated['config']))
        self.assertNotIn('wavjepa', cfg)

    def test_encoder_runs_once_per_microbatch_and_validation_clip(self):
        spectral, encoder = Spectral(), FakeEncoder()
        clean = torch.sin(torch.arange(640)[None] * .1) * .1
        damaged = clean * .6
        backward_microbatches(self.new, iter([(clean, damaged)]), spectral, 'cpu', .6, 1,
                              schedule='figure1-cosine-approx', conditioner=encoder)
        self.assertEqual(len(encoder.inputs), 1)
        torch.testing.assert_close(encoder.inputs[0], damaged)
        validation = type('Pairs', (), {})()
        validation.rng = np.random.default_rng(3)
        validation.one = lambda: (clean[0].numpy(), damaged[0].numpy(), np.zeros(640, dtype=bool), ['test'])
        cfg = {**self.payload['config'], 'validation_examples': 1, 'inference_evaluations_to_compare': [1, 5]}
        report = validate_restoration(self.new, validation, spectral, cfg, 'cpu', conditioner=encoder)
        self.assertEqual(len(encoder.inputs), 2)
        self.assertTrue(np.isfinite(report['validation_restoration_mse_nfe5']))

    def test_physical_time_alignment_and_ola_tail(self):
        spectral = Spectral()
        # Linear-valued features must interpolate by time, not frame-count resize.
        positions = torch.tensor([119.5, 279.5, 439.5, 759.5])
        features = positions[None, :, None].expand(1, -1, 768)
        result = align_features(features, positions, spectral, 8)
        expected = (torch.arange(8) * 160 - .5).clamp(119.5, 759.5)
        torch.testing.assert_close(result[0, 0], expected)

    def test_full_trainer_fork_and_resume_keeps_old_run_intact(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cfg = {**fixture(root), 'model_type': 'efficient-v1', 'groups': 1,
                   'attention_rank': 4, 'folded_width': 0, 'steps': 4,
                   'max_wall_seconds': .0001, 'input_verification': 'manifest',
                   'flow_schedule': 'figure1-cosine-approx'}
            with contextlib.redirect_stdout(io.StringIO()):
                train(cfg, root/'old')
                before = (root/'old/latest.pt').read_bytes()
                fork_checkpoint(root/'old/latest.pt', root/'conditioned')
                checkpoint = torch.load(root/'conditioned/latest.pt', weights_only=False)
                self.assertEqual(checkpoint['step'], 1)
                encoder = FakeEncoder()
                with patch('rmfsr.wavjepa.build_conditioner', return_value=encoder):
                    train(checkpoint['config'], root/'conditioned', root/'conditioned/latest.pt')
                    train(checkpoint['config'], root/'conditioned', root/'conditioned/latest.pt')
            saved = torch.load(root/'conditioned/latest.pt', weights_only=False)
            self.assertEqual(saved['step'], 3)
            self.assertEqual(saved['history'][-1]['flow_progress'], 2/20)
            self.assertEqual(len(encoder.inputs), 4)  # train + validation for each update
            self.assertEqual((root/'old/latest.pt').read_bytes(), before)
            self.assertGreater(saved['model']['conditioning.fuse.weight'].abs().sum().item(), 0)
            with self.assertRaisesRegex(ValueError, 'nonexistent'):
                fork_checkpoint(root/'old/latest.pt', root/'conditioned')

    def test_frozen_frontend_handles_short_exact_boundary_and_multiple_segments(self):
        class Core(nn.Module):
            def __init__(self):
                super().__init__()
                self.anchor = nn.Parameter(torch.ones(()))
                self.extract_audio = type('Frontend', (), {'conv_layers_spec': [(512, 10, 5)] + [(512, 3, 2)]*4 + [(512, 2, 2)]})()
                self.target_length, self.total_patches = 32159, 200
                self.calls = 0
                for key in ('teacher_encoder', 'decoder', 'decoder_to_encoder_mapper',
                            'encoder_to_decoder_mapper', 'mask_token', 'pos_encoding_decoder'):
                    setattr(self, key, None)

            @torch.inference_mode()
            def _get_segment_representation(self, segment, mask):
                self.calls += 1
                assert not mask.all()
                return segment.mean(-1, keepdim=True).expand(-1, 200, 768) + self.anchor

        upstream = type('Upstream', (), {'model': Core()})()
        # Tests require no Transformers installation; only the loader is replaced.
        module = type('Transformers', (), {'AutoModel': type('Auto', (), {
            'from_pretrained': staticmethod(lambda *args, **kwargs: upstream)})})()
        with patch.dict('sys.modules', {'transformers': module}):
            encoder = FrozenWavJEPA()
        encoder.train(True)
        self.assertFalse(encoder.training)
        self.assertTrue(all(not p.requires_grad for p in encoder.parameters()))
        spectral = Spectral()
        for length, calls in ((1, 1), (32159, 1), (32160, 2), (64000, 2)):
            encoder.core.calls = 0
            wave = torch.zeros(2, length)
            frames = spectral.encode(wave).shape[-1]
            result = encoder(wave, spectral, frames)
            self.assertEqual(tuple(result.shape), (2, 768, frames))
            self.assertEqual(encoder.core.calls, calls)
            self.assertFalse(result.is_inference())
            self.assertTrue(bool(torch.isfinite(result).all()))


if __name__ == '__main__':
    unittest.main()
