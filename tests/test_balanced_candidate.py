"""CPU checks for the opt-in dense, non-folded candidate; no quality claims."""
import copy
import io
import json
from pathlib import Path
import unittest

import torch

from rmfsr.flow import meanflow_loss, sample
from rmfsr.model import model_from_checkpoint, model_from_config


ROOT = Path(__file__).resolve().parents[1]


class BalancedCandidateTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(408)
        self.config = json.loads((ROOT / 'configs/balanced-candidate.json').read_text())
        self.model = model_from_config(self.config).cpu().eval()
        self.y = torch.randn(1, 2, 161, 9) * .05
        self.noise = torch.randn_like(self.y)

    def test_identity_start_then_nontrivial_causal_streaming(self):
        # The zero head should preserve damaged coefficients at initialization.
        identity = sample(self.model, self.y, 5, noise=self.noise)
        torch.testing.assert_close(identity, self.y, atol=1e-7, rtol=1e-6)
        # Exercise the entire network, so a zero head cannot mask cache defects.
        torch.nn.init.normal_(self.model.head.conv.weight, std=.005)
        for nfe in (1, 5):
            with self.subTest(nfe=nfe):
                full = sample(self.model, self.y, nfe, noise=self.noise)
                streamed = sample(self.model, self.y, nfe, noise=self.noise, chunk_frames=4)
                changed = self.y.clone()
                changed[..., 5:] += .5
                future = sample(self.model, changed, nfe, noise=self.noise)
                torch.testing.assert_close(full, streamed, atol=5e-5, rtol=5e-4)
                torch.testing.assert_close(full[..., :5], future[..., :5], atol=1e-6, rtol=0)
                self.assertGreater(float((full - self.y).abs().max()), 1e-6)

    def test_meanflow_updates_checkpoint_resume_and_legacy_rejection(self):
        # First update opens the zero head; the second exercises off-diagonal
        # JVPs and gradients through the actual full-size spatial TCN.
        model = self.model.train()
        clean = self.y[..., :3] * 1.3
        degraded = self.y[..., :3]
        noise = self.noise[..., :3]
        t, r = torch.tensor([.7]), torch.tensor([.2])
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

        def update(network, optim, end):
            optim.zero_grad(set_to_none=True)
            loss, _ = meanflow_loss(network, clean, degraded, t=t, r=end, noise=noise)
            self.assertTrue(torch.isfinite(loss))
            loss.backward()
            gradients = [p.grad for p in network.parameters() if p.grad is not None]
            self.assertTrue(all(torch.isfinite(g).all() for g in gradients))
            optim.step()
            return float(loss.detach())

        update(model, optimizer, t)
        update(model, optimizer, r)
        for component in (model.encoder, model.tcn, model.decoder, model.enc_attention):
            self.assertTrue(any(p.grad is not None and bool(p.grad.abs().sum() > 0)
                                for p in component.parameters()))

        payload = io.BytesIO()
        torch.save(dict(config=self.config, architecture=model.architecture(),
                        model=model.state_dict(), optimizer=optimizer.state_dict()), payload)
        payload.seek(0)
        saved = torch.load(payload, map_location='cpu', weights_only=False)
        restored = model_from_checkpoint(saved).cpu().train()
        restored.load_state_dict(saved['model'], strict=True)
        resumed_optimizer = torch.optim.AdamW(restored.parameters(), lr=1e-4)
        resumed_optimizer.load_state_dict(saved['optimizer'])
        uninterrupted_loss = update(model, optimizer, r)
        resumed_loss = update(restored, resumed_optimizer, r)
        self.assertAlmostEqual(uninterrupted_loss, resumed_loss, places=7)
        for name, tensor in model.state_dict().items():
            torch.testing.assert_close(tensor, restored.state_dict()[name], atol=1e-7, rtol=1e-6)

        mismatched = copy.deepcopy(saved)
        mismatched['config'].update(groups=16, attention_rank=16, folded_width=448)
        with self.assertRaises(ValueError):
            model_from_checkpoint(mismatched)


if __name__ == '__main__':
    unittest.main()
