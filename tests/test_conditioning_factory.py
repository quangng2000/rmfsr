"""Production opt-in, checkpoint reload and full MeanFlow differentiation."""
import copy
import unittest

import torch

from rmfsr.conditioning import ARCHITECTURE, inspect
from rmfsr.conditioning_blocks import DoubleNormalizedConv
from rmfsr.conditioning_embedding import PositionalTimeEmbedding
from rmfsr.flow import meanflow_loss
from rmfsr.model import ConditionedConv, model_from_config, model_from_checkpoint


TINY_CONFIG = dict(model_type='efficient-v1', channels=[8, 8, 16, 16, 16],
                   groups=4, attention_rank=4, folded_width=16)


class ConditioningFactoryTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(25)

    def test_legacy_default_has_no_new_parameters_or_architecture_marker(self):
        model = model_from_config(TINY_CONFIG)
        self.assertNotIn('conditioning_architecture', model.architecture())
        self.assertNotIn('_scm_architecture_version', model.state_dict())
        self.assertIsInstance(model.stem, ConditionedConv)
        self.assertFalse(any(isinstance(layer, DoubleNormalizedConv)
                             for layer in model.modules()))

    def test_factory_installs_all_layers_and_explicit_checkpoint_metadata(self):
        config = {**TINY_CONFIG, 'conditioning_architecture': ARCHITECTURE}
        model = model_from_config(config)
        self.assertIsInstance(model.embedding, PositionalTimeEmbedding)
        self.assertEqual(model.architecture()['conditioning_architecture'], ARCHITECTURE)
        self.assertEqual(inspect(model)['layers'], 74)
        self.assertFalse(any(isinstance(layer, ConditionedConv) for layer in model.modules()))
        checkpoint = dict(config=config, architecture=model.architecture(),
                          model=copy.deepcopy(model.state_dict()))
        restored = model_from_checkpoint(checkpoint)
        restored.load_state_dict(checkpoint['model'], strict=True)
        self.assertEqual(restored.architecture(), checkpoint['architecture'])
        self.assertEqual(inspect(restored), inspect(model))
        for key, value in model.state_dict().items():
            torch.testing.assert_close(restored.state_dict()[key], value, rtol=0, atol=0)

    def test_unknown_and_mismatched_architecture_are_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Unknown conditioning_architecture'):
            model_from_config({**TINY_CONFIG, 'conditioning_architecture': 'unknown'})
        config = {**TINY_CONFIG, 'conditioning_architecture': ARCHITECTURE}
        model = model_from_config(config)
        with self.assertRaisesRegex(ValueError, 'architecture differs'):
            model_from_checkpoint(dict(config=TINY_CONFIG, architecture=model.architecture()))
        with self.assertRaisesRegex(ValueError, 'architecture differs'):
            model_from_checkpoint(dict(config=config,
                                       architecture=model_from_config(TINY_CONFIG).architecture()))

    def test_baseline_variant_checkpoint_reloads_and_retains_marker(self):
        config = dict(channels=[4], encoder_dilations=[1], tcn_dilations=[1],
                      conditioning_architecture=ARCHITECTURE)
        model = model_from_config(config)
        checkpoint = dict(config=config, architecture=model.architecture(),
                          model=model.state_dict())
        restored = model_from_checkpoint(checkpoint)
        restored.load_state_dict(checkpoint['model'], strict=True)
        self.assertEqual(restored.architecture(), model.architecture())

    def test_full_meanflow_backward_reaches_new_scales(self):
        config = {**TINY_CONFIG, 'conditioning_architecture': ARCHITECTURE}
        model = model_from_config(config)
        torch.nn.init.normal_(model.head.conv.weight, std=.01)
        clean = torch.randn(1, 2, 161, 3) * .05
        damaged = torch.randn_like(clean) * .05
        loss, _ = meanflow_loss(model, clean, damaged,
                                t=torch.tensor([.8]), r=torch.tensor([.1]),
                                noise=torch.zeros_like(clean))
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        gradients = [p.grad for p in model.parameters() if p.grad is not None]
        self.assertTrue(all(torch.isfinite(value).all() for value in gradients))
        scale_gradients = [parameter.grad for name, parameter in model.named_parameters()
                           if '.scale.' in name]
        self.assertTrue(scale_gradients)
        self.assertTrue(all(value is not None for value in scale_gradients))
        self.assertTrue(any(value.count_nonzero().item() for value in scale_gradients))


if __name__ == '__main__':
    unittest.main()
