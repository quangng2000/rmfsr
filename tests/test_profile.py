"""Regression checks for optimizer-update throughput accounting."""
import unittest
from unittest.mock import patch

import torch

from rmfsr.profile import profile


class ProfileTests(unittest.TestCase):
    def test_accumulated_update_averages_gradients_and_steps_once(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor(1.0))

            def architecture(self):
                return {'model_type': 'fixture'}

        model = Model()
        gradients = []
        def optimizer_step(optimizer, *args, **kwargs):
            gradients.append(model.weight.grad.item())

        with patch('rmfsr.profile._profile_model', return_value=model), \
             patch('rmfsr.profile.meanflow_loss', side_effect=lambda *a, **kw: (model.weight * 2, {})) as loss, \
             patch('rmfsr.profile.model_inventory', return_value={'gmac_per_audio_second_per_evaluation': 0}), \
             patch.object(torch.optim.AdamW, 'step', optimizer_step):
            report = profile(device='cpu', batch=4, seconds=.01, repeat=2,
                             config={'accumulation_steps': 4})
        self.assertEqual(loss.call_count, 12)  # Warm-up + two updates, four microbatches each.
        self.assertEqual(gradients, [2.0, 2.0, 2.0])
        self.assertEqual(report['effective_batch_size'], 16)
        self.assertEqual(len(report['training_step_seconds']), 2)
        self.assertAlmostEqual(report['optimizer_updates_per_second'] * report['median_seconds'], 1)


if __name__ == '__main__':
    unittest.main()
