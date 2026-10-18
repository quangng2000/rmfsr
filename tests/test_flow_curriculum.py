import contextlib
import io
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import torch

from rmfsr.flow import schedule_values, sample_times
from rmfsr.flow_curriculum import config_digest, resolve_transition, training_flow_progress
from rmfsr.train import train
from tests.test_training_integration import fixture


def request_for(config, ramp=10000):
    return dict(version=1, expected_config_sha256=config_digest(config),
                ramp_updates=ramp, target_progress=.8)


class FlowCurriculumTests(unittest.TestCase):
    def test_ramp_is_continuous_and_reaches_broad_spans_independent_of_lr_clock(self):
        cfg = dict(steps=300000, schedule_steps=300000, flow_schedule='figure1-cosine-approx')
        transition = resolve_transition(cfg, cfg, 9000, request_for(cfg))
        active = {**cfg, 'flow_schedule_transition': transition}
        self.assertEqual(training_flow_progress(9000, active), training_flow_progress(9000, cfg))
        self.assertEqual(training_flow_progress(19000, active), .8)
        self.assertEqual(training_flow_progress(29000, active), .8)
        self.assertEqual(active['schedule_steps'], 300000)
        samples = [schedule_values(training_flow_progress(step, active), cfg['flow_schedule'])
                   for step in range(9000, 19001, 100)]
        self.assertTrue(all(a[0] >= b[0] and a[1] <= b[1] for a, b in zip(samples, samples[1:])))
        self.assertEqual(samples[-1], (.2, 1.))
        torch.manual_seed(51)
        t, r = sample_times(100000, training_flow_progress(19000, active), 'cpu', cfg['flow_schedule'])
        self.assertAlmostEqual(float((t == r).float().mean()), .2, delta=.01)
        self.assertGreater(float(((t-r)/t >= .5).float().mean()), .38)

    def test_request_cannot_silently_change_recipe_or_checkpointed_clock(self):
        cfg = dict(steps=300000, schedule_steps=300000, flow_schedule='figure1-cosine-approx')
        request = request_for(cfg)
        with self.assertRaisesRegex(ValueError, 'different configuration'):
            resolve_transition({**cfg, 'steps': 310000}, cfg, 9000, request)
        policy = resolve_transition(cfg, cfg, 9000, request)
        previous = {**cfg, 'flow_schedule_transition': policy}
        self.assertEqual(resolve_transition(cfg, previous, 12000), policy)
        with self.assertRaisesRegex(ValueError, 'differs from checkpoint'):
            resolve_transition(cfg, previous, 12000, {**request, 'ramp_updates': 20000})
        for ramp in (0, -1, True, 1.5):
            with self.assertRaises(ValueError):
                resolve_transition(cfg, cfg, 9000, {**request, 'ramp_updates': ramp})

    def test_session_resume_keeps_weights_ema_optimizer_and_schedule_exact(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cfg = {**fixture(root), 'steps': 5, 'flow_schedule': 'figure1-cosine-approx',
                   'max_wall_seconds': .0001}
            validate = {'validation_restoration_mse_nfe1': .2}
            with patch('rmfsr.train.validate', return_value=validate), contextlib.redirect_stdout(io.StringIO()):
                train(cfg, root/'seed')
                self.assertEqual(torch.load(root/'seed/latest.pt', weights_only=False)['step'], 1)
                for name in ('straight', 'split'):
                    run = root/name
                    run.mkdir()
                    shutil.copy2(root/'seed/latest.pt', run/'latest.pt')
                    (run/'flow-schedule-transition.json').write_text(json.dumps(request_for(cfg, ramp=3)))
                with patch('rmfsr.train.time.perf_counter', return_value=0.):
                    train(cfg, root/'straight', root/'straight/latest.pt')
                for _ in range(4):
                    train(cfg, root/'split', root/'split/latest.pt')
            a = torch.load(root/'straight/latest.pt', weights_only=False)
            b = torch.load(root/'split/latest.pt', weights_only=False)
            self.assertEqual(a['config']['flow_schedule_transition'], b['config']['flow_schedule_transition'])
            self.assertEqual(b['config']['flow_schedule_transition']['start_step'], 1)
            self.assertEqual(b['history'][-1]['flow_progress'], .8)
            self.assertEqual(b['history'][-1]['learning_rate_progress'], 4/20)
            self.assertEqual(a['history'][-1]['loss'], b['history'][-1]['loss'])
            self.assertEqual(a['history'][-1]['lr'], b['history'][-1]['lr'])
            for field in ('model', 'ema'):
                for key in a[field]:
                    torch.testing.assert_close(a[field][key], b[field][key], rtol=0, atol=0)
            for key, state in a['optimizer']['state'].items():
                for name, value in state.items():
                    torch.testing.assert_close(value, b['optimizer']['state'][key][name], rtol=0, atol=0)
            self.assertEqual((root/'seed/latest.pt').read_bytes(),
                             (root/'split/before-flow-schedule-step-00000001.pt').read_bytes())
            self.assertEqual(json.loads((root/'split/flow-schedule-active.json').read_text()),
                             b['config']['flow_schedule_transition'])
            self.assertNotIn('flow_schedule_transition', cfg)


if __name__ == '__main__':
    unittest.main()
