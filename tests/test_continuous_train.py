import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from rmfsr.continuous_train import main, qualification_passed, run_sessions


class ContinuousTrainingTests(unittest.TestCase):
    def test_explicit_direct_launch_bypasses_qualification(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / 'config.json'
            config.write_text(json.dumps({'steps': 300000, 'pilot': False}))
            with patch('sys.argv', ['continuous_train', '--config', str(config), '--control',
                                   str(root / 'control'), '--skip-qualification']), \
                 patch('rmfsr.continuous_train.qualification_passed', side_effect=AssertionError('benchmark called')), \
                 patch('rmfsr.continuous_train.run_sessions') as sessions:
                main()
            sessions.assert_called_once()

    def test_resume_uses_checkpoint_and_stops_at_target(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / 'config.json'
            config.write_text(json.dumps({'steps': 3, 'pilot': False}))
            run, control = root / 'run', root / 'control'
            calls = []
            def execute(command, **kwargs):
                calls.append(command)
                step = len(calls)
                (run / 'latest.pt').write_bytes(b'checkpoint fixture')
                (run / 'run.json').write_text(json.dumps(dict(completed_steps=step,
                    status='training_complete_not_evaluated' if step == 3 else 'paused_checkpointed')))
            run_sessions(config, run, control, execute)
            self.assertEqual(len(calls), 3)
            self.assertNotIn('--resume', calls[0])
            self.assertIn('--resume', calls[1])
            self.assertIn('--resume', calls[2])
            self.assertEqual(json.loads((control / 'status.json').read_text())['phase'], 'complete')
            run_sessions(config, run, control, execute)
            self.assertEqual(len(calls), 3)

    def test_no_progress_is_an_error_instead_of_an_infinite_loop(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / 'config.json'
            config.write_text(json.dumps({'steps': 3}))
            run = root / 'run'
            def execute(command, **kwargs):
                (run / 'latest.pt').write_bytes(b'checkpoint fixture')
                (run / 'run.json').write_text(json.dumps(dict(completed_steps=1, status='paused_checkpointed')))
            with self.assertRaisesRegex(RuntimeError, 'without advancing'):
                run_sessions(config, run, root / 'control', execute)

    def test_failed_qualification_never_starts_training(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'status').write_text('failed:1')
            with self.assertRaisesRegex(RuntimeError, 'Qualification failed'):
                qualification_passed(root, {})

    def test_only_completed_matching_qualification_passes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'real-data').mkdir()
            (root / 'status').write_text('complete')
            config = dict(model_type='efficient-v1', batch_size=4, accumulation_steps=4)
            record = dict(status='training_complete_not_evaluated', completed_steps=100,
                          config={**config, 'steps': 100, 'benchmark_only': True, 'pilot': False})
            manifest = root / 'real-data/run.json'
            manifest.write_text(json.dumps(record))
            (root / 'throughput.json').write_text(json.dumps({'mean_seconds_per_update': 1.5}))
            self.assertTrue(qualification_passed(root, config))
            with self.assertRaisesRegex(RuntimeError, 'recipe differs'):
                qualification_passed(root, {**config, 'batch_size': 8})
            record['completed_steps'] = 50
            manifest.write_text(json.dumps(record))
            with self.assertRaisesRegex(RuntimeError, 'did not finish'):
                qualification_passed(root, config)


if __name__ == '__main__':
    unittest.main()
