import copy
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from rmfsr.flow_curriculum import config_digest
from rmfsr.frequency_transition import (apply_frequency_transition, publish_best_checkpoint,
                                        tensor_sha256)


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Module()
        self.embedding.register_buffer('frequencies', torch.arange(32, dtype=torch.float32) / 16)
        self.weight = torch.nn.Parameter(torch.tensor([.5, -.25]))


def fixture(run):
    model = TinyModel()
    ema = copy.deepcopy(model).requires_grad_(False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.0001)
    model.weight.square().sum().backward()
    optimizer.step()
    checkpoint = dict(model=copy.deepcopy(model.state_dict()), ema=copy.deepcopy(ema.state_dict()),
                      step=100, config={'steps': 300000, 'schedule_steps': 300000},
                      best_validation=.03, validation_selection_metric='validation_restoration_mse_nfe1',
                      optimizer=copy.deepcopy(optimizer.state_dict()),
                      torch_rng=torch.get_rng_state(), numpy_rng=np.random.get_state(),
                      python_rng=random.getstate(), history=[{'step': 100, 'loss': .2}])
    request = dict(version=1, transition_id='width4-approved', from_width=1, to_width=4,
                   expected_config_sha256=config_digest(checkpoint['config']),
                   minimum_checkpoint_step=90,
                   source_frequency_sha256={kind: tensor_sha256(checkpoint[kind]['embedding.frequencies'])
                                            for kind in ('model', 'ema')},
                   target_frequency_sha256={kind: tensor_sha256(checkpoint[kind]['embedding.frequencies'] * 4)
                                            for kind in ('model', 'ema')})
    path = run / 'latest.pt'
    torch.save(checkpoint, path)
    (run / 'best-validation.pt').write_bytes(b'old width-1 best checkpoint')
    return model, ema, optimizer, checkpoint, request, path


class FrequencyTransitionTests(unittest.TestCase):
    def assert_nested_equal(self, actual, expected):
        if isinstance(expected, torch.Tensor):
            self.assertTrue(torch.equal(actual, expected))
        elif isinstance(expected, np.ndarray):
            np.testing.assert_array_equal(actual, expected)
        elif isinstance(expected, dict):
            self.assertEqual(set(actual), set(expected))
            for key in expected:
                self.assert_nested_equal(actual[key], expected[key])
        elif isinstance(expected, (tuple, list)):
            self.assertEqual(type(actual), type(expected))
            self.assertEqual(len(actual), len(expected))
            for item, reference in zip(actual, expected):
                self.assert_nested_equal(item, reference)
        else:
            self.assertEqual(actual, expected)

    def test_changes_only_two_buffers_and_archives_source_and_best(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            model, ema, optimizer, checkpoint, request, path = fixture(run)
            saved = copy.deepcopy(checkpoint)
            states = [torch.get_rng_state().clone(), copy.deepcopy(np.random.get_state()),
                      random.getstate(), copy.deepcopy(optimizer.state_dict())]
            source_bytes = path.read_bytes()
            best_bytes = (run / 'best-validation.pt').read_bytes()
            marker, applied = apply_frequency_transition(model, ema, checkpoint, request, run, path)
            self.assertTrue(applied)
            self.assertEqual(marker['after_step'], 100)
            self.assertEqual(marker['nominal_frequency_width'], 4)
            self.assertTrue(marker['best_selection_pending'])
            self.assertEqual(marker['previous_best_validation'], .03)
            for kind, module in (('model', model), ('ema', ema)):
                self.assertTrue(torch.equal(module.embedding.frequencies,
                                            saved[kind]['embedding.frequencies'] * 4))
                self.assertTrue(torch.equal(module.weight, saved[kind]['weight']))
                self.assertEqual(module.frequency_width_transition, marker)
            self.assert_nested_equal(checkpoint, saved)
            self.assert_nested_equal([torch.get_rng_state(), np.random.get_state(), random.getstate(),
                                      optimizer.state_dict()], states)
            self.assertEqual(path.read_bytes(), source_bytes)
            self.assertEqual((run / 'best-validation.pt').read_bytes(), best_bytes)
            self.assertEqual((run / marker['archives']['source_checkpoint']['path']).read_bytes(), source_bytes)
            self.assertEqual((run / marker['archives']['previous_best_checkpoint']['path']).read_bytes(), best_bytes)
            self.assertEqual(list(run.glob('*.partial')), [])

    def test_crash_before_save_replays_from_original_without_overwriting_archives(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            model, ema, _, checkpoint, request, path = fixture(run)
            first, _ = apply_frequency_transition(model, ema, checkpoint, request, run, path)
            archive_mtimes = {name: (run / record['path']).stat().st_mtime_ns
                              for name, record in first['archives'].items()}
            model.load_state_dict(checkpoint['model'])
            ema.load_state_dict(checkpoint['ema'])
            second, applied = apply_frequency_transition(model, ema, checkpoint, request, run, path)
            self.assertTrue(applied)
            self.assertEqual(first, second)
            for name, record in second['archives'].items():
                self.assertEqual((run / record['path']).stat().st_mtime_ns, archive_mtimes[name])
            self.assertTrue(torch.equal(model.embedding.frequencies,
                                        checkpoint['model']['embedding.frequencies'] * 4))

    def test_checkpoint_tensors_sharing_module_storage_are_not_mutated(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            model, ema, _, checkpoint, request, path = fixture(run)
            checkpoint['model'] = model.state_dict()
            checkpoint['ema'] = ema.state_dict()
            original = checkpoint['model']['embedding.frequencies'].clone()
            apply_frequency_transition(model, ema, checkpoint, request, run, path)
            self.assertTrue(torch.equal(checkpoint['model']['embedding.frequencies'], original))
            self.assertTrue(torch.equal(checkpoint['ema']['embedding.frequencies'], original))
            self.assertTrue(torch.equal(model.embedding.frequencies, original * 4))
            self.assertIn('embedding.frequencies', dict(model.named_buffers()))

    def test_marked_resume_never_rescales_even_without_request_or_source_archive(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            model, ema, _, checkpoint, request, path = fixture(run)
            marker, _ = apply_frequency_transition(model, ema, checkpoint, request, run, path)
            checkpoint['model'] = copy.deepcopy(model.state_dict())
            checkpoint['ema'] = copy.deepcopy(ema.state_dict())
            checkpoint['frequency_width_transition'] = copy.deepcopy(marker)
            checkpoint['step'] += 100
            for name in (None, request):
                with self.subTest(request_present=name is not None):
                    restored, applied = apply_frequency_transition(model, ema, checkpoint, name,
                                                                   run / 'absent', run / 'absent.pt')
                    self.assertFalse(applied)
                    self.assertTrue(restored['best_selection_pending'])
                    self.assertTrue(torch.equal(model.embedding.frequencies,
                                                checkpoint['model']['embedding.frequencies']))
                    self.assertIsNot(restored, checkpoint['frequency_width_transition'])

    def test_no_request_and_no_marker_is_a_noop(self):
        model = TinyModel()
        ema = copy.deepcopy(model)
        checkpoint = {'step': 100}
        self.assertEqual(apply_frequency_transition(model, ema, checkpoint, None,
                                                    '/does-not-exist', '/missing.pt'), (None, False))
        self.assertFalse(hasattr(model, 'frequency_width_transition'))

    def test_invalid_requests_fail_before_archiving_or_changing_buffers(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            model, ema, _, checkpoint, request, path = fixture(run)
            bad_requests = [
                {**request, 'unexpected': True},
                {**request, 'version': True},
                {**request, 'to_width': 16},
                {**request, 'from_width': True},
                {**request, 'transition_id': '../outside'},
                {**request, 'minimum_checkpoint_step': 101},
                {**request, 'minimum_checkpoint_step': True},
                {**request, 'expected_config_sha256': '0' * 64},
                {**request, 'source_frequency_sha256': {'model': '0' * 64, 'ema': '0' * 64}},
                {**request, 'target_frequency_sha256': {'model': '0' * 64, 'ema': '0' * 64}},
            ]
            for invalid in bad_requests:
                with self.subTest(invalid=invalid):
                    with self.assertRaises(ValueError):
                        apply_frequency_transition(model, ema, checkpoint, invalid, run, path)
                    self.assertTrue(torch.equal(model.embedding.frequencies,
                                                checkpoint['model']['embedding.frequencies']))
                    self.assertFalse(hasattr(model, 'frequency_width_transition'))
                    self.assertEqual(set(item.name for item in run.iterdir()),
                                     {'latest.pt', 'best-validation.pt'})

    def test_conflicting_archive_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            model, ema, _, checkpoint, request, path = fixture(run)
            conflict = run / 'previous_best_checkpoint-before-width4-approved-step-00000100.pt'
            conflict.write_bytes(b'conflicting prior archive')
            with self.assertRaisesRegex(ValueError, 'archive differs'):
                apply_frequency_transition(model, ema, checkpoint, request, run, path)
            self.assertEqual(conflict.read_bytes(), b'conflicting prior archive')
            self.assertFalse(list(run.glob('source_checkpoint-before-*')))
            self.assertTrue(torch.equal(model.embedding.frequencies,
                                        checkpoint['model']['embedding.frequencies']))

    def test_missing_best_is_allowed_but_missing_source_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            model, ema, _, checkpoint, request, path = fixture(run)
            (run / 'best-validation.pt').unlink()
            with self.assertRaisesRegex(ValueError, 'checkpoint is missing'):
                apply_frequency_transition(model, ema, checkpoint, request, run, run / 'absent.pt')
            marker, applied = apply_frequency_transition(model, ema, checkpoint, request, run, path)
            self.assertTrue(applied)
            self.assertEqual(set(marker['archives']), {'source_checkpoint'})

    def test_marked_resume_rejects_changed_request_corruption_and_rewound_step(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            model, ema, _, checkpoint, request, path = fixture(run)
            marker, _ = apply_frequency_transition(model, ema, checkpoint, request, run, path)
            checkpoint['model'] = copy.deepcopy(model.state_dict())
            checkpoint['ema'] = copy.deepcopy(ema.state_dict())
            checkpoint['frequency_width_transition'] = copy.deepcopy(marker)
            with self.assertRaisesRegex(ValueError, 'request differs'):
                apply_frequency_transition(model, ema, checkpoint,
                                           {**request, 'transition_id': 'different'}, run, path)
            with self.assertRaisesRegex(ValueError, 'precedes its frequency transition'):
                apply_frequency_transition(model, ema, {**checkpoint, 'step': 99}, None, run, path)
            with self.assertRaisesRegex(ValueError, 'configuration differs'):
                apply_frequency_transition(model, ema, {**checkpoint, 'config': {'steps': 42}}, None, run, path)
            model.embedding.frequencies.mul_(4)
            with self.assertRaisesRegex(ValueError, 'buffer differs'):
                apply_frequency_transition(model, ema, checkpoint, None, run, path)

    def test_unmarked_target_buffer_fails_instead_of_guessing_previous_activation(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            model, ema, _, checkpoint, request, path = fixture(run)
            for kind, module in (('model', model), ('ema', ema)):
                module.embedding.frequencies.mul_(4)
                checkpoint[kind] = copy.deepcopy(module.state_dict())
            with self.assertRaisesRegex(ValueError, 'buffer differs'):
                apply_frequency_transition(model, ema, checkpoint, request, run, path)
            self.assertFalse(list(run.glob('*-before-*')))

    def test_nan_or_unsynchronized_ema_fails_before_change(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            model, ema, _, checkpoint, request, path = fixture(run)
            ema.embedding.frequencies[0] = float('nan')
            with self.assertRaisesRegex(ValueError, 'Invalid ema'):
                apply_frequency_transition(model, ema, checkpoint, request, run, path)
            self.assertFalse(list(run.glob('*-before-*')))

    def selected_checkpoint(self, run):
        model, ema, _, checkpoint, request, path = fixture(run)
        marker, _ = apply_frequency_transition(model, ema, checkpoint, request, run, path)
        checkpoint['model'] = copy.deepcopy(model.state_dict())
        checkpoint['ema'] = copy.deepcopy(ema.state_dict())
        checkpoint['frequency_width_transition'] = copy.deepcopy(marker)
        checkpoint['frequency_width_transition']['best_selection_pending'] = False
        checkpoint['step'] = 110
        checkpoint['best_validation'] = .04
        checkpoint['history'].append({'step': 110, 'validation_restoration_mse_nfe1': .04})
        torch.save(checkpoint, path)
        return model, ema, checkpoint, request, path

    def test_marked_resume_recovers_interrupted_first_best_publication(self):
        for old_best in ('missing', 'corrupt', 'width1'):
            with self.subTest(old_best=old_best), tempfile.TemporaryDirectory() as directory:
                run = Path(directory)
                model, ema, checkpoint, request, path = self.selected_checkpoint(run)
                best = run / 'best-validation.pt'
                if old_best == 'missing':
                    best.unlink()
                elif old_best == 'width1':
                    legacy = copy.deepcopy(checkpoint)
                    legacy.pop('frequency_width_transition')
                    for kind in ('model', 'ema'):
                        legacy[kind]['embedding.frequencies'] /= 4
                    torch.save(legacy, best)
                old_bytes = best.read_bytes() if best.exists() else None
                marker, applied = apply_frequency_transition(model, ema, checkpoint, None, run, path)
                self.assertFalse(applied)
                self.assertFalse(marker['best_selection_pending'])
                self.assertEqual(best.read_bytes(), path.read_bytes())
                archives = list(run.glob('best-validation-before-frequency-recovery-*.pt'))
                if old_bytes is None:
                    self.assertEqual(archives, [])
                else:
                    self.assertEqual(len(archives), 1)
                    self.assertEqual(archives[0].read_bytes(), old_bytes)
                # Once reconciled, ordinary resume is read-only for the best file.
                before = best.stat().st_mtime_ns
                apply_frequency_transition(model, ema, checkpoint, request, run, path)
                self.assertEqual(best.stat().st_mtime_ns, before)

    def test_valid_earlier_width4_best_does_not_require_latest_to_be_validation_step(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            model, ema, checkpoint, request, path = self.selected_checkpoint(run)
            publish_best_checkpoint(path, run / 'best-validation.pt')
            checkpoint['step'] = 111
            checkpoint['history'].append({'step': 111, 'loss': .2})
            torch.save(checkpoint, path)
            _, applied = apply_frequency_transition(model, ema, checkpoint, request, run, path)
            self.assertFalse(applied)
            self.assertEqual(torch.load(run / 'best-validation.pt', weights_only=False)['step'], 110)

    def test_recovery_refuses_newer_or_better_width4_history(self):
        for change in ({'step': 111}, {'best_validation': .02}):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as directory:
                run = Path(directory)
                model, ema, checkpoint, request, path = self.selected_checkpoint(run)
                newer = {**checkpoint, **change}
                best = run / 'best-validation.pt'
                torch.save(newer, best)
                original = best.read_bytes()
                with self.assertRaisesRegex(ValueError, 'newer or better history'):
                    apply_frequency_transition(model, ema, checkpoint, request, run, path)
                self.assertEqual(best.read_bytes(), original)
                self.assertEqual(list(run.glob('best-validation-before-frequency-recovery-*.pt')), [])

    def test_recovery_refuses_to_invent_lost_best_from_nonvalidation_latest(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            model, ema, checkpoint, request, path = self.selected_checkpoint(run)
            checkpoint['step'] = 111
            checkpoint['history'].append({'step': 111, 'loss': .2})
            torch.save(checkpoint, path)
            best = run / 'best-validation.pt'
            original = best.read_bytes()
            with self.assertRaisesRegex(ValueError, 'not its selected validation step'):
                apply_frequency_transition(model, ema, checkpoint, request, run, path)
            self.assertEqual(best.read_bytes(), original)

    def test_recovery_refuses_changed_resume_file(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            model, ema, checkpoint, request, path = self.selected_checkpoint(run)
            torch.save({**checkpoint, 'step': 112}, path)
            with self.assertRaisesRegex(ValueError, 'file differs from loaded state'):
                apply_frequency_transition(model, ema, checkpoint, request, run, path)

    def test_atomic_best_copy_keeps_old_file_on_interrupted_publication(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            latest, best = run / 'latest.pt', run / 'best-validation.pt'
            latest.write_bytes(b'complete new checkpoint')
            best.write_bytes(b'old complete checkpoint')
            with patch('rmfsr.frequency_transition.os.replace', side_effect=OSError('copy interrupted')):
                with self.assertRaisesRegex(OSError, 'interrupted'):
                    publish_best_checkpoint(latest, best)
            self.assertEqual(best.read_bytes(), b'old complete checkpoint')
            self.assertEqual(list(run.glob('*.partial')), [])
            publish_best_checkpoint(latest, best)
            self.assertEqual(best.read_bytes(), b'complete new checkpoint')


if __name__ == '__main__':
    unittest.main()
