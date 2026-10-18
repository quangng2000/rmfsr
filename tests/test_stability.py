"""CPU checks for opt-in raw-metric HOLDs before optimizer/EMA/checkpoints."""
import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import torch

from rmfsr.flow_curriculum import config_digest, resolve_transition, training_flow_progress
from rmfsr.stability import (StabilityGuardError, check_stability, stability_config,
                            stability_resume_transition)
from rmfsr.train import train
from tests.test_training_integration import fixture


def settings():
    # Deliberately illustrative test limits; the implementation has no defaults.
    return dict(version=1, thresholds=dict(meanflow_loss=.6, target_rms=.5, gradient_norm=6.),
                required_exceedances=2, single_extreme_factor=10.)


class StabilityTests(unittest.TestCase):
    def test_disabled_and_healthy_guard_do_not_write_or_modify_metrics(self):
        self.assertIsNone(stability_config({}))
        self.assertIsNone(stability_config({'stability_guard': None}))
        with tempfile.TemporaryDirectory() as directory:
            raw=dict(meanflow_loss=.01, target_rms=.07)
            original=copy.deepcopy(raw)
            check_stability(None, {}, float('nan'), directory, 1)
            check_stability(settings(), raw, .2, directory, 1)
            self.assertEqual(raw, original)
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_finite_huge_values_are_held_with_atomic_valid_json(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(StabilityGuardError) as caught:
                check_stability(settings(), dict(meanflow_loss=3e12,target_rms=1e5), 1e12, directory, 84738)
            report=json.loads(caught.exception.path.read_text())
            self.assertEqual(report['attempted_step'], 84738)
            self.assertFalse(report['optimizer_step_performed'])
            self.assertEqual(report['raw_metrics']['meanflow_loss'], 3e12)
            self.assertEqual(len(list(Path(directory).iterdir())), 1)
            self.assertFalse(list(Path(directory).glob('*.partial')))

    def test_strict_count_rule_and_single_extreme(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg=settings()
            # Exact limits pass; one ordinary exceedance also passes.
            check_stability(cfg, dict(meanflow_loss=.6,target_rms=.5),6.,directory,1)
            check_stability(cfg, dict(meanflow_loss=.61,target_rms=.1),.2,directory,1)
            with self.assertRaises(StabilityGuardError):
                check_stability(cfg, dict(meanflow_loss=.1,target_rms=.51),6.1,directory,1)
            with self.assertRaises(StabilityGuardError) as caught:
                check_stability(cfg, dict(meanflow_loss=6.01,target_rms=.1),.2,directory,2)
            self.assertEqual(caught.exception.report['reasons'][0]['kind'], 'single_extreme_raw_metric')

    def test_nonfinite_and_missing_metrics_fail_closed_with_json_safe_values(self):
        with tempfile.TemporaryDirectory() as directory:
            for value, encoded in [(float('nan'),'NaN'),(float('inf'),'+Infinity'),(-float('inf'),'-Infinity')]:
                with self.subTest(value=encoded), self.assertRaises(StabilityGuardError) as caught:
                    check_stability(settings(), dict(meanflow_loss=.1,target_rms=.1),value,directory,1)
                self.assertEqual(json.loads(caught.exception.path.read_text())['raw_metrics']['gradient_norm'], encoded)
            with self.assertRaises(StabilityGuardError) as caught:
                check_stability(settings(), dict(target_rms=.1),.1,directory,2)
            self.assertEqual(caught.exception.report['reasons'][0]['kind'], 'missing_raw_metrics')

    def test_invalid_settings_are_rejected_before_preflight(self):
        bad=[{}, False, {'version':1}, {**settings(),'version':True},
             {**settings(),'required_exceedances':0}, {**settings(),'required_exceedances':True},
             {**settings(),'single_extreme_factor':1}, {**settings(),'single_extreme_factor':float('inf')},
             {**settings(),'unknown':1}]
        for value in (0,-1,True,float('nan'),float('inf'),'6'):
            bad.append({**settings(),'thresholds':{**settings()['thresholds'],'gradient_norm':value}})
        for value in bad:
            with self.subTest(value=value), patch('rmfsr.preflight.startup_check') as startup:
                with self.assertRaises(ValueError):train({'stability_guard':value},'unused')
                startup.assert_not_called()

    def test_resume_activation_is_narrow_and_keeps_flow_clock_binding(self):
        old=dict(steps=300000,schedule_steps=300000,flow_schedule='figure1-cosine-approx')
        request=dict(version=1,expected_config_sha256=config_digest(old),ramp_updates=10000,target_progress=.8)
        policy=resolve_transition(old,old,9000,request)
        previous={**old,'flow_schedule_transition':policy}
        new={**previous,'stability_guard':settings(),'allow_stability_guard_on_resume':True}
        self.assertIsNotNone(stability_resume_transition(previous,new))
        active=resolve_transition(new,previous,12000,allowed_config_changes=('stability_guard','allow_stability_guard_on_resume'))
        self.assertEqual(training_flow_progress(12000,previous),training_flow_progress(12000,{**new,'flow_schedule_transition':active}))
        self.assertEqual(active['request'],policy['request'])
        with self.assertRaises(ValueError):stability_resume_transition(previous,{**new,'allow_stability_guard_on_resume':False})
        with self.assertRaises(ValueError):stability_resume_transition(new,previous)
        with self.assertRaises(ValueError):stability_resume_transition(new,{**new,'stability_guard':{**settings(),'required_exceedances':3}})
        with self.assertRaisesRegex(ValueError,'Configuration differs'):
            resolve_transition({**new,'other_recipe_change':True},previous,12000,allowed_config_changes=('stability_guard','allow_stability_guard_on_resume'))


class TinyModel(torch.nn.Module):
    receptive_frames=1
    def __init__(self):
        super().__init__();self.weight=torch.nn.Parameter(torch.tensor(1.))
    def architecture(self):return dict(test='stability-toy')


class StabilityTrainingIntegrationTests(unittest.TestCase):
    def test_nonfinite_gradient_or_early_loss_error_writes_hold_before_step(self):
        for mode in ('gradient','microbatch'):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                root=Path(directory);cfg={**fixture(root),'steps':1,'stability_guard':settings()};run=root/'run'
                original_adam=torch.optim.AdamW;optimizers=[]
                def make_optimizer(*args,**kwargs):
                    opt=original_adam(*args,**kwargs)
                    opt.step=Mock(side_effect=AssertionError('Optimizer advanced'))
                    optimizers.append(opt);return opt
                def fail(model,*args,**kwargs):
                    if mode=='microbatch':raise FloatingPointError('Nonfinite microbatch loss')
                    model.weight.grad=torch.full_like(model.weight,float('nan'))
                    return dict(loss=.01,target_rms=.07,dp_mse=.01)
                with patch('rmfsr.train.model_from_config',side_effect=lambda cfg:TinyModel()), \
                     patch('rmfsr.train.torch.optim.AdamW',side_effect=make_optimizer), \
                     patch('rmfsr.train.backward_microbatches',side_effect=fail), \
                     patch('rmfsr.train.save_checkpoint') as save, contextlib.redirect_stdout(io.StringIO()):
                    with self.assertRaises(StabilityGuardError) as caught:train(cfg,run)
                report=json.loads(caught.exception.path.read_text())
                self.assertEqual(report['last_completed_step'],0)
                self.assertEqual(report['reasons'][0]['kind'], 'nonfinite_raw_metrics' if mode=='gradient' else 'nonfinite_microbatch')
                optimizers[0].step.assert_not_called();save.assert_not_called()
                self.assertFalse((run/'latest.pt').exists())

    def test_hold_preserves_model_adam_ema_saved_checkpoint_and_counter(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);cfg={**fixture(root),'steps':1};run=root/'run'
            def healthy(model,*args,**kwargs):
                (model.weight.square()*.01).backward()
                return dict(loss=.01,target_rms=.07,dp_mse=.01)
            with patch('rmfsr.train.model_from_config',side_effect=lambda cfg:TinyModel()), \
                 patch('rmfsr.train.backward_microbatches',side_effect=healthy), \
                 patch('rmfsr.train.validate',return_value={'validation_restoration_mse_nfe1':.2}), contextlib.redirect_stdout(io.StringIO()):
                train(cfg,run)
            checkpoint=run/'latest.pt';original_bytes=checkpoint.read_bytes()
            saved=torch.load(checkpoint,weights_only=False)
            models=[];optimizers=[];averages=[]
            original_adam=torch.optim.AdamW;original_deepcopy=copy.deepcopy
            def model_factory(cfg):
                model=TinyModel();models.append(model);return model
            def optimizer_factory(*args,**kwargs):
                opt=original_adam(*args,**kwargs)
                opt.step=Mock(side_effect=AssertionError('Optimizer advanced after HOLD'))
                optimizers.append(opt);return opt
            def capture_copy(value,*args,**kwargs):
                result=original_deepcopy(value,*args,**kwargs)
                if isinstance(value,TinyModel):averages.append(result)
                return result
            def bad(model,*args,**kwargs):
                (model.weight.square()*100.).backward()
                return dict(loss=20.,target_rms=5.,dp_mse=.1)
            enabled={**cfg,'steps':2,'stability_guard':settings(),'allow_stability_guard_on_resume':True}
            with patch('rmfsr.train.model_from_config',side_effect=model_factory), \
                 patch('rmfsr.train.torch.optim.AdamW',side_effect=optimizer_factory), \
                 patch('rmfsr.train.copy.deepcopy',side_effect=capture_copy), \
                 patch('rmfsr.train.backward_microbatches',side_effect=bad), \
                 patch('rmfsr.train.save_checkpoint') as save, \
                 patch('rmfsr.train.validate') as validate, contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaises(StabilityGuardError):train(enabled,run,checkpoint)
            optimizers[0].step.assert_not_called();save.assert_not_called();validate.assert_not_called()
            torch.testing.assert_close(models[0].weight,saved['model']['weight'],rtol=0,atol=0)
            torch.testing.assert_close(averages[0].weight,saved['ema']['weight'],rtol=0,atol=0)
            actual=optimizers[0].state_dict()['state']
            for key,state in saved['optimizer']['state'].items():
                for name,value in state.items():torch.testing.assert_close(actual[key][name],value,rtol=0,atol=0)
            self.assertEqual(checkpoint.read_bytes(),original_bytes)
            manifest=json.loads((run/'run.json').read_text())
            self.assertEqual(manifest['status'],'unstable_training');self.assertEqual(manifest['completed_steps'],1)
            self.assertEqual(len((run/'metrics.jsonl').read_text().splitlines()),1)


if __name__=='__main__':unittest.main()
