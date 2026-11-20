"""Removal of the observational head must preserve grounding and evaluation."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch import nn
from mmcv import Config

import test_birs as legacy
from propvg.models import build_head


class RetiredCorrectnessTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def options(self):
        options=legacy.BIRSTests().head_options()
        options.update(enable_birs=False,enable_detector_topk=False)
        options.pop('enable_correctness',None);options.pop('correctness',None)
        return options

    def test_archived_disabled_fields_preserve_parameters_rng_and_predictions(self):
        options=self.options()
        torch.manual_seed(29);base=build_head(options);rng=torch.get_rng_state()
        torch.manual_seed(29);old=build_head(dict(options,enable_correctness=False,
            correctness=dict(hidden_dim=128,topk=5,loss_weight=1.0)))
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        self.assertFalse(hasattr(old,'correctness'))
        self.assertEqual(base.state_dict().keys(),old.state_dict().keys())
        for key,value in base.state_dict().items():
            torch.testing.assert_close(value,old.state_dict()[key],rtol=0,atol=0)
        old.load_state_dict(base.state_dict(),strict=True)
        base.eval();old.eval()
        x=torch.randn(1,768,8,8)
        args=dict(cls_feat=torch.zeros(1,768),lan_feat=torch.randn(1,8,768),
            lan_mask=torch.tensor([[0,0,0,0,1,1,1,1]]),img=torch.zeros(1,3,128,128),
            targets=dict(img_metas=[dict(img_shape=(96,128),pad_shape=(128,128))]),with_mask=False)
        with torch.no_grad():a,_=base.forward_test(x,**args);b,_=old.forward_test(x,**args)
        self.assertFalse(any(key.startswith('top1_correct') for key in b))
        for key in ('pred_class_allobj','pred_bbox_allobj','refer_label'):
            torch.testing.assert_close(a[key],b[key],rtol=0,atol=0)

    def test_enabling_removed_head_fails_explicitly(self):
        with self.assertRaisesRegex(ValueError,'correctness head has been removed'):
            build_head(dict(self.options(),enable_correctness=True))

    def test_files_configs_and_runtime_no_longer_contain_head(self):
        root=Path(__file__).resolve().parents[1]
        for filename in ('configs/aerialvg/PropVG-aerialvg.py','configs/aerialsense/PropVG-aerialsense.py'):
            cfg=Config.fromfile(str(root/filename))
            for values in (cfg,cfg.model.head):
                self.assertNotIn('enable_correctness',values)
                self.assertNotIn('correctness',values)
        for filename in ('propvg/apis/train.py','propvg/apis/test.py','propvg/utils/checkpoint.py',
                         'tools/eval_aerialvg.py','propvg/models/det_seg/mix_ref_unimodel_OMG.py'):
            self.assertNotIn('correctness',(root/filename).read_text().lower())
        for filename in ('propvg/models/heads/top1_correctness.py','propvg/evaluations/correctness.py',
                         'tools/test_correctness.py','Correctness_head.md'):
            self.assertFalse((root/filename).exists(),filename)

    def test_both_grounding_evaluators_preserve_metrics_and_do_not_read_gt(self):
        class Loader:
            dataset=SimpleNamespace(which_set='val')
            def __len__(self):return 1
            def __iter__(self):
                meta=dict(img_shape=(50,50),pad_shape=(50,50),refer_target_index=[0])
                yield dict(img_metas=SimpleNamespace(data=[[dict(meta) for _ in range(4)]]),
                    gt_bbox=SimpleNamespace(data=[[torch.tensor([[10.,10.,20.,20.]]) for _ in range(4)]]))
        class Model(nn.Module):
            def __init__(self):
                super().__init__();self.anchor=nn.Parameter(torch.zeros(1))
            def forward(self,**kwargs):
                assert 'gt_bbox' not in kwargs
                predictions=[]
                for index in range(4):
                    first=[10.,10.,20.,20.] if index<2 else [30.,30.,40.,40.]
                    predictions.append(dict(boxes=torch.tensor([first,[10.,10.,20.,20.]]),scores=torch.tensor([.8,.1])))
                return dict(pred_bboxes=predictions,pred_masks=[])
        from propvg.apis.test import evaluate_model
        model=Model();cfg=SimpleNamespace(dataset='AerialVG',distributed=False)
        with patch('propvg.apis.test.extract_data',side_effect=lambda x:x), \
             patch('propvg.apis.test.get_root_logger'),patch('propvg.apis.test.log_evaluation'):
            self.assertEqual(evaluate_model('val',cfg,model,Loader()),(50.,100.,50.))
        spec=importlib.util.spec_from_file_location('grounding_eval_tool',Path(__file__).with_name('eval_aerialvg.py'))
        evaluator=importlib.util.module_from_spec(spec);spec.loader.exec_module(evaluator)
        with patch.object(evaluator,'extract_data',side_effect=lambda x:x):
            self.assertEqual(evaluator.evaluate(model,Loader()),(50.,100.,50.))


if __name__=='__main__':
    unittest.main()
