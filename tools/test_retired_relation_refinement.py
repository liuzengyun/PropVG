"""Retired relation refinement: no runtime branch, old disabled cfgs still work."""
from pathlib import Path
import unittest

import torch
from mmcv import Config

import test_birs as legacy
from propvg.models import build_head


class RetiredRelationRefinementTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def options(self):
        options=legacy.BIRSTests().head_options()
        options.update(enable_birs=False,enable_detector_topk=False)
        options.pop('enable_relation_refinement',None)
        options.pop('relation_refinement',None)
        return options

    def test_archived_disabled_fields_are_noop_and_checkpoint_compatible(self):
        options=self.options()
        torch.manual_seed(9);base=build_head(options);rng=torch.get_rng_state()
        torch.manual_seed(9);old=build_head(dict(options,enable_relation_refinement=False,
            relation_refinement=dict(target_topk=32,reference_topk=64,detach_inputs=True)))
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        self.assertFalse(hasattr(old,'relation_refiner'))
        self.assertFalse(hasattr(old,'_refine_relations'))
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
        for key in ('pred_class_allobj','pred_bbox_allobj','refer_label'):
            torch.testing.assert_close(a[key],b[key],rtol=0,atol=0)

    def test_enabled_retired_module_fails_explicitly(self):
        with self.assertRaisesRegex(ValueError,'has been removed'):
            build_head(dict(self.options(),enable_relation_refinement=True))

    def test_configuration_and_runtime_have_no_branch(self):
        root=Path(__file__).resolve().parents[1]
        cfg=Config.fromfile(str(root/'configs/aerialvg/PropVG-aerialvg.py'))
        for values in (cfg,cfg.model.head):
            self.assertNotIn('enable_relation_refinement',values)
            self.assertNotIn('relation_refinement',values)
        for file in ('propvg/apis/train.py','propvg/utils/checkpoint.py'):
            source=(root/file).read_text()
            self.assertNotIn('relation_refiner',source)
            self.assertNotIn('loss_relation_final',source)
        for file in ('propvg/models/heads/two_stage_relation.py','propvg/evaluations/two_stage_relation.py',
                     'tools/analyze_two_stage_relation.py','tools/test_two_stage_relation.py',
                     'AerialVG_两阶段关系评分设计.md'):
            self.assertFalse((root/file).exists(),file)


if __name__=='__main__':
    unittest.main()
