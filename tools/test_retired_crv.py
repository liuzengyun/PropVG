"""CRV removal must leave disabled historical configurations usable."""
from pathlib import Path
import unittest

import torch
from mmcv import Config

import test_birs as legacy
from propvg.models import build_head


class RetiredCRVTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def options(self):
        options=legacy.BIRSTests().head_options()
        options.update(enable_birs=False,enable_detector_topk=False)
        options.pop('enable_crv',None);options.pop('crv',None)
        return options

    def test_archived_disabled_options_weights_rng_and_forward(self):
        options=self.options()
        torch.manual_seed(19);base=build_head(options);rng=torch.get_rng_state()
        torch.manual_seed(19);old=build_head(dict(options,enable_crv=False,
            crv=dict(num_groups=10,grid_size=48,start_epoch=20)))
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        self.assertFalse(hasattr(old,'crv'))
        self.assertFalse(hasattr(old,'_crv_refer_gt'))
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

    def test_enabling_retired_crv_fails_explicitly(self):
        with self.assertRaisesRegex(ValueError,'CRV has been removed'):
            build_head(dict(self.options(),enable_crv=True))

    def test_files_configs_and_training_no_longer_contain_crv(self):
        root=Path(__file__).resolve().parents[1]
        for filename in ('configs/aerialvg/PropVG-aerialvg.py','configs/aerialsense/PropVG-aerialsense.py'):
            cfg=Config.fromfile(str(root/filename))
            for values in (cfg,cfg.model.head):
                self.assertNotIn('enable_crv',values)
                self.assertNotIn('crv',values)
        for filename in ('propvg/apis/train.py','propvg/utils/checkpoint.py'):
            self.assertNotIn('crv',(root/filename).read_text().lower())
        for filename in ('propvg/models/heads/counterfactual_refinement.py',
                         'propvg/models/heads/proposal_selection.py','tools/test_crv.py','CRV_method.md'):
            self.assertFalse((root/filename).exists(),filename)


if __name__=='__main__':
    unittest.main()
