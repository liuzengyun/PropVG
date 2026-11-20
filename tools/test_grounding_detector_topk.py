"""CPU checks for the optional detector-confidence grounding shortlist."""
import unittest
from unittest.mock import patch

import torch
from torch import nn

import test_birs as legacy
from propvg.models import build_head
from propvg.models.heads.REFHead import REFHead


class GroundingDetectorTopKTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def selector(self, enabled=True, k=3):
        head=REFHead.__new__(REFHead);nn.Module.__init__(head)
        head.enable_detector_topk=enabled;head.detector_topk=k
        return head

    def test_topk_uses_foreground_probability_not_raw_logit(self):
        q=torch.arange(5.).reshape(1,5,1).requires_grad_()
        boxes=q.expand(-1,-1,4).clone()
        # Query0 has the largest fg logit, but a still larger bg logit.
        classes=torch.tensor([[[100.,101.],[3.,0.],[2.,0.],[-2.,0.],[1.,0.]]],requires_grad=True)
        values=self.selector()._select_grounding_queries(q,boxes,classes)
        selected_q,selected_boxes,selected_cls,ids=values
        self.assertEqual(ids.tolist(),[[1,2,4]])
        for actual,original in ((selected_q,q),(selected_boxes,boxes),(selected_cls,classes)):
            torch.testing.assert_close(actual,original.gather(1,ids[...,None].expand(-1,-1,original.shape[-1])))
        selected_q.sum().backward()
        torch.testing.assert_close(q.grad.squeeze(),torch.tensor([0.,1.,1.,0.,1.]))
        self.assertIsNone(classes.grad)

    def test_disabled_is_identity_and_grab_original_ids_compose(self):
        q=torch.randn(2,5,16);boxes=torch.rand(2,5,4);classes=torch.randn(2,5,2)
        source=torch.tensor([[200,10,30,80,15],[11,12,100,0,7]])
        disabled=self.selector(False)._select_grounding_queries(q,boxes,classes,source)
        for actual,expected in zip(disabled,(q,boxes,classes,source)):self.assertIs(actual,expected)
        selected=self.selector()._select_grounding_queries(q,boxes,classes,source)
        order=classes.softmax(-1)[...,0].argsort(dim=-1,descending=True,stable=True)[:,:3]
        torch.testing.assert_close(selected[-1],source.gather(1,order))

    def test_ties_fewer_queries_and_batched_selection(self):
        q=torch.randn(2,2,16);boxes=torch.rand(2,2,4);classes=torch.zeros(2,2,2)
        out=self.selector(k=64)._select_grounding_queries(q,boxes,classes)
        self.assertEqual(out[-1].tolist(),[[0,1],[0,1]])
        self.assertEqual(out[0].shape,(2,2,16))

    def options(self, version=None, **kwargs):
        options=legacy.BIRSTests().head_options()
        options.update(enable_detector_topk=True,detector_topk=3,enable_birs=version is not None)
        if version is not None:
            options['birs']=dict(version=version,model_dim=16,num_heads=4,num_groups=2,
                seed_topk=1,min_targets=2,max_targets=3,reference_topk=8,checkpoint_chunks=True)
        options.update(kwargs)
        return options

    def test_defaults_rng_state_and_old_checkpoint_unchanged(self):
        opts=self.options();opts.pop('enable_detector_topk');opts.pop('detector_topk')
        torch.manual_seed(22);old=build_head(opts);rng=torch.get_rng_state()
        torch.manual_seed(22);off=build_head(dict(opts,enable_detector_topk=False,detector_topk=64))
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        self.assertEqual(old.state_dict().keys(),off.state_dict().keys())
        for key,value in old.state_dict().items():torch.testing.assert_close(value,off.state_dict()[key],rtol=0,atol=0)
        off.load_state_dict(old.state_dict(),strict=True)
        for k in (0,-1,True,1.5):
            with self.assertRaises((ValueError,AssertionError)):build_head(dict(opts,detector_topk=k))

    def test_train_test_detection_unchanged_and_labels_aligned(self):
        for version in (None,'v3','v5'):
            head=build_head(self.options(version))
            x=torch.randn(1,768,8,8)
            metas=[dict(img_shape=(96,128),pad_shape=(128,128),refer_target_index=[0])]
            targets=dict(mask=None,bbox=[torch.tensor([[40.,30.,64.,50.]])],img_metas=metas)
            args=dict(cls_feat=torch.zeros(1,768),lan_feat=torch.randn(1,8,768),
                      lan_mask=torch.tensor([[0,0,0,0,1,1,1,1]]),img=torch.zeros(1,3,128,128))
            labels=torch.tensor([[1.,0.,0.,0.,0.,0.,0.,0.]])
            with patch.object(head.detrloss,'forward',wraps=head.detrloss.forward) as detection, \
                 patch.object(head,'_enhance_background',wraps=head._enhance_background) as context, \
                 patch.object(head,'prepare_referring_targets',return_value=labels):
                if version is not None:
                    with patch.object(head.birs,'loss',wraps=head.birs.loss) as criterion:
                        losses,pred,extra=head.forward_train(x,targets,**args)
                    torch.testing.assert_close(criterion.call_args.args[3],labels.gather(1,extra['grounding_source_indices']))
                    self.assertEqual(head.birs.reference_topk,3)
                else:
                    losses,pred,extra=head.forward_train(x,targets,**args)
                self.assertEqual(detection.call_args.args[0].shape[2],8)
                self.assertEqual(detection.call_args.args[1].shape[2],8)
                self.assertEqual(context.call_args.args[0].shape[1],3)
                self.assertEqual(pred['pred_bbox_allobj'].shape[1],3)
                self.assertEqual(pred['refer_label'].shape[1],3)
            sum(v for k,v in losses.items() if k.startswith('loss_')).backward()
            if version is not None:
                self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in head.birs.parameters()))
                head.birs.mark_optimizer_step()
            head.eval()
            with torch.no_grad():out,extra=head.forward_test(x,targets=targets,with_mask=False,**args)
            self.assertEqual(out['pred_bbox_allobj'].shape[1],3)
            self.assertEqual(extra['grounding_source_indices'].shape,(1,3))

    def test_raw_reference_bank_source_id_mapping_even_v1(self):
        for version in ('v1','v3','v5'):
            options=self.options(version)
            if version=='v1':options['birs']=dict(version='v1',model_dim=16,num_heads=4,reference_topk=8)
            head=build_head(options)
            q=torch.randn(1,3,256);raw=torch.randn(1,8,256)
            boxes=torch.rand(1,3,4);raw_boxes=torch.rand(1,8,4)
            classes=torch.randn(1,3,2);raw_classes=torch.randn(1,8,2)
            source=torch.tensor([[7,2,5]])
            with patch.object(head.birs,'forward',return_value={}) as call:
                head._binding_residual(q,boxes,classes,torch.randn(1,8,256),torch.zeros(1,8),
                    torch.rand(1,3,1),torch.rand(1,3,1),[dict(img_shape=(96,128))],
                    raw,raw_boxes,raw_classes,source)
            torch.testing.assert_close(call.call_args.kwargs['reference_queries'],head.refer_adapter(raw))
            self.assertIs(call.call_args.kwargs['source_indices'],source)
            self.assertIs(call.call_args.kwargs['reference_boxes'],raw_boxes)

    def test_grab_filter_kept_target_not_penalized_for_later_topk(self):
        head=build_head(self.options(enable_grab=True,detector_topk=1))
        x=torch.randn(1,768,8,8)
        metas=[dict(img_shape=(96,128),pad_shape=(128,128),refer_target_index=[0])]
        targets=dict(mask=None,bbox=[torch.tensor([[40.,30.,64.,50.]])],img_metas=metas)
        args=dict(cls_feat=torch.zeros(1,768),lan_feat=torch.randn(1,8,768),
                  lan_mask=torch.tensor([[0,0,0,0,1,1,1,1]]),img=torch.zeros(1,3,128,128))
        saved={}
        def grab(queries,boxes,*args,**kwargs):
            ids=torch.tensor([[0,1]],device=queries.device)
            filters=queries.sum(-1)*0
            saved['filters']=filters
            return queries[:,:2],boxes[:,:2],filters,ids
        def select(queries,boxes,classes,source):
            return queries[:,1:2],boxes[:,1:2],classes[:,1:2],source[:,1:2]
        with patch.object(head.grab_binding,'forward',side_effect=grab), \
             patch.object(head,'_select_grounding_queries',side_effect=select), \
             patch.object(head,'prepare_referring_targets',return_value=torch.tensor([[1.,0,0,0,0,0,0,0]])):
            losses,_,_=head.forward_train(x,targets,**args)
        # Zero filter logits: BCE=log(2); no extra keep penalty since Grab kept GT.
        torch.testing.assert_close(losses['loss_filter'],torch.tensor(.3*0.6931471805599453))


if __name__=='__main__':
    unittest.main()
