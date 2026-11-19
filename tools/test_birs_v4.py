"""CPU regression tests: v4 math, supervision, isolation and head wiring."""
from pathlib import Path
import unittest

import torch

import test_birs as legacy
from propvg.models import build_head
from propvg.models.heads.binding_residual_v3 import AdaptiveBindingResidualScoring
from propvg.models.heads.binding_residual_v4 import CompetitiveBindingResidualScoring
from propvg.utils.checkpoint import _validate_birs_checkpoint


class BIRSV4Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def branch(self, **kwargs):
        torch.manual_seed(122)
        options = dict(hidden_channels=16, model_dim=16, num_heads=4, reference_topk=8,
                       seed_topk=2, min_targets=3, max_targets=7, num_groups=2,
                       chunk_size=2, checkpoint_chunks=False)
        options.update(kwargs)
        return CompetitiveBindingResidualScoring(**options)

    def forward(self, m, args=None, **kwargs):
        args = legacy.BIRSTests().inputs() if args is None else args
        refs = dict(reference_queries=args[0], reference_boxes=args[1], reference_classes=args[2])
        refs.update(kwargs)
        return m(*args, **refs)

    def test_extreme_masked_sparsemax(self):
        z = torch.tensor([[-20000., 0., -20000.]], requires_grad=True)
        allowed = torch.tensor([[True, False, True]])
        m = self.branch()
        weights, sparse = m.reference_weights(z, allowed)
        torch.testing.assert_close(sparse, torch.tensor([[.5, 0., .5]]))
        torch.testing.assert_close(weights.sum(-1), torch.ones(1))
        weights[..., 0].sum().backward()
        self.assertTrue(torch.isfinite(z.grad).all())

    def test_reference_floor_and_no_reference_fallback(self):
        m = self.branch()
        z = torch.tensor([[-1000., -1000., 1000.], [2., 3., 1000.]], requires_grad=True)
        allowed = torch.tensor([[True, True, True], [False, False, True]])
        weights, _ = m.reference_weights(z, allowed)
        torch.testing.assert_close(weights[0, :-1].sum(), torch.tensor(.1))
        torch.testing.assert_close(weights[1], torch.tensor([0., 0., 1.]))
        weights.sum().backward()
        self.assertTrue(torch.isfinite(z.grad).all())

    def test_containment_and_partial_self_exclusion(self):
        m = self.branch()
        target = torch.tensor([[[.4,.4,.6,.6]]])
        refs = torch.tensor([[[.1,.1,.9,.9], [.41,.41,.45,.45], [.7,.7,.8,.8], [.55,.4,.7,.6]]])
        result = m.filter_references(torch.ones(1,1,4,dtype=torch.bool),target,refs)
        self.assertEqual(result.tolist(), [[[False, False, True, True]]])

    def test_position_swap_keeps_own_sizes_and_consistent_geometry(self):
        m = self.branch()
        target = torch.tensor([[[.4,.4,.6,.6]]])
        refs = torch.tensor([[[.1,.1,.2,.2],[.7,.6,.9,.9]]])
        aspect = torch.tensor([[2.,1.]])
        geo = m.geometry(target,refs,aspect)
        positions = m.control_positions(geo,torch.tensor([[[1,0]]]),target,refs,aspect)
        tc = (target[..., :2]+target[..., 2:])/2
        center = tc[:,:,None]+positions[...,:2]/aspect[:,None,None]
        half = (refs[...,2:]-refs[...,:2])[:,None]/2
        synthetic = torch.cat((center-half,center+half),-1)[:,0]
        expected = m.geometry(target,synthetic,aspect)
        torch.testing.assert_close(positions,expected)
        torch.testing.assert_close(positions[...,5:7],geo[...,5:7])

    def test_identical_semantics_disable_cf_and_delta_even_with_loss_off(self):
        m = self.branch(cf_loss_weight=0)
        args = list(legacy.BIRSTests().inputs())
        refs = args[0][:,:1].expand_as(args[0]).clone()
        out = self.forward(m,args,reference_queries=refs)
        self.assertFalse(bool(out['cf_valid_controls'].any()))
        self.assertEqual(float(out['delta'].abs().sum()),0)

    def test_focused_control_mask(self):
        m = self.branch(control_reference_topk=2)
        legal = torch.tensor([[[True, False, True, True]]])
        attention = torch.tensor([[[.1,.9,.3,.2,.0]]])
        self.assertEqual(m.control_reference_mask(legal,attention).tolist(), [[[False,False,True,True]]])

    def test_out_of_image_control_rejected(self):
        m = self.branch()
        target = torch.tensor([[[.4,.4,.6,.6]]])
        refs = torch.tensor([[[0.,.2,.1,.3],[.4,.6,1.,1.]]])
        aspect = torch.ones(1,2)
        geo = m.geometry(target,refs,aspect)
        perm = torch.tensor([[[1,0]]])
        pos = m.control_positions(geo,perm,target,refs,aspect)
        q = torch.eye(2)[None]
        valid = m.valid_control(geo,pos,torch.ones(1,1,2,dtype=torch.bool),q,perm,target,refs,aspect)
        self.assertFalse(bool(valid.any()))

    def test_no_gt_forward_and_loss_switch_inference_consistency(self):
        args = list(legacy.BIRSTests().inputs())
        args[-1] = [dict(img_shape=(100,200),pad_shape=(256,256))]
        a,b = self.branch(cf_loss_weight=.1),self.branch(cf_loss_weight=0)
        b.load_state_dict(a.state_dict(),strict=True)
        a.eval();b.eval();a.trained_steps.fill_(1);b.trained_steps.fill_(1)
        left,right = self.forward(a,args),self.forward(b,args)
        for name in ('scores','delta','cf_valid_controls','selected_control_mean'):
            torch.testing.assert_close(left[name],right[name],rtol=0,atol=0)
        valid = left['cf_valid_controls']
        mean = (left['selected_controls']*valid).sum(-1)/valid.sum(-1).clamp(min=1)
        torch.testing.assert_close(mean,left['selected_control_mean'])

    def test_v4_full_crs_and_half_final_bce(self):
        m = self.branch(cf_loss_weight=0)
        args = legacy.BIRSTests().inputs()
        targets,labels = legacy.BIRSTests().supervision()
        output = self.forward(m,args)
        losses,_ = m.loss(output,targets,args[-1],labels)
        valid = output['valid']
        expected = torch.nn.functional.binary_cross_entropy_with_logits(args[6].squeeze(-1)[valid],labels[valid])
        torch.testing.assert_close(expected,losses['loss_birs_crs'])
        expected = torch.nn.functional.binary_cross_entropy_with_logits(output['final_logits'].squeeze(-1)[valid],labels[valid])
        torch.testing.assert_close(.5*expected,losses['loss_birs_final'])

    def test_v4_gradients_match_old_for_no_cf_zero_residual_control(self):
        # This tests the residual fallback, not equality of the different BCE weights.
        args = legacy.BIRSTests().inputs()
        m = self.branch(cf_loss_weight=0)
        refs = args[0][:,:1].expand_as(args[0]).clone()
        out = self.forward(m,args,reference_queries=refs)
        torch.testing.assert_close(out['final_logits'],args[6],rtol=0,atol=0)

    def test_competitive_loss_trains_strength_not_base_logits(self):
        m = self.branch(cf_margin=.2)
        gap = torch.tensor([[.3,.3]],requires_grad=True)
        base = torch.tensor([[[2.],[2.1]]],requires_grad=True)
        strength = m.max_strength*m.strength_logit.sigmoid()
        out = dict(target_indices=torch.tensor([[0,1]]),target_valid=torch.ones(1,2,dtype=torch.bool),
                   cf_valid_controls=torch.ones(1,2,2,dtype=torch.bool),selected_support=gap,
                   selected_controls=torch.zeros(1,2,2),cf_gt_iou=torch.tensor([[.8,0.]]),
                   crs_logits=base,strength=strength)
        loss, active, *_ = m.counterfactual_terms(out,torch.tensor([[1.,0.]]),torch.tensor([True]))
        self.assertAlmostEqual(float(loss),.3,places=6)
        self.assertTrue(bool(active.all()))
        loss.sum().backward()
        self.assertIsNone(base.grad)
        self.assertLess(float(gap.grad[0,0]),0)
        self.assertGreater(float(gap.grad[0,1]),0)
        gap2 = torch.tensor([[.1,.3]],requires_grad=True)
        out.update(selected_support=gap2,strength=m.max_strength*m.strength_logit.sigmoid())
        m.counterfactual_terms(out,torch.tensor([[1.,0.]]),torch.tensor([True]))[0].sum().backward()
        self.assertGreater(float(m.strength_logit.grad),0)
        # Partly overlapping negatives and low-quality Hungarian positives are ignored.
        for overlaps in (torch.tensor([[.8,.1]]),torch.tensor([[.5,0.]])):
            out['cf_gt_iou'] = overlaps
            self.assertFalse(bool(m.counterfactual_terms(out,torch.tensor([[1.,0.]]),torch.tensor([True]))[1].any()))

    def test_forward_loss_gradients_empty_cases_and_checkpoint(self):
        for kind in ('normal','no_gt','padding','no_refs','invalid','detach'):
            m = self.branch(checkpoint_chunks=True,detach_inputs=kind=='detach')
            args = list(legacy.BIRSTests().inputs())
            if kind=='no_gt': args[-1] = [dict(args[-1][0],refer_target_index=[])]
            if kind=='padding': args[4] = torch.ones_like(args[4])
            if kind=='no_refs': args[2] = torch.full_like(args[2],float('nan'))
            if kind=='invalid': args[1] = torch.zeros_like(args[1])
            out = self.forward(m,args)
            targets,labels = legacy.BIRSTests().supervision()
            losses, _ = m.loss(out,targets,args[-1],labels)
            self.assertTrue(all(torch.isfinite(v) for v in losses.values()))
            sum(losses.values()).backward()
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters()))
            self.assertIsNone(args[1].grad)
            self.assertIsNone(args[2].grad)
            if kind=='detach':
                self.assertIsNone(args[0].grad)
                self.assertIsNone(args[3].grad)

    def test_policy_rejects_other_versions_and_changed_inference(self):
        m = self.branch()
        state = {'head.birs.'+k:v for k,v in m.state_dict().items()}
        for branch in (self.branch(), self.branch(reference_coverage=.4),
                       AdaptiveBindingResidualScoring(hidden_channels=16,model_dim=16)):
            model = torch.nn.Module(); model.head=torch.nn.Module(); model.head.birs=branch
            if branch is not m and torch.equal(branch.policy_signature,m.policy_signature):
                _validate_birs_checkpoint(model,state)
            else:
                with self.assertRaises(ValueError): _validate_birs_checkpoint(model,state)

    def test_real_head_train_and_test(self):
        options = legacy.BIRSTests().head_options()
        options['birs'] = dict(version='v4',model_dim=16,num_heads=4,reference_topk=8,
                               max_targets=8,checkpoint_chunks=True)
        head = build_head(options)
        x = torch.randn(1,768,8,8)
        metas = [dict(img_shape=(96,128),pad_shape=(128,128),refer_target_index=[0])]
        targets = dict(mask=None,bbox=[torch.tensor([[40.,30.,64.,50.]])],img_metas=metas)
        args = dict(cls_feat=torch.zeros(1,768),lan_feat=torch.randn(1,8,768),
                    lan_mask=torch.tensor([[0,0,0,0,1,1,1,1]]),img=torch.zeros(1,3,128,128))
        losses,pred,_ = head.forward_train(x,targets,**args)
        self.assertIn('loss_birs_cf',losses)
        self.assertIn('candidate_output_mask',pred)
        sum(v for k,v in losses.items() if k.startswith('loss_')).backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in head.birs.parameters()))
        head.eval()
        with torch.no_grad(): pred,_ = head.forward_test(x,targets=targets,with_mask=False,**args)
        self.assertIn('candidate_output_mask',pred)

    def test_v3_matches_designated_experiment_exactly(self):
        # The designated experiment includes the user's full-CRS-weight change.
        archive = (Path(__file__).resolve().parents[1] / 'work_dir/aerialvg/PropVG-birs3+bcem8'
                   / '20261118_084936+val_top1=71.68+val_top5=89.59+val_miou=62.23+test_top1=71.29+test_top5=90.71+test_miou=61.74'
                   / 'propvg/models/heads/binding_residual_v3.py')
        if not archive.is_file():
            self.skipTest('Designated v3 experiment archive is unavailable')
        source = archive.read_text()
        namespace = {'__name__':'propvg.models.heads._historical_v3','__package__':'propvg.models.heads'}
        exec(compile(source,str(archive),'exec'),namespace)
        options = dict(hidden_channels=16,model_dim=16,num_heads=4,reference_topk=8,
                       seed_topk=2,min_targets=3,max_targets=7,num_groups=2,chunk_size=2,
                       checkpoint_chunks=False)
        old = namespace['AdaptiveBindingResidualScoring'](**options)
        current = AdaptiveBindingResidualScoring(**options)
        current.load_state_dict(old.state_dict(),strict=True)
        old.eval();current.eval();old.trained_steps.fill_(1);current.trained_steps.fill_(1)
        args = legacy.BIRSTests().inputs()
        refs = dict(reference_queries=args[0],reference_boxes=args[1],reference_classes=args[2])
        a,b = old(*args,**refs),current(*args,**refs)
        for name in ('scores','final_logits','output_mask','ranking_priority','delta',
                     'selected_support','selected_controls','selected_attention'):
            torch.testing.assert_close(a[name],b[name],rtol=0,atol=0)
        targets,labels = legacy.BIRSTests().supervision()
        for key,value in old.loss(a,targets,args[-1],labels)[0].items():
            torch.testing.assert_close(value,current.loss(b,targets,args[-1],labels)[0][key],rtol=0,atol=0)


if __name__ == '__main__':
    unittest.main()
