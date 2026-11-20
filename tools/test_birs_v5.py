"""CPU checks for v5 local references, null, gradients and legacy isolation."""
import copy
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch.nn import functional as F

import test_birs as legacy
from propvg.models import build_head
from propvg.models.heads.binding_residual_v3 import AdaptiveBindingResidualScoring
from propvg.models.heads.binding_residual_v5 import LocalBindingResidualScoring
from propvg.utils.checkpoint import _validate_birs_checkpoint


class BIRSV5Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def branch(self, **kwargs):
        torch.manual_seed(122)
        options = dict(hidden_channels=16, model_dim=16, num_heads=4,
                       reference_topk=8, seed_topk=2, min_targets=3, max_targets=7,
                       num_groups=2, chunk_size=2, checkpoint_chunks=False)
        options.update(kwargs)
        return LocalBindingResidualScoring(**options)

    def forward(self, m, args=None, **refs):
        args = legacy.BIRSTests().inputs() if args is None else args
        values = dict(reference_queries=args[0], reference_boxes=args[1], reference_classes=args[2])
        values.update(refs)
        return m(*args, **values)

    def test_isotropic_nearest_distance_and_strict_overlap(self):
        m = self.branch(neighbour_topk=1)
        targets = torch.tensor([[[.48,.48,.52,.52]]])
        refs = torch.tensor([[[.69,.49,.71,.51],[.49,.79,.51,.81],[.49,.49,.51,.51]]])
        ids, keep, distances = m.select_local_references(
            targets, refs, torch.ones(1,1,3,dtype=torch.bool), torch.tensor([[2.,1.]]))
        self.assertEqual(ids.tolist(), [[[1]]])
        self.assertTrue(keep.all())
        torch.testing.assert_close(distances, torch.tensor([[[.3]]]))

    def test_nearest_eight_no_refill_and_deterministic_ties(self):
        m = self.branch()
        targets = torch.tensor([[[.01,.01,.03,.03]]])
        centers = torch.arange(12).float()/20+.1
        refs = torch.stack((centers-.005, centers-.005, centers+.005, centers+.005), -1)[None]
        legal = torch.ones(1,1,12,dtype=torch.bool)
        ids,keep,_ = m.select_local_references(targets,refs,legal,torch.ones(1,2))
        self.assertEqual(ids.tolist(), [[list(range(8))]])
        self.assertEqual(int(keep.sum()),8)
        legal[...,2:] = False
        ids,keep,_ = m.select_local_references(targets,refs,legal,torch.ones(1,2))
        self.assertEqual(int(keep.sum()),2)
        self.assertEqual(ids.shape[-1],8)  # Remaining slots are masked padding.

    def test_per_target_pools_masks_and_compact_scoring(self):
        m = self.branch(neighbour_topk=2)
        widths=[]
        original=m._score_local
        def record(*args):
            widths.append(args[1].shape[2])
            return original(*args)
        with patch.object(m,'_score_local',side_effect=record):
            out=self.forward(m)
        self.assertTrue(all(w<=2 for w in widths))
        valid=out['local_reference_valid']; ids=out['local_reference_indices']
        self.assertTrue((ids[~valid]==-1).all())
        self.assertTrue((valid.sum(-1)<=2).all())
        target=out['target_indices']
        self.assertTrue((ids[valid]!=target[...,None].expand_as(ids)[valid]).all())
        legal=out['legal_references'].gather(1,target[...,None].expand(-1,-1,out['reference_indices'].shape[1]))
        target_boxes=out['boxes'].gather(1,target[...,None].expand(-1,-1,4))
        ref_boxes=out['boxes'].gather(1,out['reference_indices'][...,None].expand(-1,-1,4))
        from propvg.models.heads.binding_residual import pairwise_iou
        self.assertTrue((pairwise_iou(target_boxes,ref_boxes)[legal]==0).all())
        self.assertFalse(torch.equal(ids[0,0],ids[0,2]))
        attn=out['selected_attention']
        self.assertTrue((attn[...,:-1][~legal]==0).all())
        torch.testing.assert_close(attn.sum(-1),torch.ones_like(attn[...,0]))

    def test_fixed_pool_controls_diagnostics_and_gt_free_forward(self):
        m=self.branch(neighbour_topk=3)
        with m.diagnostics(): out=self.forward(m)
        diag=out['diagnostics']
        for key in ('local_reference_indices','local_reference_valid','local_reference_distances',
                    'selected_control_attention','reference_boxes','reference_confidence'):
            self.assertIn(key,diag)
        target=out['target_indices']
        legal=out['legal_references'].gather(1,target[...,None].expand(-1,-1,out['reference_indices'].shape[1]))
        controls=diag['selected_control_attention'][...,:-1]
        self.assertTrue((controls[~legal[:,:,None].expand_as(controls)]==0).all())
        args=list(legacy.BIRSTests().inputs())
        args[-1][0]['refer_target_index']=[]
        again=self.forward(m,args)
        for key in ('scores','delta','local_reference_indices','selected_controls'):
            torch.testing.assert_close(out[key],again[key],rtol=0,atol=0)

    def test_null_only_returns_crs_even_if_control_reads_references(self):
        m=self.branch();calls=[0]
        def scorer(target,reference,geometry,text,padding,sentence,legal):
            b,k,h=legal.shape
            attn=target.new_zeros(b,k,h+1)
            if calls[0]%(m.permutations+1)==0:
                attn[...,-1]=1
                score=target.sum(-1)*0+1
            else:
                attn[...,:-1]=legal/ legal.sum(-1,keepdim=True).clamp(min=1)
                attn[...,-1]=(~legal.any(-1)).float()
                score=target.sum(-1)*0-1
            calls[0]+=1
            return score,attn,(attn[...,:-1]>0).sum(-1)
        with patch.object(m,'_score_local',side_effect=scorer):out=self.forward(m)
        self.assertFalse(out['eligible'].any())
        self.assertEqual(float(out['delta'].abs().sum()),0)
        torch.testing.assert_close(out['scores'],out['base_scores'][...,None],rtol=0,atol=0)

    def test_sparsemax_extreme_masked_logits_and_null(self):
        logits=torch.tensor([[-20000.,0.,-20000.],[0.,5.,20.]],requires_grad=True)
        allowed=torch.tensor([[True,False,True],[False,False,True]])
        attn=self.branch().sparse_weights(logits,allowed)
        torch.testing.assert_close(attn,torch.tensor([[.5,0.,.5],[0.,0.,1.]]))
        attn.sum().backward()
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_bce_full_candidates_weights_and_gradients(self):
        m=self.branch(min_targets=3,max_targets=3)
        args=list(legacy.BIRSTests().inputs()); targets,labels=legacy.BIRSTests().supervision()
        labels.zero_();labels[0,6]=1
        out=self.forward(m,args);losses,_=m.loss(out,targets,args[-1],labels)
        self.assertEqual(set(losses),{'loss_birs_crs','loss_birs_final'})
        self.assertFalse(out['candidate_mask'][0,6])
        v=out['valid']
        torch.testing.assert_close(losses['loss_birs_crs'],F.binary_cross_entropy_with_logits(args[6].squeeze(-1)[v],labels[v]))
        torch.testing.assert_close(losses['loss_birs_final'],.5*F.binary_cross_entropy_with_logits(out['final_logits'].squeeze(-1)[v],labels[v]))
        sum(losses.values()).backward()
        self.assertLess(float(args[6].grad[0,6]),0)
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters()))
        self.assertIsNone(args[1].grad);self.assertIsNone(args[2].grad)

    def test_empty_references_one_reference_invalid_and_missing_gt(self):
        for kind in ('empty','single','padding','invalid','missing'):
            m=self.branch(neighbour_topk=1 if kind=='single' else 8)
            args=list(legacy.BIRSTests().inputs()); targets,labels=legacy.BIRSTests().supervision()
            refs={}
            if kind=='empty':refs['reference_queries']=torch.full_like(args[0],float('nan'))
            if kind=='padding':args[4]=torch.ones_like(args[4])
            if kind=='invalid':args[1]=torch.zeros_like(args[1])
            if kind=='missing':args[-1][0]['refer_target_index']=[];labels.zero_();targets=[torch.empty(0,4)]
            out=self.forward(m,args,**refs)
            self.assertTrue(torch.isfinite(out['scores']).all(),kind)
            if kind!='missing':self.assertEqual(float(out['delta'].abs().sum()),0,kind)
            losses,_=m.loss(out,targets,args[-1],labels)
            sum(losses.values()).backward()
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters()),kind)

    def test_checkpoint_chunk_and_padding_invariance(self):
        m=self.branch();other=copy.deepcopy(m);other.checkpoint_chunks=True;other.chunk_size=1
        gradients=[]
        for branch in (m,other):
            args=list(legacy.BIRSTests().inputs());targets,labels=legacy.BIRSTests().supervision()
            if branch is other:args[-1][0]['pad_shape']=(1000,2000)
            out=self.forward(branch,args);losses,_=branch.loss(out,targets,args[-1],labels)
            sum(losses.values()).backward()
            gradients.append([p.grad.clone() for p in branch.parameters()])
        for a,b in zip(*gradients):torch.testing.assert_close(a,b,rtol=1e-4,atol=1e-6)

    def test_detach_only_new_branch_and_joint_upstream(self):
        for detach in (False,True):
            m=self.branch(detach_inputs=detach);args=list(legacy.BIRSTests().inputs())
            with torch.no_grad():m.null_logit.bias.fill_(-100)
            out=self.forward(m,args)
            # Isolate the new final objective from the still-live original BCE.
            out['final_logits'].sum().backward()
            for index in (0,3,6):
                grad=args[index].grad
                if detach:self.assertTrue(grad is None or grad.abs().sum()==0)
                else:self.assertIsNotNone(grad)
            self.assertIsNone(args[1].grad);self.assertIsNone(args[2].grad)

    def test_policies_versions_and_validation(self):
        m=self.branch()
        wrap=lambda b:SimpleNamespace(head=SimpleNamespace(birs=b))
        state={'head.birs.'+k:v for k,v in m.state_dict().items()}
        _validate_birs_checkpoint(wrap(self.branch()),state)
        with self.assertRaises(ValueError):_validate_birs_checkpoint(wrap(self.branch(neighbour_topk=4)),state)
        old=AdaptiveBindingResidualScoring(hidden_channels=16,model_dim=16,num_heads=4,
              reference_topk=8,seed_topk=2,min_targets=3,max_targets=7,num_groups=2,chunk_size=2)
        with self.assertRaises(ValueError):_validate_birs_checkpoint(wrap(old),state)
        self.assertEqual(set(old.state_dict()),set(m.state_dict()))
        for value in (0,-1,True,1.5):
            with self.assertRaises(ValueError):self.branch(neighbour_topk=value)
        for kwargs in ({'cf_loss_weight':.1},{'attention_soft_mix':.1}):
            with self.assertRaises(ValueError):self.branch(**kwargs)

    def test_configuration_v3_kept_and_v5_selectable(self):
        path=Path(__file__).resolve().parents[1]/'configs/aerialvg/PropVG-aerialvg.py'
        source=path.read_text();old={};new={}
        exec(compile(source,str(path),'exec'),old)
        exec(compile(source.replace('birs_version = "v3"','birs_version = "v5"',1),str(path),'exec'),new)
        self.assertEqual(new['model']['head']['birs']['version'],'v5')
        self.assertEqual(new['model']['head']['birs']['neighbour_topk'],8)
        for key in ('enable_bcem','enable_grab','teacher','work_dir'):
            self.assertEqual(old[key],new[key])

    def test_real_head_v5_and_baseline_initialization_unchanged(self):
        options=legacy.BIRSTests().head_options();options.pop('birs',None);options['enable_birs']=False
        torch.manual_seed(100);baseline=build_head(options);rng=torch.get_rng_state()
        options.update(enable_birs=True,birs=dict(version='v5',model_dim=16,num_heads=4,
                       reference_topk=8,max_targets=6,checkpoint_chunks=True))
        torch.manual_seed(100);head=build_head(options)
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        for key,value in baseline.state_dict().items():self.assertTrue(torch.equal(value,head.state_dict()[key]),key)
        x=torch.randn(1,768,8,8)
        metas=[dict(img_shape=(96,128),pad_shape=(128,128),refer_target_index=[0])]
        targets=dict(mask=None,bbox=[torch.tensor([[40.,30.,64.,50.]])],img_metas=metas)
        args=dict(cls_feat=torch.zeros(1,768),lan_feat=torch.randn(1,8,768),
                  lan_mask=torch.tensor([[0,0,0,0,1,1,1,1]]),img=torch.zeros(1,3,128,128))
        losses,pred,_=head.forward_train(x,targets,**args)
        self.assertIn('candidate_output_mask',pred)
        self.assertIn('loss_birs_final',losses);self.assertIn('loss_refer',losses)
        self.assertNotIn('loss_birs_crs',losses)
        sum(v for key,v in losses.items() if key.startswith('loss_')).backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in head.birs.parameters()))
        head.birs.mark_optimizer_step();head.eval()
        with torch.no_grad():pred,_=head.forward_test(x,targets=targets,with_mask=False,**args)
        self.assertLessEqual(int(pred['candidate_output_mask'].sum()),6)
        q=torch.randn(1,8,256);raw=torch.randn_like(q);boxes=torch.rand(1,8,4);classes=torch.randn(1,8,2)
        with patch.object(head.birs,'forward',return_value={}) as call:
            head._binding_residual(q,boxes,classes,torch.randn(1,8,256),args['lan_mask'],
                 torch.rand(1,8,1),torch.rand(1,8,1),metas,raw,boxes,classes,None)
        torch.testing.assert_close(call.call_args.kwargs['reference_queries'],head.refer_adapter(raw))


if __name__=='__main__':
    unittest.main()
