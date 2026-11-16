"""Adaptive-prefix, full-scene joint CRV regressions; no legacy entropy code."""
import copy
import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

os.environ.setdefault("MPLCONFIGDIR", "/tmp/propvg_matplotlib")
import torch
from torch import nn
from torch.nn import functional as F
from mmcv import Config
from propvg.models import build_head
from propvg.models.heads.counterfactual_refinement import CounterfactualRefinement, pairwise_iou
from propvg.models.heads.proposal_selection import adaptive_prefix, spherical_groups
from propvg.apis.train import clip_training_gradients, train_model
from propvg.utils.checkpoint import _validate_crv_checkpoint


class CRVTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def module(self, **kwargs):
        options = dict(hidden_channels=16, scorer_dim=16, scorer_heads=4, scorer_layers=1,
                       grid_size=6, num_groups=10, min_targets=3, max_targets=8,
                       chunk_size=3, checkpoint_chunks=False)
        options.update(kwargs)
        torch.manual_seed(1717)
        return CounterfactualRefinement(**options)

    def inputs(self, count=8):
        torch.manual_seed(73)
        boxes = torch.tensor([[[.12+.1*i,.5,.06,.1] for i in range(count)]],requires_grad=True)
        z = torch.linspace(2.,-1.,count)[None,:,None].clone().requires_grad_()
        return [torch.randn(1,16,24,24,requires_grad=True), torch.zeros(1,24,24,dtype=torch.bool), boxes,
                torch.randn(1,6,16,requires_grad=True),torch.tensor([[0,0,0,0,1,1]]),
                z.sigmoid(),z,[dict(img_shape=(80,100),pad_shape=(100,100))],
                torch.randn(1,count,16,requires_grad=True)]

    def gt(self,args,index=2):
        index=min(index,args[2].shape[1]-1)
        return self.module().clipped_boxes(args[2])[0,index].clone()

    def test_adaptive_multiseed_continuous_prefix_and_no_recursion(self):
        q=torch.randn(1,8,16,requires_grad=True);valid=torch.ones(1,8,dtype=torch.bool)
        p=torch.linspace(.9,.2,8)[None]
        labels=torch.tensor([[0,1,1,2,1,1,2,2]])
        with patch('propvg.models.heads.proposal_selection.spherical_groups',return_value=labels):
            groups,mask,before,ids=adaptive_prefix(q,valid,p,10,6,3,3,8)
        self.assertEqual(ids[0].tolist(),[0,1,2,3,4,5])
        self.assertEqual(int(mask.sum()),6)
        # Rank4 belongs to C but does NOT pull ranks7/8 into the prefix.
        labels=torch.tensor([[0,1,1,1,2,2,2,2]])
        with patch('propvg.models.heads.proposal_selection.spherical_groups',return_value=labels):
            self.assertEqual(len(adaptive_prefix(q,valid,p,10,6,3,3,8)[-1][0]),4)

    def test_clustering_is_deterministic_no_rng_and_matches_birs(self):
        from propvg.models.heads.binding_residual import BindingInteractionResidualScoring
        args=self.inputs();valid=torch.ones(1,8,dtype=torch.bool)
        rng=torch.get_rng_state()
        groups=spherical_groups(args[-1],valid,10,6)
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        dummy=SimpleNamespace(num_groups=10,kmeans_iters=6)
        expected=BindingInteractionResidualScoring.cluster(dummy,args[-1],valid)
        torch.testing.assert_close(groups,expected,rtol=0,atol=0)

    def test_mask_geometry_padding_and_channelwise_whole_image_mean(self):
        m=self.module(grid_size=24);args=self.inputs()
        valid=torch.ones(24,24,dtype=torch.bool);valid[19:]=False
        feature=torch.arange(24*24.).reshape(1,1,24,24).repeat(1,2,1,1)
        feature[:,1]+=1000
        gt=torch.tensor([[.3,.3,.5,.5]])
        mask=m.intervention_masks(gt,feature[0],args[-2][0],valid)
        original,denominator,mean=m.pooled_original(feature,valid[None])
        masked=m.pooled_interventions(feature,valid[None],mask,None,original,denominator,mean)
        full=feature*(1-mask[:,None])+mean*mask[:,None]
        expected=full*valid/denominator.clamp(min=1e-8)
        torch.testing.assert_close(masked,expected)
        self.assertEqual(float(mean[0,1]-mean[0,0]),1000.)
        self.assertTrue((mask[:,19:]==0).all())
        self.assertEqual(float(mask.max()),1)
        self.assertTrue(((mask>0)&(mask<1)).any())

    def test_pooling_equals_explicit_mean_replacement(self):
        m=self.module(grid_size=6);feature=torch.randn(1,2,24,24,requires_grad=True)
        valid=torch.ones(1,24,24,dtype=torch.bool);valid[:,19:]=False
        masks=torch.zeros(2,24,24);masks[0,5:9,7:12]=1;masks[1,12:18,10:16]=.4
        original,den,mean=m.pooled_original(feature,valid)
        masked=m.pooled_interventions(feature,valid,masks,None,original,den,mean)
        full=feature*(1-masks[:,None])+mean*masks[:,None]
        expected=F.adaptive_avg_pool2d(full*valid[:,None],6)/den.clamp(min=1e-8)
        torch.testing.assert_close(masked,expected)
        masked.sum().backward();self.assertGreater(float(feature.grad.abs().sum()),0)

    def test_packed_masks_match_individual_geometry_and_coverage(self):
        m=self.module();args=self.inputs();feature=args[0]
        coordinates=m.spatial_grid(feature[0],args[-2][0])
        grids=(coordinates[0][None],coordinates[1][None],feature.new_tensor([coordinates[2]]),feature.new_tensor([coordinates[3]]))
        boxes=m.clipped_boxes(args[2])[0,:3];gt=self.gt(args)
        masks,coverage=m.packed_masks(boxes,torch.zeros(3,dtype=torch.long),grids,~args[1],gt[None])
        expected=m.intervention_masks(boxes,feature[0],args[-2][0],~args[1][0])
        torch.testing.assert_close(masks,expected)
        torch.testing.assert_close(coverage,m.gt_coverage(expected,gt,feature[0],args[-2][0]))

    def test_no_local_crop_large_intervention_covers_full_scene(self):
        m=self.module(grid_size=6);feature=torch.randn(1,2,24,24)
        valid=torch.ones(1,24,24,dtype=torch.bool)
        mask=m.intervention_masks(torch.tensor([[.1,.1,.9,.9]]),feature[0],
                                  dict(img_shape=(100,100),pad_shape=(100,100)),valid[0])
        original,den,mean=m.pooled_original(feature,valid)
        masked=m.pooled_interventions(feature,valid,mask,None,original,den,mean)
        torch.testing.assert_close(masked,mean.expand_as(masked),atol=1e-6,rtol=1e-5)

    def test_balanced_bce_multi_positive_ignore_and_no_positive(self):
        z=torch.tensor([1.,2.,3.,4.],requires_grad=True);iou=torch.tensor([1.,.7,.2,0.])
        actual,used=self.module().balanced_bce(z,iou)
        expected=(F.softplus(-z[:2]).mean()+F.softplus(z[3]))/2
        torch.testing.assert_close(actual,expected);self.assertTrue(used)
        actual.backward();self.assertEqual(float(z.grad[2]),0)
        self.assertLess(float(z.grad[0]),0);self.assertGreater(float(z.grad[3]),0)
        loss,used=self.module().balanced_bce(z,torch.zeros(4))
        self.assertFalse(used);self.assertEqual(float(loss),0)
        # IoU=.5 is deliberately ignored, following the selected strict rule.
        self.assertFalse(self.module().balanced_bce(z,torch.full((4,),.5))[1])

    def test_joint_bce_gradients_to_crs_visual_and_text_but_not_boxes(self):
        m,args=self.module(),self.inputs()
        _,_,losses,info=m(*args,refer_gt=[self.gt(args)])
        self.assertEqual(set(losses),{'loss_crv_fusion','loss_crv_cf','loss_crv_gt'})
        self.assertEqual(float(info['crv_bce_samples']),1)
        losses['loss_crv_fusion'].backward()
        for i in [0,3,6]:
            self.assertGreater(float(args[i].grad.abs().sum()),0)
            self.assertTrue(torch.isfinite(args[i].grad).all())
        self.assertIsNone(args[2].grad)
        self.assertIsNone(args[-1].grad)  # Only discrete selection uses Q.
        self.assertGreater(float(m.strength_logit.grad.abs()),0)
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters()))

    def test_cf_losses_train_scene_network_not_crs_logits_directly(self):
        m,args=self.module(),self.inputs()
        _,_,losses,info=m(*args,refer_gt=[self.gt(args)])
        self.assertEqual(float(info['crv_rank_samples']),1)
        (losses['loss_crv_cf']+losses['loss_crv_gt']).backward()
        self.assertGreater(float(args[0].grad.abs().sum()),0)
        self.assertGreater(float(args[3].grad.abs().sum()),0)
        self.assertTrue(args[6].grad is None or float(args[6].grad.abs().sum())==0)
        self.assertTrue(m.strength_logit.grad is None or float(m.strength_logit.grad.abs())==0)

    def test_expanded_mask_gt_coverage_rejects_disjoint_negative(self):
        m=self.module(negative_gt_coverage=.01)
        feature=torch.zeros(16,100,100);valid=torch.ones(100,100,dtype=torch.bool)
        gt=torch.tensor([.4,.4,.5,.5]);candidate=torch.tensor([[.51,.4,.61,.5]])
        self.assertEqual(float(pairwise_iou(candidate,gt[None])[0,0]),0)
        masks=m.intervention_masks(candidate,feature,dict(img_shape=(100,100),pad_shape=(100,100)),valid)
        coverage=m.gt_coverage(masks,gt,feature,dict(img_shape=(100,100),pad_shape=(100,100)))
        self.assertGreater(float(coverage[0]),m.negative_gt_coverage)

    def test_no_positive_bce_skipped_gt_anchor_still_trains(self):
        m,args=self.module(max_targets=3),self.inputs()
        gt=torch.tensor([.05,.05,.15,.15])
        _,_,losses,info=m(*args,refer_gt=[gt])
        self.assertEqual(float(info['crv_bce_samples']),0)
        self.assertEqual(float(info['crv_gt_samples']),1)
        self.assertEqual(float(losses['loss_crv_fusion']),0)
        self.assertGreater(float(losses['loss_crv_gt']),0)

    def test_all_padding_invalid_candidates_and_missing_gt_are_safe(self):
        for kind in ['text','visual','boxes','gt']:
            m,args=self.module(),self.inputs()
            if kind=='text':args[4]=torch.ones_like(args[4])
            if kind=='visual':args[1]=torch.ones_like(args[1])
            if kind=='boxes':args[2]=torch.zeros_like(args[2])
            gt=None if kind=='gt' else self.gt(args)
            _,priority,losses,info=m(*args,refer_gt=[gt])
            self.assertTrue(all(torch.isfinite(v) for v in losses.values()))
            if kind in ['text','visual','gt']:
                self.assertEqual(float(sum(losses.values())),0)
                self.assertEqual(float(info['crv_supervised_samples']),0)
            sum(losses.values()).backward()
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters()))

    def test_no_entropy_routing_even_for_confident_crs(self):
        m,args=self.module(),self.inputs()
        args[6]=torch.tensor([[[10.]]+[[0.]]*7],requires_grad=True);args[5]=args[6].sigmoid()
        _,_,_,info=m(*args,refer_gt=[self.gt(args)])
        self.assertEqual(float(info['crv_active']),1)
        self.assertFalse(hasattr(m,'uncertainty_threshold'))
        self.assertFalse(hasattr(m,'gate'))

    def test_original_scored_once_and_mask_count_is_selected_plus_gt(self):
        m,args=self.module(),self.inputs()
        with patch.object(m,'_score_original',wraps=m._score_original) as original, \
             patch.object(m,'_score_masked',wraps=m._score_masked) as masked:
            _,_,_,info=m(*args,refer_gt=[self.gt(args)])
        self.assertEqual(original.call_count,1)
        scenes=sum(call.args[3].numel() for call in masked.call_args_list)
        self.assertEqual(scenes,int(info['crv_target_count'])+1)

    def test_chunks_checkpoint_and_repeated_calls_are_deterministic(self):
        a,b=self.module(chunk_size=1),self.module(chunk_size=8,checkpoint_chunks=True)
        b.load_state_dict(a.state_dict(),strict=True)
        args=self.inputs();gt=self.gt(args);rng=torch.get_rng_state()
        sa,pa,la,ia=a(*args,refer_gt=[gt])
        with b.diagnostics():sb,pb,lb,ib=b(*args,refer_gt=[gt])
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        torch.testing.assert_close(sa,sb,atol=1e-6,rtol=1e-5)
        for key in la:torch.testing.assert_close(la[key],lb[key],atol=1e-6,rtol=1e-5)
        sum(lb.values()).backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in b.parameters()))
        self.assertEqual(ib['crv_samples'][0]['drops'].device.type,'cpu')

    def test_eval_does_not_read_gt_and_outputs_only_shortlist(self):
        m,args=self.module(max_targets=4),self.inputs()
        m.last_training_epoch.fill_(1);m.mark_optimizer_step();m.eval()
        class Forbidden:
            def __getitem__(self,k):raise AssertionError("Inference GT access")
        with torch.no_grad():_,priority,losses,info=m(*args,refer_gt=Forbidden())
        self.assertEqual(float(sum(losses.values())),0)
        self.assertLessEqual(int(info['crv_output_mask'].sum()),4)
        self.assertTrue((priority.squeeze(-1)[~info['crv_output_mask']]==0).all())
        self.assertEqual(len(info['crv_samples'][0]['indices']),int(info['crv_output_mask'].sum()))

    def test_start_epoch_and_untrained_eval_do_not_apply_random_correction(self):
        m,args=self.module(start_epoch=3),self.inputs()
        _,_,losses,info=m(*args,refer_gt=[self.gt(args)],epoch=0)
        self.assertEqual(float(info['crv_ready']),0)
        sum(losses.values()).backward()
        self.assertTrue(all(p.grad is None for p in m.parameters()))
        m.eval()
        with torch.no_grad():score,_,_,info=m(*args)
        torch.testing.assert_close(score,args[5],rtol=0,atol=0)
        m.train();_,_,losses,info=m(*args,refer_gt=[self.gt(args)],epoch=2)
        self.assertEqual(float(info['crv_ready']),1)

    def test_checkpoint_policy_and_old_schema_rejection(self):
        wrapper=nn.Module();wrapper.head=nn.Module();wrapper.head.crv=self.module()
        state=wrapper.state_dict();_validate_crv_checkpoint(wrapper,state)
        _validate_crv_checkpoint(wrapper,{'module.'+k:v for k,v in state.items()})
        altered={k:v.clone() for k,v in state.items()};altered['head.crv.schema_version'].fill_(10)
        with self.assertRaises(ValueError):_validate_crv_checkpoint(wrapper,altered)
        wrapper.head.crv=self.module(num_groups=9)
        with self.assertRaises(ValueError):_validate_crv_checkpoint(wrapper,state)
        wrapper.head.crv=None;_validate_crv_checkpoint(wrapper,{})

    def options(self,enabled):
        options=copy.deepcopy(Config.fromfile('configs/aerialvg/PropVG-aerialvg.py').model.head)
        options.pop('crv',None)
        options.update(enable_crv=enabled,enable_birs=False,enable_relation_refinement=False,
                       enable_grab=False,enable_bcem=False,enable_correctness=False,num_queries=8,
                       crv=dict(scorer_dim=16,scorer_heads=4,scorer_layers=1,grid_size=4,
                                num_groups=10,min_targets=3,max_targets=6,checkpoint_chunks=False))
        return options

    def head_inputs(self):
        torch.manual_seed(123)
        x=torch.randn(1,768,8,8)
        target=dict(mask=None,bbox=[torch.tensor([[40.,30.,64.,50.]])],
                    img_metas=[dict(img_shape=(96,128),pad_shape=(128,128),refer_target_index=[0])])
        args=dict(cls_feat=torch.zeros(1,768),lan_feat=torch.randn(1,8,768),
                  lan_mask=torch.tensor([[0,0,0,0,1,1,1,1]]),img=torch.zeros(1,3,128,128))
        return x,target,args

    def test_real_head_original_loss_rng_and_parameters_unchanged(self):
        torch.manual_seed(3);base=build_head(self.options(False));rng=torch.get_rng_state()
        torch.manual_seed(3);new=build_head(self.options(True))
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        for key,value in base.state_dict().items():torch.testing.assert_close(value,new.state_dict()[key],rtol=0,atol=0)
        x,target,args=self.head_inputs()
        torch.manual_seed(4);old,_,_=base.forward_train(x,target,**args)
        torch.manual_seed(4);losses,pred,info=new.forward_train(x,target,**args)
        for key,value in old.items():torch.testing.assert_close(value,losses[key],rtol=0,atol=0)
        self.assertIn('candidate_output_mask',pred)
        self.assertEqual(new.loss_weight['refer'],base.loss_weight['refer'])
        sum(v for key,v in losses.items() if key.startswith('loss_')).backward()
        new.crv.mark_optimizer_step();new.eval()
        with torch.no_grad(),patch.object(new,'_crv_refer_gt',side_effect=AssertionError("No inference GT")):
            pred,info=new.forward_test(x,targets=target,with_mask=False,**args)
        self.assertLessEqual(int(pred['candidate_output_mask'].sum()),6)

    def test_joint_gradient_clipping_and_correctness_isolation(self):
        class Toy(nn.Module):
            def __init__(self):
                super().__init__();self.head=nn.Module();self.head.crv=nn.Linear(1,1,bias=False)
                self.head.crv.joint_training=True;self.base=nn.Parameter(torch.ones(1))
        m=Toy()
        m.base.grad=torch.tensor([3.]);m.head.crv.weight.grad=torch.tensor([[4.]])
        clip_training_gradients(m,1)
        self.assertAlmostEqual(float(m.base.grad),.6,places=5)
        self.assertAlmostEqual(float(m.head.crv.weight.grad),.8,places=5)

    def test_global_sample_normalization_and_no_supervision_rank(self):
        m,args=self.module(),self.inputs();gt=self.gt(args)
        _,_,local,_=m(*args,refer_gt=[gt])
        def reduce(value):value.copy_(value*3)  # 3 valid samples across 4 ranks
        with patch('torch.distributed.is_available',return_value=True), \
             patch('torch.distributed.is_initialized',return_value=True), \
             patch('torch.distributed.get_world_size',return_value=4), \
             patch('torch.distributed.all_reduce',side_effect=reduce):
            _,_,global_loss,info=m(*args,refer_gt=[gt])
        for key in local:torch.testing.assert_close(global_loss[key],local[key]*4/3)
        self.assertEqual(float(info['crv_supervised_samples']),3)

    def test_trainer_collects_new_losses_and_marks_trained_steps(self):
        class Toy(nn.Module):
            def __init__(self,branch):
                super().__init__();self.head=nn.Module();self.head.crv=branch
                self.base=nn.Parameter(torch.ones(1))
            def forward(self,**kwargs):
                beta=self.head.crv.strength_logit
                return {'loss_refer':self.base.square().sum(),'loss_crv_cf':beta.square(),
                        'loss_crv_fusion':beta.square(),'loss_crv_gt':beta.square(),
                        'metric_crv_supervised_samples':beta.new_tensor(1.)},{}
        m=Toy(self.module());optimizer=torch.optim.SGD(m.parameters(),lr=.01)
        before=float(m.head.crv.strength_logit)
        cfg=SimpleNamespace(distributed=False,use_fp16=False,grad_norm_clip=None)
        with patch('propvg.apis.train.extract_data',side_effect=lambda x:x),patch('propvg.apis.train.is_main',return_value=False):
            train_model(0,cfg,m,optimizer,[{}])
        self.assertEqual(int(m.head.crv.trained_steps),1)
        self.assertAlmostEqual(float(m.head.crv.strength_logit),before*(1-.06),places=5)


if __name__=='__main__':unittest.main()
