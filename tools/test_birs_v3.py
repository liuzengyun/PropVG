"""BIRS v3 selection, interventions, full-BCE and legacy isolation tests."""
import copy
import hashlib
import json
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch import nn
from torch.nn import functional as F

import test_birs as legacy
from propvg.models import build_head
from propvg.models.heads.binding_residual_v3 import AdaptiveBindingResidualScoring
from propvg.utils.checkpoint import _validate_birs_checkpoint
from propvg.apis.train import HIDDEN_BIRS_LOG_FIELDS, train_model


class BIRSV3Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def branch(self, **kwargs):
        torch.manual_seed(122)
        options = dict(hidden_channels=16, model_dim=16, num_heads=4,
                       reference_topk=8, seed_topk=2, min_targets=3, max_targets=7,
                       num_groups=2, chunk_size=2, checkpoint_chunks=False)
        options.update(kwargs)
        return AdaptiveBindingResidualScoring(**options)

    def forward(self, m, args=None):
        args = legacy.BIRSTests().inputs() if args is None else args
        return m(*args, reference_queries=args[0], reference_boxes=args[1], reference_classes=args[2])

    def test_adaptive_multiseed_prefix_and_no_recursive_expansion(self):
        m = self.branch(seed_topk=3, min_targets=3)
        args = legacy.BIRSTests().inputs()
        valid = m.clipped_boxes(args[1], args[-1])[1]
        # CRS order 0..6. Two seed clusters extend to rank6; rank4's third
        # cluster must not recursively pull in rank7.
        labels = torch.tensor([[0,1,1,2,1,1,2,-1]])
        with patch.object(m, 'cluster', return_value=labels):
            groups, mask, ids, selected_valid, _ = m.select_targets(args[0], valid, args[5])
        self.assertEqual(mask.tolist(), [[True,True,True,True,True,True,False,False]])
        self.assertEqual(ids[0,:6].tolist(), list(range(6)))
        self.assertEqual(int(selected_valid.sum()), 6)
        labels = torch.tensor([[0,1,0,2,0,0,0,-1]])
        with patch.object(m, 'cluster', return_value=labels):
            self.assertEqual(int(m.select_targets(args[0], valid, args[5])[1].sum()), 7)

    def test_masks_and_shortlist_only_orders(self):
        m = self.branch(max_targets=5)
        out = self.forward(m)
        self.assertTrue(torch.equal(out['output_mask'], out['candidate_mask']))
        self.assertTrue((out['delta'][~out['candidate_mask']] == 0).all())
        self.assertTrue((out['ranking_priority'].squeeze(-1)[~out['candidate_mask']] == 0).all())
        self.assertEqual(set(out['after_orders'][0].tolist()), set(out['target_indices'][0,out['target_valid'][0]].tolist()))
        self.assertLessEqual(len(out['after_orders'][0]), 5)

    def test_permutation_preserves_legal_sets_and_moves_each_legal_slot(self):
        legal = torch.tensor([[[1,0,1,1,0],[0,0,0,0,0],[0,1,0,0,0]]]).bool()
        source = torch.tensor([[4,7,9]])
        for p in range(3):
            ids = self.branch().permutation_indices(legal, source, p)
            self.assertEqual(set(ids[0,0,legal[0,0]].tolist()), {0,2,3})
            self.assertTrue((ids[0,0,legal[0,0]] != torch.tensor([0,2,3])).all())
            self.assertTrue(torch.equal(ids[0,1], torch.arange(5)))
            self.assertTrue(torch.equal(ids[0,2], torch.arange(5)))

    def test_sparse_null_and_invalid_reference_masks(self):
        z = torch.tensor([[3.,-3.,-4.,0.]], requires_grad=True)
        weights = self.branch().sparse_weights(z, torch.tensor([[1,1,0,1]]).bool())
        self.assertEqual(float(weights[0,2]), 0)
        self.assertEqual(float(weights[0,1]), 0)
        self.assertAlmostEqual(float(weights.sum()), 1)
        out = self.forward(self.branch())
        attn = out['selected_attention']
        legal = out['legal_references'].gather(1,out['target_indices'][...,None].expand(-1,-1,out['legal_references'].shape[-1]))
        self.assertTrue((attn[...,:-1][~legal] == 0).all())
        torch.testing.assert_close(attn.sum(-1), torch.ones_like(attn[...,0]))

    def test_equal_geometry_or_equal_reference_semantics_cancel(self):
        args = list(legacy.BIRSTests().inputs())
        m = self.branch()
        # Each legal reference has identical appearance: permuting positions
        # changes token order only, so a set reader must give the same score.
        refs = args[0][:,:1].expand_as(args[0]).clone()
        out = m(*args, reference_queries=refs,reference_boxes=args[1],reference_classes=args[2])
        self.assertLess(float(out['raw_binding'].abs().max()), 2e-6)

    def test_full_bce_and_missing_positive_shortlist(self):
        m = self.branch(max_targets=3, min_targets=3)
        args = list(legacy.BIRSTests().inputs())
        t, labels = legacy.BIRSTests().supervision()
        labels.zero_(); labels[0,6] = 1
        out = self.forward(m,args)
        losses, info = m.loss(out,t,args[-1],labels)
        valid = out['valid']
        # Match the user's archived v3 experiment: full original CRS BCE.
        expected_base = F.binary_cross_entropy_with_logits(args[6].squeeze(-1)[valid], labels[valid])
        expected_final = F.binary_cross_entropy_with_logits(out['final_logits'].squeeze(-1)[valid], labels[valid])*.5
        torch.testing.assert_close(losses['loss_birs_crs'], expected_base)
        torch.testing.assert_close(losses['loss_birs_final'], expected_final)
        self.assertEqual(set(losses), {'loss_birs_crs','loss_birs_final'})
        self.assertEqual(float(info['refer_in_topn']), 0)
        self.assertEqual(float(info['supervised_samples']), 1)
        sum(losses.values()).backward()
        self.assertLess(float(args[6].grad[0,6]), 0)

    def test_joint_and_detached_gradients(self):
        for detach in [False,True]:
            m, args = self.branch(detach_inputs=detach), list(legacy.BIRSTests().inputs())
            t, labels = legacy.BIRSTests().supervision()
            out = self.forward(m,args)
            losses,_ = m.loss(out,t,args[-1],labels)
            losses['loss_birs_final'].backward(retain_graph=True)
            for i in [0,3,6]:
                grad = args[i].grad
                if detach:self.assertTrue(grad is None or float(grad.abs().sum()) == 0)
                else:self.assertGreater(float(grad.abs().sum()),0)
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters()))
            self.assertIsNone(args[1].grad)
            self.assertIsNone(args[2].grad)
            if detach:
                losses['loss_birs_crs'].backward()
                self.assertGreater(float(args[6].grad.abs().sum()),0)

    def test_all_padding_no_references_and_no_gt(self):
        for kind in ['padding','references','invalid','no_gt']:
            args = list(legacy.BIRSTests().inputs())
            if kind=='padding':args[4] = torch.ones_like(args[4])
            if kind=='references':args[2] = torch.full_like(args[2],float('nan'))
            if kind=='invalid':args[1] = torch.zeros_like(args[1])
            if kind=='no_gt':args[-1] = [dict(args[-1][0],refer_target_index=[])]
            m = self.branch(checkpoint_chunks=True)
            out = self.forward(m,args)
            t, labels = legacy.BIRSTests().supervision()
            losses, info = m.loss(out,t,args[-1],labels)
            self.assertTrue(all(torch.isfinite(v) for v in losses.values()))
            if kind!='no_gt':self.assertEqual(float(out['delta'].abs().sum()),0)
            if kind in ['invalid','no_gt']:self.assertEqual(float(sum(losses.values())),0)
            sum(losses.values()).backward()
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters()))

    def test_chunks_checkpoint_determinism_and_diagnostics(self):
        a,b = self.branch(chunk_size=1),self.branch(chunk_size=7,checkpoint_chunks=True)
        b.load_state_dict(a.state_dict(),strict=True)
        args = legacy.BIRSTests().inputs(); rng = torch.get_rng_state()
        with b.diagnostics():ob = self.forward(b,args)
        oa = self.forward(a,args)
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        torch.testing.assert_close(oa['final_logits'],ob['final_logits'],rtol=1e-5,atol=1e-6)
        self.assertEqual(ob['diagnostics']['selected_control_attention'].shape[2],2)
        t,labels=legacy.BIRSTests().supervision()
        loss,_=b.loss(ob,t,args[-1],labels);sum(loss.values()).backward()

    def test_policy_and_configuration_failures(self):
        def wrapper(branch):
            m=nn.Module();m.head=nn.Module();m.head.birs=branch;return m
        model=wrapper(self.branch());state=model.state_dict()
        _validate_birs_checkpoint(model,state)
        for kwargs in [dict(seed_topk=1),dict(permutations=1),dict(final_mix=.3)]:
            with self.assertRaises(ValueError):_validate_birs_checkpoint(wrapper(self.branch(**kwargs)),state)
        with self.assertRaises(ValueError):_validate_birs_checkpoint(wrapper(legacy.BIRSTests().module()),state)
        for kwargs in [dict(min_targets=1),dict(final_mix=0),dict(permutations=0)]:
            with self.assertRaises(ValueError):self.branch(**kwargs)

    def test_cf_loss_only_options_preserve_legacy_state_and_inference(self):
        old = self.branch()
        new = self.branch(cf_loss_weight=.1, cf_margin=.2, attention_soft_mix=0)
        self.assertEqual(list(old.state_dict()), list(new.state_dict()))
        for key, value in old.state_dict().items():
            torch.testing.assert_close(value, new.state_dict()[key], rtol=0, atol=0)
        # Reproduce the archived v3 signature, not just two new constructors.
        policy = dict(version='v3', hidden_channels=16, num_groups=2, kmeans_iters=6,
                      seed_topk=2, min_targets=3, max_targets=7, reference_topk=8,
                      duplicate_iou=.8, model_dim=16, num_heads=4, permutations=2,
                      max_strength=2.0, final_mix=.5, reference_source='pre_bcem_decoder',
                      permutation='legal_cyclic_v1', output='prefix_only', loss='full_valid_two_stage_bce')
        expected = torch.tensor(list(hashlib.sha256(json.dumps(policy, sort_keys=True).encode()).digest()),
                                dtype=torch.uint8)
        torch.testing.assert_close(old.policy_signature, expected, rtol=0, atol=0)
        wrapper=nn.Module();wrapper.head=nn.Module();wrapper.head.birs=new
        _validate_birs_checkpoint(wrapper, {'head.birs.'+key:value for key,value in old.state_dict().items()})
        old.mark_optimizer_step();new.mark_optimizer_step();old.eval();new.eval()
        args=legacy.BIRSTests().inputs();rng=torch.get_rng_state()
        with torch.no_grad():a,b=self.forward(old,args),self.forward(new,args)
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        for key in ('scores','final_logits','ranking_priority','output_mask','delta','selected_attention'):
            torch.testing.assert_close(a[key],b[key],rtol=0,atol=0)

    def test_soft_attention_is_opt_in_and_checkpoint_guarded(self):
        a,b=self.branch(),self.branch(attention_soft_mix=.1)
        wrapper=nn.Module();wrapper.head=nn.Module();wrapper.head.birs=b
        with self.assertRaises(ValueError):
            _validate_birs_checkpoint(wrapper,{'head.birs.'+k:v for k,v in a.state_dict().items()})
        # No new parameter/buffer names or RNG changes are introduced.
        self.assertEqual(list(a.state_dict()),list(b.state_dict()))
        for key,value in a.state_dict().items():
            if key!='policy_signature':torch.testing.assert_close(value,b.state_dict()[key],rtol=0,atol=0)
        logits=torch.tensor([[-4.,-4.,4.]],requires_grad=True)
        legal=torch.tensor([[True,False,True]])
        mixed,sparse=b.reference_weights(logits,legal)
        self.assertEqual(float(sparse[0,0]),0)
        self.assertGreater(float(mixed[0,0]),0)
        self.assertEqual(float(mixed[0,1]),0)
        torch.testing.assert_close(mixed.sum(-1),torch.ones(1))
        mixed[0,0].backward()
        self.assertGreater(float(logits.grad[0,0]),0)
        self.assertEqual(float(logits.grad[0,1]),0)

    def test_cf_hinge_supervises_only_positive_unscaled_support_gap(self):
        m=self.branch(cf_loss_weight=.1,attention_soft_mix=.1)
        args=list(legacy.BIRSTests().inputs());targets,labels=legacy.BIRSTests().supervision()
        out=self.forward(m,args);k=out['target_indices'].shape[1]
        factual=torch.zeros(1,k,requires_grad=True)
        controls=torch.zeros(1,k,2,requires_grad=True)
        out['selected_support'],out['selected_controls']=factual,controls
        out['cf_valid_controls']=out['target_valid'][...,None].expand(-1,-1,2).clone()
        loss,info=m.loss(out,targets,args[-1],labels)
        torch.testing.assert_close(loss['loss_birs_cf'],torch.tensor(.02))
        self.assertEqual(float(info['cf_samples']),1)
        slot=int((out['target_indices'][0]==0).nonzero()[0])
        loss['loss_birs_cf'].backward()
        self.assertAlmostEqual(float(factual.grad[0,slot]),-.1,places=6)
        self.assertAlmostEqual(float(controls.grad[0,slot,0]),.05,places=6)
        self.assertAlmostEqual(float(controls.grad[0,slot,1]),.05,places=6)
        remaining=torch.ones(k,dtype=torch.bool);remaining[slot]=False
        self.assertEqual(float(factual.grad[:,remaining].abs().sum()),0)
        self.assertEqual(float(controls.grad[:,remaining].abs().sum()),0)
        self.assertTrue(args[6].grad is None or float(args[6].grad.abs().sum())==0)
        self.assertTrue(m.strength_logit.grad is None or float(m.strength_logit.grad.abs())==0)

    def test_cf_hinge_stops_at_margin(self):
        m=self.branch(cf_loss_weight=.1);args=legacy.BIRSTests().inputs()
        targets,labels=legacy.BIRSTests().supervision();out=self.forward(m,args)
        out['selected_support']=torch.full_like(out['selected_support'],.3,requires_grad=True)
        out['selected_controls']=torch.zeros_like(out['selected_controls'],requires_grad=True)
        loss,info=m.loss(out,targets,args[-1],labels)
        self.assertEqual(float(loss['loss_birs_cf']),0)
        self.assertEqual(float(info['cf_margin_met']),1)
        loss['loss_birs_cf'].backward()
        self.assertEqual(float(out['selected_support'].grad.abs().sum()),0)

    def test_cf_direct_gradients_and_detached_input_option(self):
        for detach in (False,True):
            m=self.branch(cf_loss_weight=.1,cf_margin=10.,attention_soft_mix=.1,detach_inputs=detach)
            args=list(legacy.BIRSTests().inputs());targets,labels=legacy.BIRSTests().supervision()
            out=self.forward(m,args);loss,info=m.loss(out,targets,args[-1],labels)
            self.assertEqual(float(info['cf_samples']),1)
            loss['loss_birs_cf'].backward()
            for index in (0,3):
                grad=args[index].grad
                if detach:self.assertTrue(grad is None or float(grad.abs().sum())==0)
                else:self.assertGreater(float(grad.abs().sum()),0)
            self.assertTrue(args[6].grad is None or float(args[6].grad.abs().sum())==0)
            self.assertIsNone(args[1].grad);self.assertIsNone(args[2].grad)
            self.assertGreater(float(m.geometry_projection[0].weight.grad.abs().sum()),0)
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters()))

    def test_cf_gap_does_not_depend_on_beta(self):
        a,b=self.branch(cf_loss_weight=.1),self.branch(cf_loss_weight=.1)
        with torch.no_grad():a.strength_logit.fill_(-20);b.strength_logit.fill_(20)
        args=legacy.BIRSTests().inputs();targets,labels=legacy.BIRSTests().supervision()
        x,_=a.loss(self.forward(a,args),targets,args[-1],labels)
        y,_=b.loss(self.forward(b,args),targets,args[-1],labels)
        torch.testing.assert_close(x['loss_birs_cf'],y['loss_birs_cf'],rtol=0,atol=0)

    def test_cf_skips_missing_positive_unchanged_controls_and_invalid_inputs(self):
        for kind in ('outside','no_gt','no_references','padding','invalid','unchanged','equal_geometry'):
            m=self.branch(cf_loss_weight=.1,attention_soft_mix=.1,min_targets=3,max_targets=3)
            args=list(legacy.BIRSTests().inputs());targets,labels=legacy.BIRSTests().supervision()
            if kind=='outside':labels.zero_();labels[0,6]=1
            if kind=='no_gt':args[-1]=[dict(args[-1][0],refer_target_index=[])]
            if kind=='no_references':args[2]=torch.full_like(args[2],float('nan'))
            if kind=='padding':args[4]=torch.ones_like(args[4])
            if kind=='invalid':args[1]=torch.zeros_like(args[1])
            if kind=='unchanged':
                with patch.object(m,'permutation_indices',side_effect=lambda legal,source,number:
                                  torch.arange(legal.shape[-1]).expand_as(legal)):
                    out=self.forward(m,args)
            elif kind=='equal_geometry':
                with patch.object(m,'geometry',side_effect=lambda targets,references,aspect:
                                  targets.new_zeros(targets.shape[0],targets.shape[1],references.shape[1],9)):
                    out=self.forward(m,args)
            else:out=self.forward(m,args)
            loss,info=m.loss(out,targets,args[-1],labels)
            self.assertEqual(float(loss['loss_birs_cf']),0,kind)
            self.assertEqual(float(info['cf_samples']),0,kind)
            sum(loss.values()).backward()
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters()),kind)

    def test_cf_ignores_identity_controls_in_its_mean(self):
        m=self.branch(cf_loss_weight=.1);args=legacy.BIRSTests().inputs()
        out=self.forward(m,args);targets,labels=legacy.BIRSTests().supervision()
        # Only the first of two controls is valid. The second score must not
        # change the supervised gap, even if it is numerically huge.
        out['cf_valid_controls'][...,1]=False
        out['selected_support']=torch.full_like(out['selected_support'],.1)
        out['selected_controls']=torch.stack((torch.zeros_like(out['selected_support']),
                                             torch.full_like(out['selected_support'],1000.)),-1)
        loss,info=m.loss(out,targets,args[-1],labels)
        torch.testing.assert_close(loss['loss_birs_cf'],torch.tensor(.01))
        self.assertEqual(float(info['cf_control_count']),1)

    def test_cf_enabled_inference_needs_no_refer_gt_or_matching_metadata(self):
        m=self.branch(cf_loss_weight=.1,attention_soft_mix=.1)
        args=list(legacy.BIRSTests().inputs())
        args[-1]=[{key:value for key,value in meta.items() if key!='refer_target_index'} for meta in args[-1]]
        m.mark_optimizer_step();m.eval()
        with torch.no_grad():out=self.forward(m,args)
        self.assertTrue(torch.isfinite(out['scores']).all())
        self.assertEqual(set(out['after_orders'][0].tolist()),set(out['before_orders'][0][:len(out['after_orders'][0])].tolist()))

    def test_soft_mix_can_train_past_a_pure_sparsemax_null_dead_zone(self):
        gradients=[]
        for mix in (0.,.1):
            m=self.branch(cf_loss_weight=.1,attention_soft_mix=mix)
            with torch.no_grad():
                m.null_logit.weight.zero_();m.null_logit.bias.fill_(4)
                m.evidence_logit.weight.zero_();m.evidence_logit.bias.fill_(-4)
            args=legacy.BIRSTests().inputs();targets,labels=legacy.BIRSTests().supervision()
            out=self.forward(m,args);loss,info=m.loss(out,targets,args[-1],labels)
            self.assertEqual(float(info['sparse_reference_count']),0)
            loss['loss_birs_cf'].backward()
            gradients.append(float(m.geometry_projection[0].weight.grad.abs().sum()))
        self.assertEqual(gradients[0],0)
        self.assertGreater(gradients[1],0)

    def test_cf_global_valid_sample_normalization(self):
        m=self.branch(cf_loss_weight=.1,attention_soft_mix=.1)
        args=legacy.BIRSTests().inputs();targets,labels=legacy.BIRSTests().supervision()
        local,_=m.loss(self.forward(m,args),targets,args[-1],labels)
        def reduce(counts):counts[:2].copy_(counts.new_tensor([2.,3.]))
        with patch('torch.distributed.is_available',return_value=True), \
             patch('torch.distributed.is_initialized',return_value=True), \
             patch('torch.distributed.get_world_size',return_value=4), \
             patch('torch.distributed.all_reduce',side_effect=reduce):
            global_loss,info=m.loss(self.forward(m,args),targets,args[-1],labels)
        torch.testing.assert_close(global_loss['loss_birs_crs'],local['loss_birs_crs']*2)
        torch.testing.assert_close(global_loss['loss_birs_final'],local['loss_birs_final']*2)
        torch.testing.assert_close(global_loss['loss_birs_cf'],local['loss_birs_cf']*4/3)
        self.assertEqual(float(info['cf_samples']),3)

    def test_cf_configuration_validation(self):
        for options in (dict(cf_loss_weight=-.1),dict(cf_margin=-.1),dict(cf_margin=float('nan')),
                        dict(cf_loss_weight=float('inf')),dict(attention_soft_mix=-.1),dict(attention_soft_mix=1.1)):
            with self.assertRaises(ValueError):self.branch(**options)

    def test_real_head_collects_cf_without_changing_crs_weight(self):
        options=legacy.BIRSTests().head_options();options['birs']=dict(
            version='v3',model_dim=16,num_heads=4,reference_topk=8,max_targets=8,
            cf_loss_weight=.1,cf_margin=.2,attention_soft_mix=.1,checkpoint_chunks=True)
        head=build_head(options)
        x=torch.randn(1,768,8,8)
        metas=[dict(img_shape=(96,128),pad_shape=(128,128),refer_target_index=[0])]
        targets=dict(mask=None,bbox=[torch.tensor([[40.,30.,64.,50.]])],img_metas=metas)
        args=dict(cls_feat=torch.zeros(1,768),lan_feat=torch.randn(1,8,768),
                  lan_mask=torch.tensor([[0,0,0,0,1,1,1,1]]),img=torch.zeros(1,3,128,128))
        loss,_,_=head.forward_train(x,targets,**args)
        self.assertIn('loss_birs_cf',loss)
        self.assertIn('metric_birs_cf_samples',loss)
        self.assertEqual(head.birs.final_mix,.5)
        self.assertEqual(head.loss_weight['refer'],options['loss_weight']['refer'])
        sum(value for key,value in loss.items() if key.startswith('loss_')).backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in head.birs.parameters()))

    def test_quiet_text_log_preserves_losses_tracking_and_training(self):
        class Branch(nn.Module):
            def __init__(self):
                super().__init__();self.weight=nn.Parameter(torch.ones(()))
                self.register_buffer('trained_steps',torch.zeros((),dtype=torch.long))
            def mark_optimizer_step(self):self.trained_steps.add_(1)
        class Toy(nn.Module):
            def __init__(self):
                super().__init__();self.head=nn.Module();self.head.birs=Branch()
                self.base=nn.Parameter(torch.ones(()))
            def forward(self,**kwargs):
                loss={'loss_refer':self.base.square(),'loss_birs_final':self.head.birs.weight.square(),
                      'loss_birs_cf':(self.head.birs.weight-3).square(),
                      'metric_birs_cf_samples':self.base.new_tensor(1.)}
                loss.update({'metric_'+key:self.base.new_tensor(1.)
                             for key in HIDDEN_BIRS_LOG_FIELDS if key.startswith('birs_')})
                return loss,{}
        model=Toy();optimizer=torch.optim.SGD(model.parameters(),lr=.1)
        cfg=SimpleNamespace(distributed=False,use_fp16=False,grad_norm_clip=None,
                            log_interval=1,data=SimpleNamespace(samples_per_gpu=1))
        with patch('propvg.apis.train.extract_data',side_effect=lambda value:value), \
             patch('propvg.apis.train.is_main',return_value=True), \
             patch('propvg.apis.train.get_root_logger') as logger, \
             patch('propvg.apis.train.set_tracking_context'), \
             patch('propvg.apis.train.log_metrics') as tracking:
            train_model(0,cfg,model,optimizer,[{}])
        message=logger.return_value.info.call_args.args[0]
        for key in HIDDEN_BIRS_LOG_FIELDS:self.assertNotIn(key+':',message)
        self.assertIn('loss_refer:',message)
        self.assertIn('loss_birs_cf:',message)
        metrics=tracking.call_args.args[0]
        for key in HIDDEN_BIRS_LOG_FIELDS:
            self.assertIn(key,metrics)
            self.assertIn('train/'+key,metrics)
        self.assertEqual(metrics['train/loss_total'],6.)
        self.assertAlmostEqual(float(model.base),.8,places=5)
        # Removing final BCE from optimization would give 1.4, not 1.2.
        self.assertAlmostEqual(float(model.head.birs.weight),1.2,places=5)
        self.assertEqual(int(model.head.birs.trained_steps),1)

    def test_shortlist_postprocessing_aligns_clipping_scores_nms_and_correctness(self):
        from propvg.models.det_seg.mix_ref_unimodel_OMG import MIXRefUniModel_OMG
        for nms in [False,True]:
            model=MIXRefUniModel_OMG.__new__(MIXRefUniModel_OMG);nn.Module.__init__(model)
            model.with_nms=nms
            pred=dict(pred_class_allobj=torch.tensor([[[1.,0.],[2.,0.],[3.,0.],[4.,0.]]]),
                      pred_bbox_allobj=torch.tensor([[[.2,.2,.1,.1],[1.2,.3,.1,.1],[.7,.7,.1,.1],[.8,.8,.1,.1]]]),
                      refer_label=torch.tensor([[[.1],[.9],[.8],[.7]]]),
                      ranking_priority=torch.tensor([[[0.],[3.],[2.],[1.]]]),
                      candidate_output_mask=torch.tensor([[False,True,True,True]]),
                      top1_correct_query_index=torch.tensor([2]),top1_correct_probability=torch.tensor([.6]),
                      top1_correct_logit=torch.tensor([.4]),top1_grouped_entropy=torch.tensor([.2]))
            out=model.get_predictions_parts(pred,[dict(img_shape=(100,100),scale_factor=[2.]*4)],
                                            rescale=True,with_bbox=True)['pred_bboxes'][0]
            self.assertEqual(len(out['boxes']),2)
            torch.testing.assert_close(out['scores'],torch.tensor([.8,.7]))
            torch.testing.assert_close(out['ranking_logits'],torch.tensor([2.,1.]))
            torch.testing.assert_close(out['boxes'],torch.tensor([[32.5,32.5,37.5,37.5],[37.5,37.5,42.5,42.5]]),atol=1e-5,rtol=1e-5)
            if nms:self.assertEqual(len(out['filtered_boxes']),2)

    def test_variable_batch_prefix_and_unique_padding(self):
        m=self.branch(seed_topk=2,min_targets=3)
        args=legacy.BIRSTests().inputs()
        q=args[0].expand(2,-1,-1)
        valid=m.clipped_boxes(args[1],args[-1])[1].expand(2,-1).clone()
        labels=torch.tensor([[0,1,2,2,2,2,2,-1],[0,1,2,2,0,1,0,-1]])
        with patch.object(m,'cluster',return_value=labels):
            _,mask,ids,pad,_=m.select_targets(q,valid,args[5].expand(2,-1,-1))
        self.assertEqual(mask.sum(-1).tolist(),[3,7])
        self.assertEqual(pad.sum(-1).tolist(),[3,7])
        self.assertTrue(all(len(set(row.tolist()))==ids.shape[1] for row in ids))

    def test_real_head_v3_train_test_and_legacy_initialization(self):
        options=legacy.BIRSTests().head_options();options.pop('birs',None);options['enable_birs']=False
        torch.manual_seed(100);baseline=build_head(options);rng=torch.get_rng_state()
        options.update(enable_birs=True,birs=dict(version='v3',model_dim=16,num_heads=4,
                                                reference_topk=8,max_targets=6,checkpoint_chunks=True))
        torch.manual_seed(100);head=build_head(options)
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        for key,value in baseline.state_dict().items():self.assertTrue(torch.equal(value,head.state_dict()[key]),key)
        x=torch.randn(1,768,8,8)
        metas=[dict(img_shape=(96,128),pad_shape=(128,128),refer_target_index=[0])]
        targets=dict(mask=None,bbox=[torch.tensor([[40.,30.,64.,50.]])],img_metas=metas)
        args=dict(cls_feat=torch.zeros(1,768),lan_feat=torch.randn(1,8,768),
                  lan_mask=torch.tensor([[0,0,0,0,1,1,1,1]]),img=torch.zeros(1,3,128,128))
        losses,pred,extra=head.forward_train(x,targets,**args)
        self.assertIn('candidate_output_mask',pred)
        self.assertNotIn('loss_birs_binding',losses)
        self.assertNotIn('loss_birs_crs',losses)
        self.assertIn('loss_refer',losses)
        sum(v for key,v in losses.items() if key.startswith('loss_')).backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in head.birs.parameters()))
        head.birs.mark_optimizer_step();head.eval()
        with torch.no_grad():pred,_=head.forward_test(x,targets=targets,with_mask=False,**args)
        self.assertLessEqual(int(pred['candidate_output_mask'].sum()),6)
        # Prove v3 receives the pre-BCEM decoder references even without Grab.
        q=torch.randn(1,8,256);raw=torch.randn_like(q);boxes=torch.rand(1,8,4);classes=torch.randn(1,8,2)
        with patch.object(head.birs,'forward',return_value={}) as call:
            head._binding_residual(q,boxes,classes,torch.randn(1,8,256),args['lan_mask'],
                                   torch.rand(1,8,1),torch.rand(1,8,1),metas,raw,boxes,classes,None)
        torch.testing.assert_close(call.call_args.kwargs['reference_queries'],head.refer_adapter(raw))


if __name__=='__main__':unittest.main()
