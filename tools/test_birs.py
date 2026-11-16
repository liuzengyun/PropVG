"""CPU mathematical, isolation, selection and integration regression tests for BIRS."""
import copy
import itertools
import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

os.environ.setdefault("MPLCONFIGDIR", "/tmp/propvg_matplotlib")
import torch
from torch import nn
from mmcv import Config
from propvg.models import build_head
from propvg.models.heads.binding_residual import BindingInteractionResidualScoring, binding_statistics
from propvg.utils.checkpoint import _validate_birs_checkpoint
from propvg.apis.train import clip_training_gradients, train_model


class BIRSTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def module(self, **kwargs):
        torch.manual_seed(122)
        options = dict(hidden_channels=16, model_dim=16, num_heads=4, target_topn=8, reference_topk=8)
        options.update(kwargs)
        return BindingInteractionResidualScoring(**options)

    def inputs(self):
        torch.manual_seed(123)
        q = torch.randn(1, 8, 16, requires_grad=True)
        boxes = torch.tensor([[[.15,.5,.1,.1],[.151,.5,.1,.1],[.4,.5,.1,.1],[.7,.5,.1,.1],
                               [.85,.2,.1,.1],[.5,.5,.99,.9],[.99,.99,.2,.2],[1.2,.1,.1,.1]]], requires_grad=True)
        classes = torch.stack((torch.tensor([0.,1.,4.,3.,2.,-2.,1.,10.]), torch.zeros(8)), -1)[None].requires_grad_()
        text = torch.randn(1,6,16,requires_grad=True)
        mask = torch.tensor([[0,0,0,1,1,1]])
        z = torch.tensor([[[2.],[1.9],[1.8],[1.7],[.1],[-1.],[-2.],[3.]]], requires_grad=True)
        metas = [dict(img_shape=(100,200), pad_shape=(256,256), refer_target_index=[0])]
        return q,boxes,classes,text,mask,z.sigmoid(),z,metas

    def supervision(self):
        return [torch.tensor([[20.,45.,40.,55.], [100.,1.,110.,10.]])], torch.tensor([[1.,0,0,0,0,0,0,0]])

    def test_exact_permutation_expectation(self):
        torch.manual_seed(5)
        h, g = torch.randn(2,4,8,dtype=torch.double), torch.randn(2,4,8,dtype=torch.double)
        residual, paired, null = binding_statistics(h,g,torch.ones(2,4,dtype=torch.bool))
        values = [(h*g[:,p,:]).sum((-2,-1))/4/(8**.5) for p in itertools.permutations(range(4))]
        torch.testing.assert_close(null,torch.stack(values).mean(0))
        torch.testing.assert_close(residual,paired-null)

    def test_additive_scores_cancel(self):
        # Explicit embedding whose bilinear score is a_j + b_k + c.
        h = torch.tensor([[[1.,2.,1.],[3.,2.,1.],[5.,2.,1.]]])
        g = torch.tensor([[[1.,1.,2.],[1.,1.,4.],[1.,1.,7.]]])
        residual, _, _ = binding_statistics(h,g,torch.ones(1,3,dtype=torch.bool))
        torch.testing.assert_close(residual, torch.zeros(1), rtol=0, atol=0)

    def test_masks_single_reference_and_identical_semantics(self):
        h,g=torch.randn(1,4,8),torch.randn(1,4,8)
        valid=torch.tensor([[1,1,0,0]],dtype=torch.bool)
        torch.testing.assert_close(binding_statistics(h,g,valid)[0],binding_statistics(h[:,:2],g[:,:2],valid[:,:2])[0])
        self.assertEqual(binding_statistics(h,g,torch.tensor([[1,0,0,0]],dtype=torch.bool))[0].item(),0)
        h=h[:,:1].expand_as(h)
        torch.testing.assert_close(binding_statistics(h,g,valid)[0],torch.zeros(1),rtol=0,atol=0)
        torch.testing.assert_close(binding_statistics(h[:,:3],g[:,:3],torch.ones(1,3,dtype=torch.bool))[0],
                                   torch.zeros(1),rtol=0,atol=0)

    def test_binding_audit_negatives_and_crs_control(self):
        args=self.inputs();targets,labels=self.supervision()
        m=self.module()
        with m.diagnostics():
            out=m(*args)
            m.loss(out,targets,args[-1],labels)
        for pair in out["diagnostics"]["binding_pairs"]:
            bi,pos,neg=pair["sample"],pair["positive"],pair["negative"]
            iou=out["diagnostics"]["gt_iou"][bi]
            self.assertGreaterEqual(float(iou[pos]),.5)
            self.assertTrue((iou[neg]==0).all())
        control=self.module(binding_mode="crs_control")
        out=control(*args)
        torch.testing.assert_close(out["scores"],args[5],rtol=0,atol=0)
        losses,_=control.loss(out,targets,args[-1],labels)
        self.assertEqual(float(losses["loss_birs_binding"]),0)

    def test_topk_before_dedup_and_source_self(self):
        m,args=self.module(reference_topk=6),self.inputs()
        out=m(*args)
        self.assertEqual(out["reference_indices"].tolist(),[[2,3,4,1,6,0]])
        refs=out["reference_indices"][0].tolist()
        self.assertFalse(out["reference_valid"][0,refs.index(0)])
        self.assertFalse(out["legal_references"][0,0,refs.index(1)])
        self.assertEqual(out["scores"].shape,(1,8,1))
        # Strict no-overlap was NOT applied to real contextual objects.
        all_refs=self.module()(*args)
        self.assertTrue(all_refs["legal_references"][0,0,all_refs["reference_indices"][0].tolist().index(5)])

    def test_padding_and_chunk_invariance(self):
        m,args=self.module(),self.inputs()
        first=m(*args)
        args[-1][0]["pad_shape"]=(1000,2000)
        m.chunk_size=1
        second=m(*args)
        torch.testing.assert_close(first["raw_binding"],second["raw_binding"],rtol=1e-5,atol=1e-6)
        torch.testing.assert_close(first["scores"],second["scores"])
        groups=first["groups"]; delta=first["delta"]
        for group in groups.unique():
            if int(group)>=0:
                self.assertLess(abs(float(delta[(groups==group)&first["eligible"]].sum())),1e-6)

    def test_gradients_joint_and_detached(self):
        for detach in (False,True):
            m,args=self.module(detach_inputs=detach),self.inputs()
            targets,labels=self.supervision()
            out=m(*args)
            losses,metrics=m.loss(out,targets,args[-1],labels)
            self.assertEqual(metrics["binding_samples"].item(),1)
            sum(losses.values()).backward()
            for index in (0,3,6):
                grad=args[index].grad
                if detach:
                    self.assertTrue(grad is None or grad.abs().sum()==0)
                else:
                    self.assertIsNotNone(grad)
                    self.assertGreater(float(grad.abs().sum()),0)
            self.assertIsNone(args[1].grad)
            self.assertIsNone(args[2].grad)
            self.assertGreater(float(m.semantic_binding[0].weight.grad.abs().sum()),0)
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters()))

    def test_auxiliary_gt_does_not_affect_loss(self):
        m,args=self.module(),self.inputs()
        targets,labels=self.supervision()
        out=m(*args)
        first,_=m.loss(out,targets,args[-1],labels)
        targets[0][1]=torch.tensor([0.,0.,200.,100.])
        second,_=m.loss(out,targets,args[-1],labels)
        for key in first:
            torch.testing.assert_close(first[key],second[key],rtol=0,atol=0)

    def test_all_padding_empty_references_and_no_referent(self):
        m,args=self.module(),list(self.inputs())
        args[4]=torch.ones_like(args[4])
        out=m(*args)
        torch.testing.assert_close(out["scores"],args[5],rtol=0,atol=0)
        self.assertFalse(out["eligible"].any())
        args[-1][0]["refer_target_index"]=[]
        losses,metrics=m.loss(out,[torch.empty(0,4)],args[-1],torch.zeros(1,8))
        self.assertEqual(metrics["supervised_samples"].item(),0)
        sum(losses.values()).backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters()))

    def test_eval_untrained_fallback_diagnostics_roundtrip(self):
        m,args=self.module(),self.inputs()
        m.eval()
        out=m(*args)
        self.assertIs(out["scores"],args[5])
        self.assertIsNone(out["ranking_priority"])
        m.mark_optimizer_step()
        with m.diagnostics():
            out=m(*args)
        self.assertIn("reference_boxes",out["diagnostics"])
        self.assertFalse(m.capture_diagnostics)
        clone=self.module().eval()
        clone.load_state_dict(m.state_dict(),strict=True)
        torch.testing.assert_close(clone(*args)["scores"],out["scores"],rtol=0,atol=0)

    def test_checkpoint_policy(self):
        def wrapper(branch):
            result=nn.Module(); result.head=nn.Module(); result.head.birs=branch
            return result
        model=wrapper(self.module())
        state=model.state_dict()
        _validate_birs_checkpoint(model,state)
        with self.assertRaises(ValueError):
            _validate_birs_checkpoint(model,{})
        with self.assertRaises(ValueError):
            _validate_birs_checkpoint(wrapper(self.module(target_topn=2)),state)
        _validate_birs_checkpoint(wrapper(None),{})

    def test_deterministic_forward_and_all_invalid(self):
        m,args=self.module(),list(self.inputs())
        rng=torch.get_rng_state()
        first=m(*args)
        torch.testing.assert_close(rng,torch.get_rng_state())
        second=m(*args)
        torch.testing.assert_close(first["scores"],second["scores"],rtol=0,atol=0)
        args[1]=torch.zeros_like(args[1])
        out=m(*args)
        self.assertFalse(out["valid"].any())
        self.assertEqual(len(out["after_orders"][0]),0)

    def test_topn_no_clustering_full_reference_pool_and_membership(self):
        m,args=self.module(target_topn=5),self.inputs()
        self.assertEqual(m.version, "v2")
        m.cluster = lambda *args: self.fail("v2 must not execute clustering")
        out=m(*args)
        self.assertEqual(out["target_indices"].tolist(),[[0,1,2,3,4]])
        self.assertEqual(out["scores"].shape,(1,8,1))
        self.assertEqual(int(out["candidate_mask"].sum()),5)
        self.assertIn(6,out["reference_indices"][0].tolist())
        self.assertEqual(set(out["before_orders"][0][:5].tolist()),set(out["after_orders"][0][:5].tolist()))
        torch.testing.assert_close(out["before_orders"][0][5:],out["after_orders"][0][5:],rtol=0,atol=0)
        outside=~out["candidate_mask"]
        torch.testing.assert_close(out["scores"][outside],args[5][outside],rtol=0,atol=0)
        self.assertTrue((out["delta"][outside]==0).all())
        # Simulate an extreme residual that pushes candidate logits below an
        # outsider: priorities, not probabilities, enforce the hard boundary.
        def forced(module,inputs,result):
            return result*300
        handle=m.geometry_binding.register_forward_hook(forced)
        exaggerated=m(*args);handle.remove()
        self.assertEqual(set(exaggerated["after_orders"][0][:5].tolist()),{0,1,2,3,4})

    def test_topn_losses_scope_and_missing_matched_positive(self):
        m,args=self.module(target_topn=5),self.inputs()
        targets,labels=self.supervision();out=m(*args)
        losses,info=m.loss(out,targets,args[-1],labels)
        expected=torch.nn.functional.binary_cross_entropy_with_logits(
            out["final_logits"].squeeze(-1)[out["valid"]],labels[out["valid"]])
        torch.testing.assert_close(losses["loss_birs_final"],expected)
        z=torch.zeros_like(labels);z[0,6]=1
        missing,info=m.loss(out,targets,args[-1],z)
        expected=torch.nn.functional.binary_cross_entropy_with_logits(
            out["final_logits"].squeeze(-1)[out["valid"]],z[out["valid"]])
        torch.testing.assert_close(missing["loss_birs_final"],expected)
        self.assertEqual(float(missing["loss_birs_binding"]),0)
        self.assertEqual(float(info["refer_in_topn"]),0)
        self.assertEqual(float(info["supervised_samples"]),1)
        sum(missing.values()).backward()
        self.assertNotEqual(float(args[6].grad[0,6]),0)
        self.assertTrue(all(p.grad is not None for p in m.parameters()))

    def test_topn_short_samples_unique_padding_and_original_source_mapping(self):
        m,args=self.module(target_topn=5),list(self.inputs())
        original_boxes=args[1].detach().clone();original_q=args[0];original_class=args[2]
        args[1]=original_boxes.clone();args[1][:,3:]=0
        out=m(*args)
        ids=out["target_indices"][0]
        self.assertEqual(len(ids.unique()),5)
        self.assertEqual(int(out["target_valid"].sum()),3)
        self.assertEqual(len(out["after_orders"][0]),3)
        # Grab-like pruned target pool with explicit full-detector query IDs.
        full=self.inputs();source=torch.tensor([[2,3,4,6]])
        pruned=[x[:,source[0]] if i in (0,1,2,5,6) else x for i,x in enumerate(full)]
        out=m(*pruned,reference_queries=full[0],reference_boxes=full[1],reference_classes=full[2],source_indices=source)
        for local,query in enumerate(source[0]):
            matches=out["reference_indices"][0]==query
            self.assertFalse(out["legal_references"][0,local,matches].any())

    def test_topn_checkpoint_policy_and_detachment(self):
        m=self.module(target_topn=5)
        self.assertEqual(int(m.schema_version),3)
        other=self.module(target_topn=8)
        self.assertFalse(torch.equal(m.policy_signature,other.policy_signature))
        with self.assertRaises(TypeError):
            self.module(candidate_mode="cluster")
        with self.assertRaises(ValueError):
            self.module(version="unknown")
        wrapper=nn.Module();wrapper.head=nn.Module();wrapper.head.birs=m
        state={k:v.clone() for k,v in wrapper.state_dict().items()}
        state["head.birs.schema_version"].fill_(2)
        with self.assertRaises(ValueError):
            _validate_birs_checkpoint(wrapper,state)
        m,args=self.module(target_topn=5,detach_inputs=True),self.inputs()
        t,l=self.supervision();out=m(*args);losses,_=m.loss(out,t,args[-1],l)
        sum(losses.values()).backward()
        self.assertTrue(all(args[i].grad is None or args[i].grad.abs().sum()==0 for i in (0,1,2,3,6)))
        self.assertGreater(float(m.semantic_binding[0].weight.grad.abs().sum()),0)

    def test_version1_clusters_all_candidates_and_separates_checkpoint_policy(self):
        m, args = self.module(version="v1", num_groups=2, target_topn=1), self.inputs()
        rng = torch.get_rng_state()
        out = m(*args)
        torch.testing.assert_close(rng, torch.get_rng_state())
        self.assertEqual(int(m.schema_version), 1)
        self.assertTrue(torch.equal(out["candidate_mask"], out["valid"]))
        self.assertEqual(int(out["candidate_mask"].sum()), 7)
        self.assertEqual(out["groups"][0, 7].item(), -1)
        self.assertTrue(((out["groups"][out["valid"]] >= 0) & (out["groups"][out["valid"]] < 2)).all())
        self.assertTrue(torch.equal(out["groups"], m.cluster(args[0], out["valid"])))
        for i in out["groups"][out["valid"]].unique():
            use = out["eligible"] & (out["groups"] == i)
            self.assertLess(float(out["delta"][use].sum().abs()), 1e-6)
        t, labels = self.supervision()
        losses, _ = m.loss(out, t, args[-1], labels)
        sum(losses.values()).backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters()))
        wrapper = nn.Module(); wrapper.head = nn.Module(); wrapper.head.birs = m
        _validate_birs_checkpoint(wrapper, wrapper.state_dict())
        wrong = nn.Module(); wrong.head = nn.Module(); wrong.head.birs = self.module(version="v2")
        with self.assertRaises(ValueError):
            _validate_birs_checkpoint(wrong, wrapper.state_dict())

    def test_v2_default_unchanged_and_v1_untrained_and_empty_fallback(self):
        default, explicit = self.module(), self.module(version="v2", num_groups=3, kmeans_iters=2)
        for key, value in default.state_dict().items():
            torch.testing.assert_close(value, explicit.state_dict()[key], rtol=0, atol=0)
        args = list(self.inputs())
        out = self.module(version="v1").eval()(*args)
        self.assertIsNone(out["ranking_priority"])
        self.assertTrue(torch.equal(out["scores"], args[5]))
        args[1] = torch.zeros_like(args[1])
        m = self.module(version="v1")
        out = m(*args)
        t, labels = self.supervision()
        losses, metrics = m.loss(out, t, args[-1], labels)
        self.assertEqual(float(metrics["supervised_samples"]), 0)
        self.assertEqual(float(sum(losses.values())), 0)
        sum(losses.values()).backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters()))

    def test_full_bce_survives_no_binding_evidence_and_uses_all_candidates(self):
        m,args=self.module(target_topn=5),list(self.inputs())
        args[4]=torch.ones_like(args[4])
        out=m(*args);targets,labels=self.supervision()
        losses,metrics=m.loss(out,targets,args[-1],labels)
        expected=torch.nn.functional.binary_cross_entropy_with_logits(
            args[6].squeeze(-1)[out["valid"]],labels[out["valid"]])
        torch.testing.assert_close(losses["loss_birs_final"],expected)
        self.assertEqual(float(metrics["supervised_samples"]),1)
        self.assertEqual(float(metrics["binding_samples"]),0)
        sum(losses.values()).backward()
        self.assertTrue(all(p.grad is not None for p in m.parameters()))

    def test_ddp_global_count_scaling_and_rank_without_supervision(self):
        m,args=self.module(),self.inputs()
        targets,labels=self.supervision()
        output=m(*args)
        single,_=m.loss(output,targets,args[-1],labels)
        # Same non-empty data on both ranks: world/global_count preserves
        # each rank's single-process gradient after DDP averaging.
        with patch("torch.distributed.is_initialized",return_value=True), \
             patch("torch.distributed.get_world_size",return_value=2), \
             patch("torch.distributed.all_reduce",side_effect=lambda x:x.mul_(2)):
            doubled,_=m.loss(output,targets,args[-1],labels)
        for key in single:
            torch.testing.assert_close(single[key],doubled[key])
        # Other rank has no positives: this rank contributes twice its local
        # loss, so DDP averaging yields the global non-empty-sample mean.
        with patch("torch.distributed.is_initialized",return_value=True), \
             patch("torch.distributed.get_world_size",return_value=2), \
             patch("torch.distributed.all_reduce",side_effect=lambda x:x):
            heterogeneous,_=m.loss(output,targets,args[-1],labels)
        for key in single:
            torch.testing.assert_close(single[key]*2,heterogeneous[key])

    def test_detached_branch_has_independent_gradient_clipping(self):
        model=nn.Module(); model.head=nn.Module()
        model.base=nn.Parameter(torch.zeros(1)); model.head.birs=self.module(detach_inputs=True)
        model.base.grad=torch.tensor([3.])
        model.head.birs.strength_logit.grad=torch.tensor(4.)
        clip_training_gradients(model,1.)
        self.assertAlmostEqual(float(model.base.grad),1.,places=5)
        self.assertAlmostEqual(float(model.head.birs.strength_logit.grad),1.,places=5)

    def test_trainer_optimizes_both_new_losses_and_marks_step(self):
        class ToyModel(nn.Module):
            def __init__(self,branch):
                super().__init__(); self.head=nn.Module(); self.head.birs=branch
                self.base=nn.Parameter(torch.zeros(1))

            def forward(self,**kwargs):
                return {"loss_det":(self.base-1).square().sum(),
                        "loss_birs_final":(self.head.birs.strength_logit-1).square(),
                        "loss_birs_binding":(self.head.birs.strength_logit-2).square(),
                        "metric_birs_supervised_samples":self.base.new_tensor(1.)}, {}

        model=ToyModel(self.module())
        before=float(model.head.birs.strength_logit)
        cfg=SimpleNamespace(distributed=False,use_fp16=False,grad_norm_clip=None)
        optimizer=torch.optim.SGD(model.parameters(),lr=.01)
        with patch("propvg.apis.train.extract_data",side_effect=lambda x:x), \
             patch("propvg.apis.train.is_main",return_value=False):
            train_model(0,cfg,model,optimizer,[{}])
        expected=before-.01*(2*(before-1)+2*(before-2))
        self.assertAlmostEqual(float(model.head.birs.strength_logit),expected,places=5)
        self.assertEqual(int(model.head.birs.trained_steps),1)

    def test_linear_offset_cancellation_but_nonlinear_geometry_is_target_specific(self):
        refs=torch.tensor([[[.2,.2,.3,.3],[.6,.4,.7,.5],[.8,.8,.9,.9]]])
        targets=torch.tensor([[[.1,.1,.2,.2],[.4,.5,.5,.6]]])
        geometry=self.module().geometry(targets,refs,torch.ones(1,2))
        centered=geometry-geometry.mean(2,keepdim=True)
        torch.testing.assert_close(centered[:,0,:,:2],centered[:,1,:,:2])
        self.assertGreater(float((centered[:,0,:,2:]-centered[:,1,:,2:]).abs().sum()),.1)

    def head_options(self):
        options=copy.deepcopy(Config.fromfile("configs/aerialvg/PropVG-aerialvg.py").model.head)
        options.pop("birs", None)  # Legacy fixtures do not inherit the active v3 policy.
        options.update(enable_grab=False,enable_bcem=False,enable_crv=False,enable_correctness=False,
                       enable_relation_refinement=False,num_queries=8)
        return options

    def test_disabled_state_rng_and_enabled_legacy_parameters(self):
        options=self.head_options(); options.pop("enable_birs",None); options.pop("birs",None)
        torch.manual_seed(100)
        old=build_head(options); rng=torch.get_rng_state()
        torch.manual_seed(100)
        off=build_head(dict(options,enable_birs=False))
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        self.assertEqual(set(old.state_dict()),set(off.state_dict()))
        off.load_state_dict(old.state_dict(),strict=True)
        torch.manual_seed(100)
        on=build_head(dict(options,enable_birs=True,birs=dict(target_topn=5)))
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        for key,value in old.state_dict().items():
            self.assertTrue(torch.equal(value,on.state_dict()[key]),key)
        for key in ("enable_crv","enable_relation_refinement"):
            with self.assertRaises(ValueError):
                build_head(dict(options,enable_birs=True,**{key:True}))

    def test_real_head_cpu_train_and_test(self):
        options=self.head_options()
        options.update(enable_birs=True,birs=dict(target_topn=5,reference_topk=8,model_dim=16,num_heads=4))
        head=build_head(options)
        x=torch.randn(1,768,8,8)
        metas=[dict(img_shape=(96,128),pad_shape=(128,128),refer_target_index=[0])]
        targets=dict(mask=None,bbox=[torch.tensor([[40.,30.,64.,50.]])],img_metas=metas)
        args=dict(cls_feat=torch.zeros(1,768),lan_feat=torch.randn(1,8,768),
                  lan_mask=torch.tensor([[0,0,0,0,1,1,1,1]]),img=torch.zeros(1,3,128,128))
        losses,pred,extra=head.forward_train(x,targets,**args)
        self.assertIn("loss_birs_final",losses)
        self.assertIn("loss_birs_binding",losses)
        sum(v for k,v in losses.items() if k.startswith("loss_")).backward()
        self.assertTrue(all(p.grad is not None for p in head.birs.parameters()))
        head.eval()
        with torch.no_grad():
            pred,_=head.forward_test(x,targets=targets,with_mask=False,**args)
        self.assertNotIn("ranking_priority",pred)  # No optimizer step yet.
        head.birs.mark_optimizer_step()
        with torch.no_grad():
            pred,_=head.forward_test(x,targets=targets,with_mask=False,**args)
        self.assertEqual(pred["ranking_priority"].shape,(1,8,1))


if __name__ == "__main__":
    unittest.main()
