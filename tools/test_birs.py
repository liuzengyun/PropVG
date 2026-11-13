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
        options = dict(hidden_channels=16, model_dim=16, num_heads=4, num_groups=1, reference_topk=8)
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
            _validate_birs_checkpoint(wrapper(self.module(num_groups=2)),state)
        _validate_birs_checkpoint(wrapper(None),{})

    def test_cluster_deterministic_and_all_invalid(self):
        m,args=self.module(num_groups=4),self.inputs()
        valid=torch.ones(1,8,dtype=torch.bool)
        rng=torch.get_rng_state()
        first=m.cluster(args[0],valid)
        torch.testing.assert_close(rng,torch.get_rng_state())
        self.assertTrue(torch.equal(first,m.cluster(args[0],valid)))
        self.assertTrue((m.cluster(args[0],~valid)==-1).all())

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
        on=build_head(dict(options,enable_birs=True,birs=dict(num_groups=1)))
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        for key,value in old.state_dict().items():
            self.assertTrue(torch.equal(value,on.state_dict()[key]),key)
        for key in ("enable_crv","enable_relation_refinement"):
            with self.assertRaises(ValueError):
                build_head(dict(options,enable_birs=True,**{key:True}))

    def test_real_head_cpu_train_and_test(self):
        options=self.head_options()
        options.update(enable_birs=True,birs=dict(num_groups=1,reference_topk=8,model_dim=16,num_heads=4))
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
