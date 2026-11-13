"""CPU checks: reference selection, geometry, gradients, ranking and audit API."""
import copy
import csv
import os
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

os.environ.setdefault("MPLCONFIGDIR", "/tmp/propvg_matplotlib")
import numpy as np
import torch
from mmcv import Config

from propvg.models import build_head
from propvg.models.heads.two_stage_relation import TwoStageRelationRefinement, masked_softmax
from propvg.utils.checkpoint import _validate_relation_refinement_checkpoint, load_pretrained_checkpoint
from propvg.evaluations.two_stage_relation import sample_record, summarize
from analyze_two_stage_relation import original_baseline_comparison, write_reports


class TwoStageRelationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def module(self, **options):
        torch.manual_seed(1122)
        return TwoStageRelationRefinement(**dict(dict(hidden_channels=16, model_dim=16, num_heads=4,
                                                      target_topk=4, reference_topk=6), **options))

    def inputs(self, batch=1):
        torch.manual_seed(3344)
        boxes = torch.tensor([[[.15,.5,.1,.1], [.151,.5,.1,.1], [.4,.5,.1,.1], [.7,.5,.1,.1],
                               [.85,.2,.1,.1], [.5,.5,.99,.9], [.99,.99,.2,.2], [1.2,.1,.1,.1]]]).repeat(batch,1,1).requires_grad_()
        q = torch.randn(batch,8,16,requires_grad=True)
        text = torch.randn(batch,6,16,requires_grad=True)
        mask = torch.tensor([[0,0,0,1,1,1]]).repeat(batch,1)
        z = torch.tensor([[[2.],[1.9],[1.8],[1.7],[.1],[-1.],[-2.],[3.]]]).repeat(batch,1,1).requires_grad_()
        classes = torch.stack((torch.tensor([0.,1.,4.,3.,2.,-2.,1.,10.]), torch.zeros(8)), -1)[None].repeat(batch,1,1).requires_grad_()
        metas = [dict(img_shape=(100,200),pad_shape=(256,256),refer_target_index=[0]) for _ in range(batch)]
        return q,boxes,classes,text,mask,z.sigmoid(),z,metas

    def test_topk_before_dedup_no_refill_and_never_crs_reference_filter(self):
        m, args = self.module(), self.inputs()
        out = m(*args)
        s = out["layout"]
        self.assertEqual(s["target_indices"][0].tolist(), [0,1,2,3])
        self.assertEqual(s["group_slots"][0,s["group_valid"][0]].tolist(), [0,2,3])
        self.assertEqual(s["target_groups"][0].tolist(), [0,0,1,2])
        self.assertEqual(s["pre_nms_indices"][0].tolist(), [2,3,4,1,6,0])
        kept = s["reference_indices"][0,s["reference_valid"][0]].tolist()
        self.assertEqual(kept, [2,3,4,1,6])
        self.assertNotIn(5, kept)
        self.assertIn(4, kept)  # Low target-CRS score, valid contextual candidate.

    def test_self_duplicates_blocked_but_real_overlap_allowed(self):
        m, args = self.module(reference_topk=8), self.inputs()
        out = m(*args)
        refs = out["layout"]["reference_indices"][0].tolist()
        self.assertFalse(out["pair_valid"][0,0,refs.index(1)])
        self.assertTrue(out["pair_valid"][0,0,refs.index(5)])  # Large enclosing region.

    def test_padding_does_not_enter_geometry(self):
        m, args = self.module(), self.inputs()
        first = m(*args)
        args[-1][0]["pad_shape"] = (1000,2000)
        other = m(*args)
        torch.testing.assert_close(first["geometry"], other["geometry"], rtol=0, atol=0)
        g = m.geometry_vector(torch.tensor([[[.1,.4,.2,.6]]]), torch.tensor([[[.3,.1,.5,.3]]]))
        torch.testing.assert_close(g[0,0,0,:2], torch.tensor([.25,-.3]))
        self.assertAlmostEqual(float(g[0,0,0,3]), np.log(2), places=5)
        self.assertAlmostEqual(float(g[0,0,0,4]), 0., places=5)

    def test_votes_normalize_across_targets_masks_are_zero(self):
        out = self.module()(*self.inputs())
        expected = out["pair_valid"].any(1).float()
        torch.testing.assert_close(out["votes"].sum(1), expected)
        self.assertTrue((out["votes"][~out["pair_valid"]] == 0).all())
        torch.testing.assert_close(out["reference_weights"].sum(1)+out["null_weight"], torch.ones(1))

    def test_uniform_evidence_is_exactly_neutral_and_has_gradient(self):
        m = self.module()
        e = torch.zeros(1,3,2,requires_grad=True)
        raw, info = m.aggregate(e, torch.ones_like(e,dtype=torch.bool), torch.zeros(1,2), torch.zeros(1,1), torch.zeros(1,3))
        torch.testing.assert_close(raw, torch.zeros_like(raw), rtol=0, atol=0)
        raw[0,0].backward()
        self.assertGreater(float(e.grad.abs().sum()), 0)
        self.assertLess(float(info["discriminability"].max()), 1e-12)

    def test_competition_prior_has_uniform_floor_and_is_detached(self):
        m = self.module(prior_uniform_mix=.2)
        z = torch.tensor([[100.,-100.,-100.]],requires_grad=True)
        e = torch.tensor([[[2.],[0.],[-1.]]],requires_grad=True)
        raw, info = m.aggregate(e, torch.ones_like(e,dtype=torch.bool), torch.zeros(1,1), torch.zeros(1,1), z)
        self.assertGreaterEqual(float(info["prior"].min()), .2/3-1e-6)
        raw[0,0].backward()
        self.assertIsNone(z.grad)
        self.assertGreater(float(e.grad.abs().sum()), 0)

    def test_reference_support_that_cannot_separate_competitors_has_low_discrimination(self):
        m = self.module(prior_uniform_mix=0.)
        valid = torch.ones(1,32,1,dtype=torch.bool)
        e = torch.full((1,32,1), -100.)
        e[:,:2] = 0.
        z = torch.full((1,32), -100.)
        z[:,:2] = 0.
        _, info = m.aggregate(e,valid,torch.zeros(1,1),torch.zeros(1,1),z)
        self.assertLess(float(info["discriminability"].max()), 1e-6)

    def test_null_option_can_suppress_reference_without_language_parser(self):
        m = self.module()
        e = torch.tensor([[[3.],[0.],[-3.]]])
        valid = torch.ones_like(e,dtype=torch.bool)
        active, _ = m.aggregate(e,valid,torch.zeros(1,1),torch.tensor([[-100.]]),torch.zeros(1,3))
        absent, info = m.aggregate(e,valid,torch.zeros(1,1),torch.tensor([[100.]]),torch.zeros(1,3))
        self.assertGreater(float(active.abs().sum()), .01)
        self.assertLess(float(absent.abs().sum()), 1e-20)
        self.assertAlmostEqual(float(info["null_weight"]),1.)

    def test_all_invalid_boxes_empty_text_and_one_group_are_finite_identity(self):
        for case in ("invalid", "empty_text", "one_group"):
            args = list(self.inputs())
            m = self.module(target_topk=1 if case == "one_group" else 4)
            if case == "invalid":
                args[1] = torch.zeros_like(args[1])
            if case == "empty_text":
                args[4] = torch.ones_like(args[4])
            out = m(*args)
            self.assertTrue(torch.isfinite(out["final_logits"]).all())
            torch.testing.assert_close(out["delta"], torch.zeros_like(out["delta"]), rtol=0,atol=0)
            self.assertEqual(out["final_orders"][0].tolist(), out["layout"]["crs_orders"][0].tolist())

    def test_all_masked_softmax_is_finite(self):
        values = torch.randn(2,4,3,requires_grad=True)
        out = masked_softmax(values,torch.zeros_like(values,dtype=torch.bool),1)
        self.assertTrue(torch.isfinite(out).all())
        self.assertTrue((out == 0).all())
        out.sum().backward()
        self.assertTrue(torch.isfinite(values.grad).all())

    def test_forward_does_not_read_gt_and_capture_does_not_change_outputs(self):
        m, args = self.module().eval(), list(self.inputs())
        before = m(*args)
        args[-1][0].update(refer_target_index=[999999], gt_bbox="MUST NOT READ")
        with m.diagnostics():
            after = m(*args)
            self.assertTrue(m.capture_diagnostics)
        self.assertFalse(m.capture_diagnostics)
        for key in ("scores","ranking_priority","final_logits","pair_logits"):
            torch.testing.assert_close(after[key],before[key],rtol=0,atol=0)
        self.assertTrue(all(not v.requires_grad and v.device.type == "cpu" for v in after["diagnostics"][0].values() if torch.is_tensor(v)))

    def test_diagnostics_context_restores_after_exception(self):
        m = self.module()
        with self.assertRaises(RuntimeError):
            with m.diagnostics():
                raise RuntimeError("stop")
        self.assertFalse(m.capture_diagnostics)

    def test_joint_loss_gradients_reach_features_text_crs_not_box_coordinates(self):
        m, args = self.module(), self.inputs()
        out = m(*args)
        loss, metrics = m.loss(out,[torch.tensor([[20.,45.,40.,55.]])],args[-1])
        self.assertEqual(float(metrics["supervised_samples"]),1.)
        loss.backward()
        for tensor in (args[0],args[3],args[6]):
            self.assertIsNotNone(tensor.grad)
            self.assertGreater(float(tensor.grad.abs().sum()),0.)
        self.assertIsNone(args[1].grad)
        self.assertIsNone(args[2].grad)
        self.assertGreater(float(m.pair_score.weight.grad.abs().sum()),0.)
        self.assertGreater(float(m.strength_logit.grad.abs()),0.)

    def test_detach_blocks_all_base_inputs_but_trains_new_scoring_network(self):
        m,args=self.module(detach_inputs=True),self.inputs()
        # Include an independent, unfiltered reference source as used by Grab.
        references=args[0].detach().clone().requires_grad_()
        out=m(*args,reference_queries=references)
        loss,_=m.loss(out,[torch.tensor([[20.,45.,40.,55.]])],args[-1])
        loss.backward()
        for tensor in (args[0],args[1],args[2],args[3],args[6],references):
            self.assertIsNone(tensor.grad)
        for branch in (m.visual,m.text,m.reader,m.pair_score):
            self.assertGreater(sum(float(p.grad.abs().sum()) for p in branch.parameters() if p.grad is not None),0.)
        self.assertGreater(float(m.strength_logit.grad.abs()),0.)

    def test_detach_is_checkpoint_compatible_and_forward_numerically_identical(self):
        joint=self.module(detach_inputs=False).eval()
        detached=self.module(detach_inputs=True).eval()
        detached.load_state_dict(joint.state_dict(),strict=True)
        self.assertEqual(list(joint.state_dict()),list(detached.state_dict()))
        first,second=joint(*self.inputs()),detached(*self.inputs())
        for key in ("scores","ranking_priority","selected_logits","final_logits","delta","pair_logits","votes"):
            torch.testing.assert_close(first[key],second[key],rtol=0,atol=0)
        wrapped=torch.nn.Module();wrapped.head=torch.nn.Module();wrapped.head.relation_refiner=detached
        _validate_relation_refinement_checkpoint(wrapped,{'head.relation_refiner.'+k:v for k,v in joint.state_dict().items()})

    def test_multi_positive_mass_loss_does_not_force_duplicate_positives_apart(self):
        m, args = self.module(), self.inputs()
        out = m(*args)
        loss, _ = m.loss(out,[torch.tensor([[20.,45.,40.,55.]])],args[-1])
        z = out["final_logits"][0]
        expected = torch.logsumexp(z,0)-torch.logsumexp(z[:2],0)
        torch.testing.assert_close(loss,expected)

    def test_unreachable_and_empty_gt_losses_are_connected_zero(self):
        for empty in (False,True):
            m, args = self.module(), self.inputs()
            if empty:
                args[-1][0]["refer_target_index"] = []
            out = m(*args)
            loss, metrics = m.loss(out,[torch.tensor([[0.,0.,1.,1.]])],args[-1])
            self.assertEqual(float(loss),0.)
            self.assertEqual(float(metrics["supervised_samples"]),0.)
            loss.backward()
            self.assertTrue(torch.isfinite(args[6].grad).all())

    def test_ddp_global_valid_mean_compensates_gradient_averaging(self):
        m, args = self.module(), self.inputs(batch=2)
        out = m(*args)
        targets = [torch.tensor([[20.,45.,40.,55.]])]*2
        expected, _ = m.loss(out,targets,args[-1])
        with patch("torch.distributed.is_available",return_value=True), patch("torch.distributed.is_initialized",return_value=True), \
             patch("torch.distributed.get_world_size",return_value=2), patch("torch.distributed.all_reduce",side_effect=lambda x: x.mul_(2)):
            actual, metrics = m.loss(out,targets,args[-1])
        torch.testing.assert_close(actual,expected)
        self.assertEqual(float(metrics["supervised_samples"]),4.)

    def test_refinement_keeps_shortlist_members_and_tail_order(self):
        m, args = self.module(), self.inputs()
        out = m(*args)
        old,new=out["layout"]["crs_orders"][0],out["final_orders"][0]
        self.assertEqual(set(old[:4].tolist()),set(new[:4].tolist()))
        self.assertEqual(old[4:].tolist(),new[4:].tolist())
        ranked=out["ranking_priority"][0,:,0].argsort(descending=True,stable=True)[:len(new)]
        self.assertEqual(ranked.tolist(),new.tolist())
        torch.testing.assert_close(out["delta"][0,0],out["delta"][0,1],rtol=0,atol=0)

    def test_masked_text_tokens_cannot_change_relation_score(self):
        m,args=self.module().eval(),list(self.inputs())
        first=m(*args)
        args[3]=args[3].detach().clone()
        args[3][:,3:]=torch.randn_like(args[3][:,3:])*100
        other=m(*args)
        torch.testing.assert_close(first["final_logits"],other["final_logits"],rtol=0,atol=0)

    def test_all_aggregations_and_geometry_semantics_ablations_are_finite(self):
        for mode in ("competitive","vote","weighted","max","crs_control"):
            for geometry,semantics in ((True,True),(False,True),(True,False)):
                m=self.module(aggregation=mode,use_geometry=geometry,use_reference_semantics=semantics)
                out=m(*self.inputs())
                self.assertTrue(torch.isfinite(out["final_logits"]).all())
                if mode == "crs_control":
                    self.assertTrue((out["delta"] == 0).all())

    def test_checkpoint_exact_policy_and_ddp_keys(self):
        m=torch.nn.Module();m.head=torch.nn.Module();m.head.relation_refiner=self.module()
        _validate_relation_refinement_checkpoint(m,m.state_dict())
        _validate_relation_refinement_checkpoint(SimpleNamespace(module=m),{"module."+k:v for k,v in m.state_dict().items()})
        with self.assertRaisesRegex(ValueError,"complete checkpoint"):
            _validate_relation_refinement_checkpoint(m,{})
        other=torch.nn.Module();other.head=torch.nn.Module();other.head.relation_refiner=self.module(prior_uniform_mix=.3)
        with self.assertRaisesRegex(ValueError,"policy differs"):
            _validate_relation_refinement_checkpoint(other,m.state_dict())
        m.head.relation_refiner=None
        _validate_relation_refinement_checkpoint(m,{})

    def test_finetuning_accepts_baseline_but_rejects_foreign_stage_policy(self):
        model=torch.nn.Module();model.body=torch.nn.Linear(2,2);model.head=torch.nn.Module()
        model.head.relation_refiner=self.module()
        initial=copy.deepcopy(model.head.relation_refiner.state_dict())
        base={key:value for key,value in model.state_dict().items() if not key.startswith("head.relation_refiner.")}
        with patch("propvg.utils.checkpoint.torch.load",return_value=dict(state_dict=base)),patch("propvg.utils.checkpoint.is_main",return_value=False):
            load_pretrained_checkpoint(model,finetune_from="mock-baseline.pth")
        for key,value in initial.items():
            torch.testing.assert_close(model.head.relation_refiner.state_dict()[key],value,rtol=0,atol=0)
        foreign=torch.nn.Module();foreign.head=torch.nn.Module();foreign.head.relation_refiner=self.module(prior_uniform_mix=.5)
        with patch("propvg.utils.checkpoint.torch.load",return_value=dict(state_dict=foreign.state_dict())), \
             self.assertRaisesRegex(ValueError,"policy differs"):
            load_pretrained_checkpoint(model,finetune_from="mock-foreign.pth")

    def test_invalid_hyperparameters_are_rejected(self):
        for options in (dict(target_topk=0),dict(reference_topk=True),dict(num_heads=3),dict(loss_weight=0),
                        dict(prior_uniform_mix=2.),dict(discriminability_floor=0),dict(initial_strength=3.),
                        dict(relation_temperature=float("nan")),dict(aggregation="unknown"),dict(detach_inputs=1)):
            with self.subTest(options=options),self.assertRaises(ValueError):
                self.module(**options)

    def test_offline_audit_matches_known_repair_harm_counts(self):
        m,args=self.module(),self.inputs()
        with m.diagnostics():
            detail=m(*args)["diagnostics"][0]
        first=sample_record(detail,torch.tensor([.1,.45,.2,.55]),0,"x.jpg","target")
        second=copy.deepcopy(first)
        second.update(dataset_index=1,crs_top1_correct=False,final_top1_correct=True,corrected=True,harmed=False)
        third=copy.deepcopy(first)
        third.update(dataset_index=2,crs_top1_correct=True,final_top1_correct=False,corrected=False,harmed=True)
        summary=summarize([second,third])
        self.assertEqual((summary["corrected"],summary["harmed"],summary["net"]),(1,1,0))

    def test_audit_agrees_with_formal_postprocessor_and_box_metrics(self):
        from propvg.models.det_seg.mix_ref_unimodel_OMG import MIXRefUniModel_OMG
        from propvg.apis.test import aerialvg_box_metrics
        m,args=self.module(),self.inputs()
        with m.diagnostics():
            out=m(*args)
        row=sample_record(out["diagnostics"][0],torch.tensor([.1,.45,.2,.55]),0)
        post=MIXRefUniModel_OMG.__new__(MIXRefUniModel_OMG)
        torch.nn.Module.__init__(post)
        post.with_nms=False
        for mode in ("crs","final"):
            raw=dict(pred_class_allobj=args[2].detach(),pred_bbox_allobj=args[1].detach(),
                     refer_label=args[5].detach() if mode == "crs" else out["scores"].detach())
            if mode == "final":
                raw["ranking_priority"]=out["ranking_priority"]
            prediction=post.get_predictions_parts(raw,args[-1],with_bbox=True)["pred_bboxes"]
            expected=aerialvg_box_metrics(prediction,[torch.tensor([[20.,45.,40.,55.]])])
            actual=summarize([row])[mode]
            self.assertAlmostEqual(actual["top1_acc"],expected[0],places=5)
            self.assertAlmostEqual(actual["top5_acc"],expected[1],places=5)
            self.assertAlmostEqual(actual["miou"],expected[2],places=3)

    def test_original_baseline_matching_and_report_export(self):
        m,args=self.module(),self.inputs()
        with m.diagnostics():
            detail=m(*args)["diagnostics"][0]
        row=sample_record(detail,torch.tensor([.1,.45,.2,.55]),0,"/images/x.jpg","target")
        summary=summarize([row]);summary["split"]="test"
        with tempfile.TemporaryDirectory(prefix="propvg_relation_audit_") as name:
            path=Path(name)
            baseline=path/"baseline.csv"
            with baseline.open("w",newline="") as handle:
                writer=csv.DictWriter(handle,fieldnames=["dataset_index","filename","expression","top1_iou"])
                writer.writeheader();writer.writerow(dict(dataset_index=0,filename="x.jpg",expression="target",top1_iou=.9))
            previous=original_baseline_comparison([row],baseline)
            self.assertEqual(previous["original_top1"],100.)
            summary["original_baseline"]=previous
            write_reports(path,[row],summary)
            self.assertTrue((path/"summary.json").is_file())
            self.assertTrue((path/"report.md").is_file())
            with self.assertRaisesRegex(ValueError,"does not match"):
                original_baseline_comparison([dict(row,expression="another caption")],baseline)

    def test_invalid_boxes_loss_and_backward_remain_finite(self):
        m,args=self.module(),list(self.inputs())
        args[1]=torch.zeros_like(args[1])
        out=m(*args)
        loss,metrics=m.loss(out,[torch.tensor([[20.,45.,40.,55.]])],args[-1])
        self.assertEqual(float(loss),0.)
        self.assertEqual(float(metrics["reachable"]),0.)
        loss.backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None))


class RelationHeadIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.options=copy.deepcopy(Config.fromfile("configs/aerialvg/PropVG-aerialvg.py").model.head)
        cls.options["enable_birs"]=False
        cls.options.update(num_queries=8,enable_grab=False,enable_bcem=False,enable_mask=False,enable_crv=False,
                           enable_correctness=False,enable_relation_refinement=False,
                           relation_refinement=dict(target_topk=4,reference_topk=6,model_dim=16,num_heads=4,
                                                    detach_inputs=False))

    def inputs(self):
        from test_optional_pipeline import OptionalPipelineTests
        return OptionalPipelineTests().inputs()

    def test_disabled_option_is_exactly_legacy_initialization_and_rng(self):
        options=dict(self.options);options.pop("enable_relation_refinement");options.pop("relation_refinement")
        torch.manual_seed(123);old=build_head(options);state=torch.get_rng_state().clone()
        torch.manual_seed(123);new=build_head(self.options)
        torch.testing.assert_close(torch.get_rng_state(),state,rtol=0,atol=0)
        self.assertIsNone(new.relation_refiner)
        self.assertEqual(list(old.state_dict()),list(new.state_dict()))
        new.load_state_dict(old.state_dict(),strict=True)
        x,target,args=self.inputs()
        torch.manual_seed(321);expected=old.forward_train(x,target,**args)
        torch.manual_seed(321);actual=new.forward_train(x,target,**args)
        for key in expected[0]:
            torch.testing.assert_close(actual[0][key],expected[0][key],rtol=0,atol=0)
        torch.testing.assert_close(actual[1]["refer_label"],expected[1]["refer_label"],rtol=0,atol=0)

    def test_enabling_preserves_existing_parameters_and_rng_and_joint_backward(self):
        torch.manual_seed(123);base=build_head(self.options);state=torch.get_rng_state().clone()
        torch.manual_seed(123);model=build_head(dict(self.options,enable_relation_refinement=True))
        torch.testing.assert_close(torch.get_rng_state(),state,rtol=0,atol=0)
        for name,value in base.state_dict().items():
            torch.testing.assert_close(model.state_dict()[name],value,rtol=0,atol=0)
        x,target,args=self.inputs()
        losses,pred,info=model.forward_train(x,target,**args)
        self.assertIn("loss_relation_final",losses)
        self.assertIn("ranking_priority",pred)
        sum(v for k,v in losses.items() if k.startswith("loss_")).backward()
        self.assertGreater(sum(float(p.grad.abs().sum()) for p in model.mrsm.parameters() if p.grad is not None),0.)

    def test_final_loss_alone_reaches_crs_and_shared_encodings_not_box_heads(self):
        torch.manual_seed(123)
        model=build_head(dict(self.options,enable_relation_refinement=True))
        x,target,args=self.inputs()
        torch.manual_seed(789)
        _,pred,_=model.forward_train(x,target,**args)
        # Use a predicted shortlist box as a controlled refer GT, without
        # changing the pool or inserting GT into candidate selection.
        order=pred["refer_label"][0,:,0].cpu().argsort(descending=True)
        box=pred["pred_bbox_allobj"][0,order[0]]
        xyxy=torch.cat((box[:2]-box[2:]/2,box[:2]+box[2:]/2)).clamp(0,1)
        target["bbox"]=[xyxy*torch.tensor([128.,96.,128.,96.])]
        torch.manual_seed(789)
        losses,_,_=model.forward_train(x,target,**args)
        self.assertEqual(float(losses["metric_relation_refinement_supervised_samples"]),1.)
        losses["loss_relation_final"].backward()
        for branch in (model.mrsm,model.img_embedding,model.lan_embedding,model.decoder_transformer):
            self.assertGreater(sum(float(p.grad.abs().sum()) for p in branch.parameters() if p.grad is not None),0.)
        self.assertTrue(all(p.grad is None for p in model.bbox_embed.parameters()))
        self.assertTrue(all(p.grad is None for p in model.class_embed.parameters()))

    def test_detached_stage_preserves_all_base_gradients_clipping_and_adam_update(self):
        from propvg.apis.train import clip_training_gradients
        for bcem in (False,True):
            options=dict(self.options,enable_bcem=bcem,
                         bcem=dict(context_dim=16,num_heads=4,sampling_mode='ray',num_rays=4,radial_samples=4))
            torch.manual_seed(123);base=build_head(options)
            torch.manual_seed(123);model=build_head(dict(options,enable_relation_refinement=True,
                relation_refinement=dict(options['relation_refinement'],detach_inputs=True)))
            x,target,args=self.inputs()
            torch.manual_seed(789)
            _,pred,_=base.forward_train(x,target,**args)
            slot=int(pred['refer_label'][0,:,0].cpu().argsort(descending=True)[0])
            box=pred['pred_bbox_allobj'][0,slot]
            target['bbox']=[torch.cat((box[:2]-box[2:]/2,box[:2]+box[2:]/2)).clamp(0,1)*torch.tensor([128.,96.,128.,96.])]
            torch.manual_seed(789);old_loss,_,_=base.forward_train(x,target,**args)
            torch.manual_seed(789);new_loss,_,_=model.forward_train(x,target,**args)
            self.assertEqual(float(new_loss['metric_relation_refinement_supervised_samples']),1.)
            self.assertGreater(float(new_loss['loss_relation_final']),0.)
            sum(v for k,v in old_loss.items() if k.startswith('loss_')).backward()
            sum(v for k,v in new_loss.items() if k.startswith('loss_')).backward()
            self.assertGreater(sum(float(p.grad.abs().sum()) for p in model.relation_refiner.parameters()
                                   if p.grad is not None),0.)
            def compare():
                other=dict(model.named_parameters())
                for key,parameter in base.named_parameters():
                    if parameter.grad is None:
                        self.assertIsNone(other[key].grad)
                    else:
                        torch.testing.assert_close(other[key].grad,parameter.grad,rtol=0,atol=0)
            compare()
            clip_training_gradients(base,.15)
            clip_training_gradients(model,.15)
            compare()
            a=torch.optim.Adam(base.parameters(),lr=1e-4,amsgrad=True)
            b=torch.optim.Adam(model.parameters(),lr=1e-4,amsgrad=True)
            a.step();b.step()
            current=dict(model.named_parameters())
            for key,parameter in base.named_parameters():
                torch.testing.assert_close(current[key],parameter,rtol=0,atol=0)

    def test_train_loop_uses_final_loss_logs_metrics_and_marks_optimizer_step(self):
        from propvg.apis.train import train_model
        class Wrapper(torch.nn.Module):
            def __init__(inner):
                super().__init__()
                inner.head=torch.nn.Module()
                inner.head.relation_refiner=TwoStageRelationRefinement(hidden_channels=16,model_dim=16,num_heads=4)
            def forward(inner,**kwargs):
                value=inner.head.relation_refiner.strength_logit
                return dict(loss_det=value*0,loss_refer=value*0,loss_relation_final=(value-1).square(),
                            metric_relation_refinement_supervised_samples=value.detach().new_tensor(1.)),None
        model=Wrapper()
        before=model.head.relation_refiner.strength_logit.detach().clone()
        optimizer=torch.optim.SGD(model.parameters(),lr=.01)
        cfg=SimpleNamespace(distributed=False,use_fp16=False,grad_norm_clip=.15,log_interval=1,
                            data=SimpleNamespace(samples_per_gpu=1),world_size=1)
        with patch("propvg.apis.train.extract_data",side_effect=lambda x:x),patch("propvg.apis.train.log_metrics") as log:
            train_model(0,cfg,model,optimizer,[{}])
        self.assertEqual(int(model.head.relation_refiner.trained_steps),1)
        self.assertFalse(torch.equal(model.head.relation_refiner.strength_logit,before))
        self.assertIn("train/loss_relation_final",log.call_args[0][0])

    def test_untrained_eval_is_exact_crs_fallback_and_inference_never_reads_gt(self):
        torch.manual_seed(123);base=build_head(self.options).eval()
        torch.manual_seed(123);model=build_head(dict(self.options,enable_relation_refinement=True)).eval()
        x,target,args=self.inputs()
        with torch.no_grad():
            expected,_=base.forward_test(x,targets=target,with_mask=False,**args)
            actual,_=model.forward_test(x,targets=target,with_mask=False,**args)
        self.assertNotIn("ranking_priority",actual)
        torch.testing.assert_close(expected["refer_label"],actual["refer_label"],rtol=0,atol=0)
        model.relation_refiner.mark_optimizer_step()
        target["bbox"]="MUST NOT READ"
        target["img_metas"][0]["refer_target_index"]=[999]
        with torch.no_grad(),model.relation_refiner.diagnostics():
            actual,info=model.forward_test(x,targets=target,with_mask=False,**args)
        self.assertIn("ranking_priority",actual)
        self.assertIn("relation_refinement_samples",info)

    def test_grab_keeps_unfiltered_reference_source_and_bcem_coexists(self):
        for grab,bcem in ((True,False),(False,True),(True,True)):
            options=dict(self.options,enable_relation_refinement=True,enable_grab=grab,enable_bcem=bcem,
                         bcem=dict(context_dim=16,num_heads=4,sampling_mode="ray",num_rays=4,radial_samples=4))
            model=build_head(options)
            x,target,args=self.inputs()
            with model.relation_refiner.diagnostics():
                loss,pred,info=model.forward_train(x,target,**args)
            self.assertEqual(len(info["relation_refinement_samples"][0]["reference_boxes"]),8)
            self.assertTrue(torch.isfinite(loss["loss_relation_final"]))

    def test_crv_conflict_is_explicit(self):
        with self.assertRaisesRegex(ValueError,"not two sequential rerankers"):
            build_head(dict(self.options,enable_relation_refinement=True,enable_crv=True))


if __name__ == "__main__":
    unittest.main()
