"""CPU checks for detailed GrabVG binding and its Grab-only filtering loss."""
import copy
import math
import os
import unittest
from unittest import mock

os.environ.setdefault("MPLCONFIGDIR", "/tmp/propvg_matplotlib")
import torch
from torch import nn
from torch.nn import functional as F
from mmcv import Config

from propvg.models import build_head
from propvg.models.heads.grab_binding import GrabGraphBinding, GraphBlock


class GrabPaperTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def model(self, **kwargs):
        torch.manual_seed(1049)
        options = dict(hidden_channels=32, num_heads=8, layers=2, k=2)
        options.update(kwargs)
        return GrabGraphBinding(**options)

    def inputs(self, batch=2, count=8):
        torch.manual_seed(1051)
        queries = torch.randn(batch, count, 32, requires_grad=True)
        boxes = torch.rand(batch, count, 4) * .6 + .2
        boxes[..., 2:] *= .3
        text = torch.randn(batch, 7, 32, requires_grad=True)
        padding = torch.tensor([[False] * 4 + [True] * 3]).expand(batch, -1)
        visuals = [torch.randn(batch, 32, size, size, requires_grad=True) for size in (16, 8, 4, 2)]
        masks = [torch.zeros(batch, size, size, dtype=torch.bool) for size in (16, 8, 4, 2)]
        for mask in masks:
            mask[:, :, mask.shape[-1] // 2:] = True
        return queries, boxes, text, padding, visuals, masks

    def test_three_square_rings_and_structure(self):
        model = self.model()
        points = model.base_points
        self.assertEqual(tuple(points.shape), (48, 2))
        self.assertEqual(len(points.unique(dim=0)), 48)
        for ring, radius in zip(points.split(16), (.25, .5, .6)):
            torch.testing.assert_close(ring.abs().amax(-1), torch.full((16,), radius))
            torch.testing.assert_close(ring.mean(0), torch.zeros(2), atol=1e-7, rtol=0)
        self.assertEqual(model.geo_mlp[0].in_features, 8)
        self.assertEqual(model.geo_mlp[-1].out_features, 64)
        self.assertEqual(model.text_edge[0].out_features, 64)
        self.assertEqual(model.edge_bias[0].out_features, 8)
        self.assertEqual(model.level_fusion.in_features, 4 * 32)
        self.assertIsInstance(model.blocks[0].out_proj, nn.Linear)
        self.assertTrue(hasattr(model, "anchor_text_ffn"))
        self.assertTrue(hasattr(model, "appearance_ffn"))
        with self.assertRaisesRegex(ValueError, "multiple of 12"):
            self.model(anchors=47)

    def test_source_to_receiver_geometry_and_self_loop(self):
        boxes = torch.tensor([[[.25, .25, .2, .4], [.75, .50, .4, .2]]])
        geometry, distance = GrabGraphBinding._geometry(boxes)
        length = math.sqrt(.5 ** 2 + .25 ** 2)
        expected = torch.tensor([.5, .25, length, math.log(length + 1e-6),
                                 .5 / (length + 1e-6), .25 / (length + 1e-6),
                                 math.log(2), math.log(.5)])
        torch.testing.assert_close(geometry[0, 0, 1], expected)
        self.assertEqual(distance[0, 0, 0].item(), 0)
        self.assertTrue(torch.isfinite(geometry).all())
        torch.testing.assert_close(geometry[0, 0, 0, 3], torch.tensor(math.log(1e-6)))

    def test_symmetric_knn_and_single_node(self):
        model = self.model(k=1)
        centers = torch.tensor([[0., 0.], [.1, 0.], [.3, 0.], [1., 0.]])
        distance = torch.cdist(centers, centers)[None]
        adjacency = model._adjacency(distance)
        expected = torch.tensor([[[1, 1, 0, 0], [1, 1, 1, 0],
                                  [0, 1, 1, 1], [0, 0, 1, 1]]], dtype=torch.bool)
        torch.testing.assert_close(adjacency, expected)
        torch.testing.assert_close(model._adjacency(torch.zeros(1, 1, 1)), torch.ones(1, 1, 1, dtype=torch.bool))

    def test_headwise_attention_matches_explicit_reference(self):
        model = self.model(layers=1, k=1).eval()
        torch.manual_seed(1061)
        appearance = torch.randn(1, 4, 32)
        boxes = torch.tensor([[[.1, .2, .1, .1], [.2, .2, .2, .1],
                               [.4, .2, .1, .2], [.8, .2, .1, .1]]])
        sentence = torch.randn(1, 32)
        captured = []
        handle = model.blocks[0].register_forward_pre_hook(lambda _, args: captured.append(args[1].detach()))
        with torch.no_grad():
            output = model._topology(appearance, boxes, sentence)
            geometry, distance = model._geometry(boxes)
            adjacency = model._adjacency(distance)[0]
            edge = model.geo_mlp(geometry) * model.text_edge(sentence)[:, None, None]
            q, key, value = [projection(appearance)[0].reshape(4, 8, 4)
                             for projection in (model.q_proj[0], model.k_proj[0], model.v_proj[0])]
            edge_key = model.edge_k[0](edge)[0].reshape(4, 4, 8, 4)
            edge_value = model.edge_v[0](edge)[0].reshape(4, 4, 8, 4)
            bias = model.edge_bias[0](edge)[0]
            message = torch.zeros(4, 8, 4)
            for receiver in range(4):
                neighbors = adjacency[receiver].nonzero(as_tuple=True)[0]
                for head in range(8):
                    logits = ((key[neighbors, head] + edge_key[receiver, neighbors, head])
                              * q[receiver, head]).sum(-1) / 2 + bias[receiver, neighbors, head]
                    weights = logits.softmax(0)
                    message[receiver, head] = (weights[:, None] *
                                              (value[neighbors, head] + edge_value[receiver, neighbors, head])).sum(0)
            expected_message = message.reshape(1, 4, 32)
            torch.testing.assert_close(captured[0], expected_message)
            z = model.blocks[0](appearance, expected_message)
            gate = model.final_gate(torch.cat((appearance, z), dim=-1)).sigmoid()
            torch.testing.assert_close(output, appearance + gate * (z - appearance))
        handle.remove()

    def test_filter_keeps_original_queries_and_boxes(self):
        model = self.model()
        queries, boxes, text, padding, _, _ = self.inputs()
        chosen, chosen_boxes, logits, indices = model._filter(queries, boxes, text, padding)
        self.assertEqual(tuple(logits.shape), (2, 8))
        self.assertEqual(tuple(chosen.shape), (2, 4, 32))
        torch.testing.assert_close(chosen, queries.gather(1, indices[..., None].expand(-1, -1, 32)))
        torch.testing.assert_close(chosen_boxes, boxes.gather(1, indices[..., None].expand(-1, -1, 4)))

    def test_padding_coordinates_and_per_level_sampling(self):
        model = self.model().eval()
        for parameter in model.offset_head.parameters():
            nn.init.zeros_(parameter)
        queries, boxes, text, padding, visuals, masks = self.inputs(batch=1, count=2)
        grids = []
        original_sample = F.grid_sample

        def capture(visual, grid, **kwargs):
            grids.append(grid.detach())
            return original_sample(visual, grid, **kwargs)

        with mock.patch("propvg.models.heads.grab_binding.F.grid_sample", side_effect=capture):
            model._appearance(queries, boxes, text, padding, visuals, masks)
        points = (boxes[:, :, None, :2] + model.base_points[None, None] * boxes[:, :, None, 2:]).clamp(0, 1)
        expected = (points * points.new_tensor([.5, 1.]) * 2 - 1).reshape(1, 96, 1, 2)
        self.assertEqual(len(grids), 4)
        for grid in grids:
            torch.testing.assert_close(grid, expected)

    def test_full_forward_backward_and_text_padding_invariance(self):
        model = self.model().eval()
        args = self.inputs()
        graph, boxes, logits, selected = model(*args)
        changed_text = args[2].detach().clone()
        changed_text[args[3]] = 1e4
        actual = model(args[0], args[1], changed_text, *args[3:])
        for expected, found in zip((graph, boxes, logits, selected), actual):
            torch.testing.assert_close(expected, found)
        (graph.square().mean() + logits.square().mean()).backward()
        for name, parameter in model.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)
        for feature in args[4]:
            self.assertIsNotNone(feature.grad)
            self.assertTrue(torch.isfinite(feature.grad).all())
        self.assertEqual(tuple(graph.shape), (2, 4, 32))
        torch.testing.assert_close(boxes, args[1].gather(1, selected[..., None].expand(-1, -1, 4)))

    def test_legacy_layout_is_only_an_initialization_descriptor(self):
        legacy = GrabGraphBinding(legacy_layout=True)
        self.assertEqual(legacy.geo_mlp[0].in_features, 7)
        self.assertEqual(legacy.geo_mlp[-1].out_features, 256)
        self.assertFalse(hasattr(legacy, "anchor_text_ffn"))
        self.assertIsInstance(legacy.blocks[0].out_proj, nn.Identity)
        with self.assertRaisesRegex(RuntimeError, "initialization-only"):
            legacy(*self.inputs())

    def test_graph_block_projects_message_before_residual(self):
        block = GraphBlock(8)
        x, message = torch.randn(1, 3, 8), torch.randn(1, 3, 8)
        expected = block.norm1(x + F.linear(message, block.out_proj.weight, block.out_proj.bias))
        expected = block.norm2(expected + block.ffn(expected))
        torch.testing.assert_close(block(x, message), expected)


class GrabFilterLossTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        options = copy.deepcopy(Config.fromfile("configs/aerialvg/PropVG-aerialvg.py").model.head)
        options["enable_birs"] = False
        options.update(enable_grab=True, enable_mask=False, enable_bcem=False, enable_correctness=False,
                       enable_relation_refinement=False,
                       enable_crv=False, num_queries=4)
        torch.manual_seed(1081)
        cls.head = build_head(options).eval()

    def test_paper_weights_missing_only_average_and_gradients(self):
        missed = [-2., 3., 2., 1.]
        kept = [3., 2., 1., -2.]
        cases = [
            ("one_missed", [missed, kept], [True, True], 1, 0),
            ("same_miss_more_kept", [missed, kept, kept, kept], [True] * 4, 1, 0),
            ("two_missed_and_no_refer", [missed, [-1., 3., 2., 1.], kept, missed],
             [True, True, True, False], 1, 0),
            ("none_missed", [kept, kept], [True, True], 1, 0),
            ("no_refer", [missed, kept], [False, False], 1, 0),
            ("ddp_only_this_rank_missed", [missed, kept], [True, True], 2, 0),
            ("ddp_only_remote_rank_missed", [kept, kept], [True, True], 2, 1),
            ("ddp_unequal_missed_counts", [missed, kept, kept, kept], [True] * 4, 2, 3),
            ("ddp_no_rank_missed", [kept, kept], [True, True], 2, 0),
        ]
        penalties = {}
        for name, values, exists, world_size, remote_misses in cases:
            with self.subTest(case=name):
                logits = torch.tensor(values, requires_grad=True)
                batch = len(values)
                selection = logits.detach().topk(2, dim=1).indices

                def fixed_filter(queries, boxes, text, text_mask):
                    return (queries.gather(1, selection[..., None].expand(-1, -1, 256)),
                            boxes.gather(1, selection[..., None].expand(-1, -1, 4)), logits, selection)

                indices = [(torch.tensor([0, 1]), torch.tensor([0, 1])) for _ in values]
                target = dict(mask=None,
                              bbox=[torch.tensor([[20., 10., 40., 30.], [5., 3., 10., 8.]]) for _ in values],
                              img_metas=[dict(img_shape=(48, 64), pad_shape=(64, 64),
                                              refer_target_index=[0] if flag else []) for flag in exists])
                kwargs = dict(cls_feat=torch.zeros(batch, 768), lan_feat=torch.randn(batch, 6, 768),
                              lan_mask=torch.tensor([[False] * 4 + [True] * 2]).expand(batch, -1),
                              img=torch.zeros(batch, 3, 64, 64))
                with mock.patch.object(self.head.grab_binding, "_filter", side_effect=fixed_filter), \
                        mock.patch.object(self.head.detrloss, "forward",
                                          return_value=({"dummy": logits.new_zeros(())}, [indices] * 3)), \
                        mock.patch("torch.distributed.is_available", return_value=True), \
                        mock.patch("torch.distributed.is_initialized", return_value=world_size > 1), \
                        mock.patch("torch.distributed.get_world_size", return_value=world_size), \
                        mock.patch("torch.distributed.all_reduce", side_effect=lambda count: count.add_(remote_misses)) as reduce:
                    losses, _, _ = self.head.forward_train(torch.randn(batch, 768, 4, 4), target, **kwargs)
                self.assertEqual(reduce.call_count, int(world_size > 1))
                positive = torch.tensor([[1., 1., 0., 0.]]).expand(batch, -1)
                bce = F.binary_cross_entropy_with_logits(logits, positive)
                missing = torch.tensor(exists) & ~(selection == 0).any(1)
                # Independent reference: select only pruned referents before
                # averaging. Non-missing/no-refer samples must not dilute it.
                global_count = max(int(missing.sum()) + remote_misses, 1)
                keep = F.softplus(-logits[missing, 0]).sum() * world_size / global_count
                expected = .3 * bce + keep
                torch.testing.assert_close(losses["loss_filter"], expected)
                expected_gradient = .3 * (logits.detach().sigmoid() - positive) / logits.numel()
                if missing.any():
                    expected_gradient[missing, 0] += (logits.detach()[missing, 0].sigmoid() - 1) * world_size / global_count
                losses["loss_filter"].backward()
                torch.testing.assert_close(logits.grad, expected_gradient)
                penalties[name] = (losses["loss_filter"] - .3 * bce).detach()
                self.assertTrue(torch.isfinite(losses["loss_filter"]))
        torch.testing.assert_close(penalties["one_missed"], penalties["same_miss_more_kept"])
        torch.testing.assert_close(penalties["none_missed"], torch.tensor(0.))
        torch.testing.assert_close(penalties["no_refer"], torch.tensor(0.))
        torch.testing.assert_close(penalties["ddp_only_this_rank_missed"], 2 * penalties["one_missed"])
        torch.testing.assert_close(penalties["ddp_unequal_missed_counts"], .5 * penalties["one_missed"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
