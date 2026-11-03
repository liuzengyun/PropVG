"""CPU checks for checkpoint retention, best selection and resumable snapshots."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from propvg.utils.checkpoint import save_checkpoint


class CheckpointRetentionTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="propvg_checkpoint_retention_")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.model = torch.nn.Linear(2, 1)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=.01)
        self.model(torch.ones(1, 2)).sum().backward()
        self.optimizer.step()
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=3)

    def save(self, epoch, metric=1., best=0., interval=1, keep=10, seg=False):
        info = dict(epoch=epoch-1, top1_acc=metric, best_top1_acc=best)
        if seg:
            info.update(mIoU=metric, best_mIoU=best)
        save_checkpoint(str(self.root), interval, self.model, self.optimizer,
                        self.scheduler, info,
                        save_key_metric={"Det": "top1_acc", "Seg": "mIoU" if seg else None},
                        max_keep_ckpts=keep)

    def snapshots(self):
        return {p.name for p in self.root.glob("epoch_*.pth") if p.is_file()}

    def test_last_ten_numeric_epochs_best_and_full_training_state(self):
        # The best epoch falls outside the final rolling window.
        for epoch in range(1, 16):
            self.save(epoch, metric=100. if epoch == 2 else 1., best=100. if epoch > 2 else 0.)
        self.assertEqual(self.snapshots(), {f"epoch_{epoch}.pth" for epoch in range(6, 16)})
        best = torch.load(self.root / "top1_best.pth", map_location="cpu")
        latest = torch.load(self.root / "latest.pth", map_location="cpu")
        self.assertEqual(best["epoch"], 1)
        self.assertEqual(latest["epoch"], 14)
        self.assertEqual((self.root / "epoch_15.pth").read_bytes(), (self.root / "latest.pth").read_bytes())
        self.assertTrue(latest["optimizer"]["state"])
        self.assertEqual(latest["optimizer_param_names"], [["weight", "bias"]])
        self.assertTrue(latest["scheduler_state_is_next_epoch"])
        resumed_model = torch.nn.Linear(2, 1)
        resumed_model.load_state_dict(latest["state_dict"])
        resumed_optimizer = torch.optim.Adam(resumed_model.parameters())
        resumed_optimizer.load_state_dict(latest["optimizer"])
        resumed_scheduler = torch.optim.lr_scheduler.StepLR(resumed_optimizer, step_size=3)
        resumed_scheduler.load_state_dict(latest["scheduler"])
        self.assertEqual(resumed_optimizer.param_groups[0]["lr"], .01)

    def test_pruning_protects_named_files_directories_and_symlinks(self):
        for name in ("top1_best.pth", "det_best.pth", "segm_best.pth", "notes.pth", "epoch_0.pth.bak"):
            (self.root / name).write_bytes(b"preserve")
        (self.root / "epoch_0.pth").mkdir()
        (self.root / "epoch_1.pth").symlink_to(self.root / "notes.pth")
        for epoch in range(2, 6):
            self.save(epoch, metric=0., best=0., keep=2)
        for name in ("top1_best.pth", "det_best.pth", "segm_best.pth", "notes.pth", "epoch_0.pth.bak"):
            self.assertEqual((self.root / name).read_bytes(), b"preserve")
        self.assertTrue((self.root / "epoch_0.pth").is_dir())
        self.assertTrue((self.root / "epoch_1.pth").is_symlink())
        self.assertFalse((self.root / "epoch_2.pth").exists())
        self.assertFalse((self.root / "epoch_3.pth").exists())
        self.assertTrue((self.root / "epoch_4.pth").exists())
        self.assertTrue((self.root / "epoch_5.pth").exists())

    def test_interval_and_disabled_pruning(self):
        for epoch in range(1, 13):
            self.save(epoch, interval=3, keep=2)
        self.assertEqual(self.snapshots(), {"epoch_9.pth", "epoch_12.pth"})
        self.save(13, keep=0)
        self.assertEqual(self.snapshots(), {"epoch_9.pth", "epoch_12.pth", "epoch_13.pth"})
        self.save(14, interval=-1)
        self.assertNotIn("epoch_14.pth", self.snapshots())

    def test_segmentation_best_is_kept(self):
        self.save(1, metric=10., seg=True, keep=1)
        self.save(2, metric=5., best=10., seg=True, keep=1)
        self.assertEqual(self.snapshots(), {"epoch_2.pth"})
        self.assertEqual(torch.load(self.root / "segm_best.pth", map_location="cpu")["epoch"], 0)

    def test_failed_snapshot_does_not_prune_previous_epochs(self):
        self.save(1)
        self.save(2)
        with patch("propvg.utils.checkpoint.shutil.copyfile", side_effect=OSError("disk full")):
            with self.assertRaises(OSError):
                self.save(3, keep=1)
        self.assertIn("epoch_1.pth", self.snapshots())
        self.assertIn("epoch_2.pth", self.snapshots())

    def test_rejects_invalid_retention_before_writing(self):
        for value in (-1, True, 1.5, "10"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.save(1, keep=value)
        self.assertEqual(list(self.root.iterdir()), [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
