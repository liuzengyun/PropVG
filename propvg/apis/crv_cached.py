"""Opt-in offline CRV training helpers. Not imported by the legacy runners."""
import copy
import ast
from contextlib import contextmanager
import fcntl
import hashlib
import inspect
import json
import os
import textwrap
from pathlib import Path
from types import SimpleNamespace

import torch
from torch.utils.data import Dataset

from propvg.core.structure.boxes import Boxes
from propvg.apis.bcem_baseline_validation import validate_baseline_bcem
from propvg.layers.box_ops import box_cxcywh_to_xyxy
from propvg.models.heads.counterfactual_refinement import CounterfactualRefinement
from propvg.models.heads.top1_correctness import Top1CorrectnessHead
from propvg.utils.checkpoint import _validate_correctness_checkpoint


CACHE_VERSION = 1
FEATURE_KEYS = ("visual", "visual_mask", "boxes", "text", "text_mask",
                "probabilities", "logits", "candidate_features")
COMPRESSIBLE_KEYS = {"visual", "text", "candidate_features"}
# Audited v1 extractors from this implementation: identical frozen forward /
# tensor packing, different training/validation/cleanup helpers in the file.
# Pin their feature-code identity; NEVER map unknown old code to current code.
LEGACY_FEATURE_ENCODERS = {
    ("1c991c3f77e1d0a001ff51296c70848d6a0694cad6e90e808537aa7b0c6557bf",
     "0d93f848cf7c52356828f5af8b4120e0844c84ea626267278e3d44c53766b9ac"):
        "89827c3b8191cdbc62726a6e08c30fa8845c048656fbfc5f681e2e9bb5c5d13e",
    ("59d872f4bed06340f29ec95e112bb451b3ed51461e2b4143967512b95b185f15",
     "60713ed112c0ce7ed602cc1f0154cc2896ad54a71771c489988be680549a0fb0"):
        "89827c3b8191cdbc62726a6e08c30fa8845c048656fbfc5f681e2e9bb5c5d13e",
}


def atomic_torch_save(value, path):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(value, temporary)
    os.replace(str(temporary), str(path))


def atomic_json_save(value, path):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(str(temporary), str(path))


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@contextmanager
def feature_cache_lock(directory):
    """Keep another new-entry run from reading a cache while it is removed."""
    root = Path(directory).absolute()
    root.parent.mkdir(parents=True, exist_ok=True)
    # Outside the removable root; never unlink a locked inode (race with waiters).
    path = root.parent / ("." + root.name + ".crv.lock")
    with path.open("a+") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("Feature cache is in use by another run: {}".format(root))
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _identity_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def feature_encoding_sha256():
    """Fingerprint only the actual frozen-forward and sample-packing code."""
    functions = {}
    for name in ("extract_frozen_inputs", "make_record"):
        tree = ast.parse(textwrap.dedent(inspect.getsource(globals()[name])))
        body = tree.body[0].body
        if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, (ast.Str, ast.Constant)) \
                and isinstance(getattr(body[0].value, "value", getattr(body[0].value, "s", None)), str):
            body.pop(0)  # Documentation/comments do not affect numerical features.
        functions[name] = ast.dump(tree, include_attributes=False)
    return _identity_hash(dict(functions=functions, feature_keys=FEATURE_KEYS,
                               compressible=sorted(COMPRESSIBLE_KEYS), version=CACHE_VERSION))


def _semantic_cache_details(details):
    result = copy.deepcopy(details)
    old_extractor = result.pop("extractor_sha256", None)
    old_runner = result.pop("runner_sha256", None)  # Audit only for new semantic identities.
    if "feature_encoding_sha256" not in result:
        encoding = LEGACY_FEATURE_ENCODERS.get((old_extractor, old_runner))
        if encoding is None:
            raise ValueError("Unknown legacy cache extractor/runner pair; cannot safely reuse features from unaudited code")
        result["feature_encoding_sha256"] = encoding
    head = result["model"]["head"]
    if not head.get("enable_bcem", False):
        head.pop("bcem", None)  # Disabled BCEM is neither constructed nor used.
    return result


def validate_cache_identity(manifest, expected_fingerprint, expected_details=None):
    """Read-only compatibility: keep old per-sample fingerprints, never rewrite features."""
    stored = manifest.get("fingerprint")
    if stored == expected_fingerprint:
        return stored
    if expected_details is None or not isinstance(manifest.get("details"), dict):
        raise ValueError("CRV cache does not match current baseline/config/data; use a new cache_dir")
    previous = manifest["details"]
    old_semantic = _semantic_cache_details(previous)
    # Verify old metadata integrity, supporting either v1 whole-file hashes or
    # the new semantic identity. No blanket bypass of provenance checks.
    if stored not in (_identity_hash(previous), _identity_hash(old_semantic)):
        raise ValueError("CRV cache fingerprint/manifest integrity mismatch")
    current = _semantic_cache_details(expected_details)
    if _identity_hash(old_semantic) != _identity_hash(current):
        differences = [key for key in sorted(set(old_semantic) | set(current))
                       if _identity_hash(old_semantic.get(key)) != _identity_hash(current.get(key))]
        raise ValueError("CRV cache does not match current baseline/config/data or feature extraction; "
                         "changed fields: {}. Use a new cache_dir.".format(", ".join(differences)))
    return stored


def cache_identity(cfg, split, checkpoint_hash):
    """Never silently reuse features from different weights/data/preprocessing."""
    model = copy.deepcopy(dict(cfg.model))
    model["head"].pop("crv", None)
    model["head"]["enable_crv"] = False
    # This is ONLY the cache identity, not the instantiated model config.
    # Cached inputs precede the observer; toggling it does not change V/T/Q/z.
    model["head"]["enable_correctness"] = False
    model["head"].pop("correctness", None)
    model["vis_enc"]["pretrain"] = None
    repository = Path(__file__).resolve().parents[2]
    sources = {}
    # Conservative invalidation over the repo's model/data/geometry code.
    for folder in ("propvg/models", "propvg/datasets", "propvg/core/structure", "propvg/layers"):
        digest = hashlib.sha256()
        for path in sorted((repository / folder).rglob("*.py")):
            digest.update(str(path.relative_to(repository)).encode())
            digest.update(path.read_bytes())
        sources[folder] = digest.hexdigest()
    annotation = Path(cfg.data[split].annsfile).resolve()
    details = dict(version=CACHE_VERSION, split=split, model=model,
                   dataset=copy.deepcopy(dict(cfg.data[split])),
                   annotation_sha256=file_sha256(annotation), checkpoint_sha256=checkpoint_hash,
                   dtype=cfg.offline_crv.cache_dtype, sources=sources,
                   feature_encoding_sha256=feature_encoding_sha256(),
                   extractor_sha256=file_sha256(__file__),
                   runner_sha256=file_sha256(repository / "tools/train_crv_cached.py"),
                   torch_version=torch.__version__, seed=int(cfg.seed),
                   deterministic=bool(cfg.get("deterministic", True)),
                   strict_reproducibility=bool(cfg.get("strict_reproducibility", True)))
    return _identity_hash(_semantic_cache_details(details)), details


def load_baseline_state(path, keep_correctness=False, return_metadata=False):
    checkpoint = torch.load(path, map_location="cpu")
    if "state_dict" not in checkpoint:
        raise ValueError("baseline_checkpoint must be a complete PropVG checkpoint with state_dict")
    state = {key[7:] if key.startswith("module.") else key: value
             for key, value in checkpoint["state_dict"].items()}
    # A previously trained correctness head belongs to the frozen baseline
    # when enabled. Old CRV weights are never used to initialize this new CRV.
    state = {key: value for key, value in state.items()
             if not key.startswith("head.crv.") and (keep_correctness or not key.startswith("head.correctness."))}
    metadata = {key: checkpoint[key] for key in ("frozen_bcem_config",) if key in checkpoint}
    return (state, metadata) if return_metadata else state


def load_configured_baseline(cfg, return_validation=False):
    """Validate/preserve the optional pretrained observer even without a full model."""
    enabled = bool(cfg.model.head.get("enable_correctness", False))
    state, metadata = load_baseline_state(cfg.offline_crv.baseline_checkpoint, keep_correctness=enabled,
                                          return_metadata=True)
    bcem_validation = validate_baseline_bcem(cfg, state, metadata)
    if enabled:
        options = copy.deepcopy(dict(cfg.model.head.get("correctness") or {}))
        seed = int(options.pop("init_seed", 2717))
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed)
            observer = Top1CorrectnessHead(hidden_channels=cfg.model.head.hidden_channels, **options)
        holder = SimpleNamespace(head=SimpleNamespace(correctness=observer))
        try:
            _validate_correctness_checkpoint(holder, state)
            observer.load_state_dict({key[len("head.correctness."):]: value for key, value in state.items()
                                      if key.startswith("head.correctness.")}, strict=True)
        except (ValueError, RuntimeError) as error:
            raise ValueError("enable_correctness=True requires complete, trained, shape-compatible correctness "
                             "weights in baseline_checkpoint. Select that checkpoint and match the correctness "
                             "config, or disable this head. It is frozen, not trained, in the CRV-only entry.") from error
    return (state, bcem_validation) if return_validation else state


def load_frozen_baseline(model, state):
    """Strict for every active tensor; tolerate only known omitted old modules."""
    _validate_correctness_checkpoint(model, state)
    expected = model.state_dict()
    omitted = set()
    for prefix, module in model.named_modules():
        omitted.update((prefix + "." if prefix else "") + name
                       for name in getattr(module, "_omitted_parameter_names", []))
    missing = sorted(set(expected) - set(state))
    unexpected = sorted(set(state) - set(expected) - omitted)
    mismatch = [key for key in set(expected) & set(state) if expected[key].shape != state[key].shape]
    if missing or unexpected or mismatch:
        raise ValueError("Baseline/config mismatch: missing={}, unexpected={}, shapes={}. "
                         "Use the baseline's original architecture, Grab/BCEM switches and image size."
                         .format(missing[:8], unexpected[:8], mismatch[:8]))
    model.load_state_dict({key: state[key] for key in expected}, strict=True)
    model.requires_grad_(False)
    model.eval()
    return model


@torch.no_grad()
def extract_frozen_inputs(model, inputs):
    """One image/text forward. GT is intentionally not passed into the model."""
    if model.training or any(parameter.requires_grad for parameter in model.parameters()):
        raise ValueError("Cache extraction requires a fully frozen eval-mode grounding model")
    image = inputs["img"]
    visual, text, cls = model.extract_visual_language(image, inputs["ref_expr_inds"],
                                                     inputs["text_attention_mask"])
    batch, _, height, width = image.shape
    visual = visual.transpose(-1, -2).reshape(batch, -1, height // model.patch_size, width // model.patch_size)
    features, _ = model.head.forward_test(
        visual, cls_feat=cls, lan_feat=text, lan_mask=inputs["text_attention_mask"], img=image,
        targets={"img_metas": inputs["img_metas"]}, with_mask=False, return_crv_inputs=True)
    return features


def make_record(features, inputs, index, sample_index, fingerprint, dtype="float32"):
    if dtype not in ("float32", "float16"):
        raise ValueError("cache_dtype must be float32 or float16")
    record = {}
    for key in FEATURE_KEYS:
        tensor = features[key][index].detach().cpu()
        if tensor.is_floating_point():
            tensor = tensor.to(torch.float16 if dtype == "float16" and key in COMPRESSIBLE_KEYS else torch.float32)
        # Clone: torch.save(view) would otherwise serialize the WHOLE batch's
        # underlying storage for every individual sample.
        record[key] = tensor.contiguous().clone()
    meta = copy.deepcopy(inputs["img_metas"][index])
    record["img_meta"] = meta
    referred = meta["refer_target_index"]
    if len(referred) != 1:
        raise ValueError("Offline AerialVG CRV requires exactly one original refer GT per sample")
    original_gt = inputs["gt_bbox"][index][int(referred[0])].detach().cpu().reshape(4)
    record["refer_gt_pixel"] = original_gt.float().clone()
    height, width = meta["img_shape"][:2]
    # AerialVG's loader uses float64 boxes. Match REFHead._crv_refer_gt:
    # normalize in the original dtype BEFORE converting to visual float32.
    record["refer_gt"] = (original_gt / original_gt.new_tensor([width, height, width, height])).float().clone()
    record.update(sample_index=int(sample_index), fingerprint=fingerprint, cache_version=CACHE_VERSION)
    return record


def live_feature_batch(features, inputs):
    """Reuse no-grad GPU features directly, without a disk/CPU V/T/Q roundtrip."""
    result = {key: features[key].detach() for key in FEATURE_KEYS}
    result.update(img_metas=inputs["img_metas"], refer_gt=[], refer_gt_pixel=[])
    for boxes, meta in zip(inputs["gt_bbox"], inputs["img_metas"]):
        referred = meta["refer_target_index"]
        if len(referred) != 1:
            raise ValueError("CRV requires exactly one original refer GT per sample")
        box = boxes[int(referred[0])].detach().reshape(4)
        height, width = meta["img_shape"][:2]
        result["refer_gt_pixel"].append(box.float())
        result["refer_gt"].append((box / box.new_tensor([width, height, width, height])).float())
    return result


def cleanup_feature_caches(cfg, checkpoint_hash, splits, results_directory, logger):
    """Delete only manifest-owned sample files, after a successful whole run.

    Never recursively delete cache_dir: it can contain unrelated experiments,
    a different split, user files or symbolic links. Preflight EVERY split
    before deleting anything, and retain a JSON audit beside the checkpoints.
    """
    results = Path(results_directory).resolve()
    for filename in ("latest.pth", "top1_best.pth", "config.py"):
        path = results / filename
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError("Refusing cache cleanup before successful model/config saving: {}".format(path))
    configured = Path(cfg.offline_crv.cache_dir).absolute()
    if any(path.is_symlink() for path in (configured,) + tuple(configured.parents)):
        raise ValueError("Refusing cache cleanup through symlinks: {}".format(configured))
    root = configured.resolve()
    repository = Path(__file__).resolve().parents[2]
    protected = {Path("/"), Path("/tmp"), Path.home().resolve(), repository,
                 repository / "data", repository / "temp", repository / "work_dir"}
    protected.update(repository.parents)
    if root in protected or root == results or root in results.parents:
        raise ValueError("Refusing broad cache cleanup target: {}".format(root))
    plans = []
    for split in dict.fromkeys(splits):
        if split not in ("train", "val", "test"):
            raise ValueError("Invalid cache split for cleanup")
        directory = root / split
        if directory.is_symlink():
            raise ValueError("Refusing symlink cache split: {}".format(directory))
        manifest_path = directory / "manifest.json"
        if manifest_path.is_symlink():
            raise ValueError("Refusing symlink cache manifest")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected, expected_details = cache_identity(cfg, split, checkpoint_hash)
        if manifest.get("version") != CACHE_VERSION or not manifest.get("complete"):
            raise ValueError("Refusing cleanup of incomplete/foreign {} cache".format(split))
        try:
            validate_cache_identity(manifest, expected, expected_details)
        except ValueError as error:
            raise ValueError("Refusing cleanup of foreign {} cache: {}".format(split, error)) from error
        count = manifest.get("count")
        if not isinstance(count, int) or isinstance(count, bool) or count < 1:
            raise ValueError("Invalid cache sample count")
        owned = {"{:08d}.pt".format(index) for index in range(count)} | {"manifest.json"}
        entries = list(directory.iterdir())
        if {path.name for path in entries} != owned or any(path.is_symlink() or not path.is_file() for path in entries):
            raise ValueError("Refusing cache cleanup: unexpected/missing files or symlinks in {}".format(directory))
        plans.append((directory, entries, manifest))
    audit = dict(status="validated", root=str(root),
                 splits={path.name: dict(count=manifest["count"], fingerprint=manifest["fingerprint"])
                         for path, _, manifest in plans})
    atomic_json_save(audit, results / "cache_cleanup.json")
    for directory, entries, manifest in plans:
        logger.info("Removing generated %s cache: %s (%d samples); images/annotations/weights are untouched",
                    directory.name, directory, manifest["count"])
        for path in entries:
            if path.name != "manifest.json":
                path.unlink()
        # Keep the manifest until last: if cleanup is interrupted, a later
        # cache extraction can still identify and repair the partial cache.
        (directory / "manifest.json").unlink()
        directory.rmdir()
    if root.is_dir() and not any(root.iterdir()):
        root.rmdir()  # Empty run-specific root only; never recurse into parents.
    audit["status"] = "deleted"
    atomic_json_save(audit, results / "cache_cleanup.json")
    logger.info("Generated feature caches removed; regenerate them for a future cached run")
    return audit


class CachedCRVDataset(Dataset):
    def __init__(self, directory, expected_fingerprint=None, expected_details=None):
        self.directory = Path(directory)
        self.manifest = json.loads((self.directory / "manifest.json").read_text(encoding="utf-8"))
        if self.manifest.get("version") != CACHE_VERSION or not self.manifest.get("complete"):
            raise ValueError("Incomplete/unsupported CRV cache; run --stage cache first")
        self.legacy_compatible = False
        if expected_fingerprint is not None:
            validate_cache_identity(self.manifest, expected_fingerprint, expected_details)
            self.legacy_compatible = self.manifest["fingerprint"] != expected_fingerprint
        if self.manifest["count"] < 1:
            raise ValueError("Empty CRV cache")

    def __len__(self):
        return self.manifest["count"]

    def __getitem__(self, index):
        record = torch.load(self.directory / ("{:08d}.pt".format(index)), map_location="cpu")
        if record.get("fingerprint") != self.manifest["fingerprint"] or record.get("sample_index") != index:
            raise ValueError("Mismatched/corrupt CRV cache sample {}".format(index))
        return record


def collate_cached(records):
    return {**{key: torch.stack([record[key] for record in records]) for key in FEATURE_KEYS},
            "img_metas": [record["img_meta"] for record in records],
            "refer_gt": [record["refer_gt"] for record in records],
            "refer_gt_pixel": [record["refer_gt_pixel"] for record in records]}


def call_cached_crv(crv, batch, device, training=False, epoch=None):
    features = {key: batch[key].to(device=device, non_blocking=True) for key in FEATURE_KEYS}
    features = {key: value.float() if value.is_floating_point() else value for key, value in features.items()}
    # No GT is supplied during evaluation, even though a cache contains it.
    return crv(**{key: features[key] for key in FEATURE_KEYS}, img_metas=batch["img_metas"],
               refer_gt=[gt.to(device) for gt in batch["refer_gt"]] if training else None,
               epoch=epoch if training else None)


def cached_predictions(batch, probabilities, priority=None):
    """The same float conversion, clipping, filtering and ranking as PropVG."""
    predictions = []
    for index, (raw, meta) in enumerate(zip(batch["boxes"], batch["img_metas"])):
        height, width = meta["img_shape"][:2]
        boxes = Boxes(box_cxcywh_to_xyxy(raw.detach().cpu()))
        boxes.scale(scale_x=width, scale_y=height)
        boxes.clip((height, width))
        kept = boxes.nonempty()
        prediction = dict(boxes=boxes.tensor[kept], scores=probabilities[index].detach().cpu().reshape(-1)[kept])
        if priority is not None:
            prediction["ranking_logits"] = priority[index].detach().cpu().reshape(-1)[kept]
        predictions.append(prediction)
    return predictions


def assemble_checkpoint(baseline, crv, epoch, optimizer=None, metadata=None):
    state = dict(baseline)
    state.update({"head.crv." + key: value.detach().cpu().clone() for key, value in crv.state_dict().items()})
    checkpoint = dict(state_dict=state, epoch=int(epoch), offline_crv=True, **(metadata or {}))
    if optimizer is not None:
        checkpoint["crv_optimizer"] = optimizer.state_dict()
    return checkpoint


def build_crv(cfg):
    options = copy.deepcopy(dict(cfg.model.head.crv))
    seed = int(options.pop("init_seed", 1717))
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        return CounterfactualRefinement(hidden_channels=cfg.model.head.hidden_channels, **options)
