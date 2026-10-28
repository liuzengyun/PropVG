"""Strict BCEM configuration validation for the opt-in frozen CRV runner only."""
import copy
import inspect
import json
from pathlib import Path
import re

from mmcv import Config
import torch

from propvg.models.heads.background_context import BackgroundContextEnhancement


def _plain(value):
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def bcem_config_contract(head):
    """Include ALL constructor arguments, not just shape-changing arguments."""
    enabled = head.get("enable_bcem", False)
    if not isinstance(enabled, bool):
        raise ValueError("model.head.enable_bcem must be True or False")
    if not enabled:
        return dict(version=1, enabled=False)
    signature = inspect.signature(BackgroundContextEnhancement.__init__)
    arguments = {name: copy.deepcopy(parameter.default) for name, parameter in signature.parameters.items()
                 if name != "self"}
    options = copy.deepcopy(dict(head.get("bcem") or {}))
    unknown = set(options) - set(arguments)
    if unknown or "hidden_channels" in options:
        raise ValueError("Invalid model.head.bcem arguments: {}".format(sorted(unknown | ({"hidden_channels"} & set(options)))))
    arguments.update(options)
    arguments["hidden_channels"] = head.get("hidden_channels", 256)
    # Resolve None exactly as the BCEM constructor does. Explicitly writing
    # defaults is equivalent; lists/tuples of radial edges are equivalent too.
    if arguments["radial_samples"] is None:
        arguments["radial_samples"] = (4 if arguments["sampling_mode"] == "ray"
                                       and arguments["ray_lateral_offset"] == 0 else 2)
    if arguments["min_valid_points"] is None:
        arguments["min_valid_points"] = 2 if arguments["sampling_mode"] == "ray" else 5
    result = dict(version=1, enabled=True, arguments=_plain(arguments))
    json.dumps(result, allow_nan=False)
    return result


def _flatten(value, prefix=""):
    result = {}
    for key, item in value.items():
        name = prefix + "." + key if prefix else key
        if isinstance(item, dict):
            result.update(_flatten(item, name))
        else:
            result[name] = item
    return result


def _assert_equal(expected, actual, source):
    left, right = _flatten(expected), _flatten(actual)
    differences = []
    for key in sorted(set(left) | set(right)):
        old, new = left.get(key, "<missing>"), right.get(key, "<missing>")
        if json.dumps(old, sort_keys=True) != json.dumps(new, sort_keys=True):
            differences.append("{}: baseline={!r}, current={!r}".format(key, old, new))
    if differences:
        raise ValueError("BCEM config mismatch against {}:\n{}\nCopy the baseline's complete model.head.bcem "
                         "settings and enable_bcem switch; do not change a frozen baseline's behavior."
                         .format(source, "\n".join(differences)))


def _contract_from_file(path):
    if not path.is_file():
        raise ValueError("Original baseline_config not found: {}".format(path))
    source = Config.fromfile(str(path))
    if "model" not in source or "head" not in source.model:
        raise ValueError("baseline_config has no model.head: {}".format(path))
    return bcem_config_contract(source.model.head)


def validate_baseline_bcem(cfg, state, metadata):
    """Require checkpoint metadata / original config whenever BCEM is active."""
    current = bcem_config_contract(cfg.model.head)
    weights = {key[len("head.bcem."):]: value for key, value in state.items() if key.startswith("head.bcem.")}
    if current["enabled"] != bool(weights):
        raise ValueError("BCEM enable_bcem/checkpoint mismatch: current enable_bcem={}, checkpoint BCEM weights={}. "
                         "Use the baseline's original switch and a checkpoint that actually contains BCEM."
                         .format(current["enabled"], bool(weights)))
    embedded = metadata.get("frozen_bcem_config")
    explicit = cfg.offline_crv.get("baseline_config")
    sources = []
    if embedded is not None:
        if not isinstance(embedded, dict) or embedded.get("version") != 1:
            raise ValueError("Unsupported BCEM config metadata in baseline checkpoint")
        canonical = bcem_config_contract(dict(enable_bcem=embedded.get("enabled"),
                                               hidden_channels=embedded.get("arguments", {}).get("hidden_channels", 256),
                                               bcem={key: value for key, value in embedded.get("arguments", {}).items()
                                                     if key != "hidden_channels"}))
        _assert_equal(canonical, embedded, "checkpoint BCEM metadata schema")
        sources.append(("checkpoint.frozen_bcem_config", embedded))
    if explicit:
        path = Path(explicit).resolve()
        sources.append((str(path), _contract_from_file(path)))
    elif embedded is None and (current["enabled"] or weights):
        parent = Path(cfg.offline_crv.baseline_checkpoint).resolve().parent
        # Read only known config-snapshot filenames, not arbitrary .py scripts.
        candidates = [path for path in parent.glob("*.py") if path.name == "config.py"
                      or re.fullmatch(r"\d{8}_\d{6}(?:_\d{6})?_.+\.py", path.name)]
        if len(candidates) != 1:
            raise ValueError("Cannot confirm original BCEM config: {} config snapshots beside baseline_checkpoint. "
                             "Set offline_crv.baseline_config to its ORIGINAL training config snapshot. "
                             "BCEM weights alone cannot verify ray/sampling/context settings."
                             .format(len(candidates)))
        sources.append((str(candidates[0]), _contract_from_file(candidates[0])))
    if current["enabled"] and not sources:
        raise ValueError("Cannot confirm original BCEM configuration")
    for name, expected in sources:
        _assert_equal(expected, current, name)
    if current["enabled"]:
        # Also verify tensors during cache-only training, where no full model
        # is constructed. Avoid perturbing either the CPU or CUDA RNG stream.
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(0)
            module = BackgroundContextEnhancement(**current["arguments"])
        try:
            module.load_state_dict(weights, strict=True)
        except RuntimeError as error:
            raise ValueError("BCEM weights are missing, unexpected or shape-incompatible with the original config") from error
    return dict(contract=current, sources=[name for name, _ in sources],
                status="verified" if current["enabled"] else "disabled")
