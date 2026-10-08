"""Omit inactive branches without changing the legacy initialization stream."""
import torch
from torch.utils._python_dispatch import TorchDispatchMode


class _ReplayInitializationRNG(TorchDispatchMode):
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        # Meta initialization allocates no weights and consumes no RNG. Replay
        # its random draws on CPU, where the original head was initialized.
        if func in (torch.ops.aten.uniform_.default, torch.ops.aten.normal_.default):
            tensor = args[0]
            if tensor.device.type == "meta":
                scratch = torch.empty_like(tensor, device="cpu")
                func(scratch, *args[1:], **kwargs)
        elif func == torch.ops.aten.randn.default and torch.device(kwargs.get("device", "cpu")).type == "meta":
            func(*args, **dict(kwargs, device=torch.device("cpu")))
        return func(*args, **kwargs)


def build_optional_module(factory, enabled, name, omitted_parameter_names):
    if enabled:
        return factory()
    # Only shape-only module descriptors are constructed. No inactive weights
    # survive in the model, its optimizer, CUDA memory or its checkpoint.
    with torch.device("meta"), _ReplayInitializationRNG():
        descriptor = factory()
    omitted_parameter_names.extend(name + "." + key for key, _ in descriptor.named_parameters())
    return None
