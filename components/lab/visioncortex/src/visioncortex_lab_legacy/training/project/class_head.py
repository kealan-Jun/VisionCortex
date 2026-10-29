"""Class head responsibilities for explicitly authorized training."""
from __future__ import annotations

import hashlib
import re
from typing import Any



def transfer_class_outputs(
    source: dict,
    target: dict,
    old_count: int,
    new_count: int,
    *,
    migrate_legacy_tip_box: bool = False,
) -> list[str]:
    """Preserve only verified shared class output rows when adding new classes."""
    import torch

    if not 0 < old_count < new_count:
        raise ValueError("Class output transfer requires an ontology extension")
    if migrate_legacy_tip_box and (old_count, new_count) != (21, 23):
        raise ValueError(
            "Legacy tip-box migration requires the verified 21-to-23 class ontology"
        )
    copied = []
    with torch.no_grad():
        for key, value in target.items():
            previous = source.get(key)
            if previous is None or not (".cv3." in key or ".one2one_cv3." in key):
                continue
            if (
                value.ndim in {1, 4}
                and previous.ndim == value.ndim
                and previous.shape[0] == old_count
                and value.shape[0] == new_count
                and previous.shape[1:] == value.shape[1:]
            ):
                initial_tip = value[11].clone() if migrate_legacy_tip_box else None
                value[:old_count].copy_(
                    previous.to(device=value.device, dtype=value.dtype)
                )
                if migrate_legacy_tip_box:
                    value[22].copy_(
                        previous[11].to(device=value.device, dtype=value.dtype)
                    )
                    value[11].copy_(initial_tip)
                copied.append(key)
    if (
        not copied
        or not any(k.endswith(".weight") for k in copied)
        or not any(k.endswith(".bias") for k in copied)
    ):
        raise ValueError("No compatible class output weights and biases to preserve")
    return copied



def restrict_to_class_outputs(model: Any, class_count: int) -> list[str]:
    """Freeze a verified YOLO classification probe without changing forward mode."""
    import torch

    names = []
    for name, module in model.named_modules():
        if (
            re.search(r"(?:^|\.)(?:one2one_)?cv3\.\d+\.2$", name)
            and isinstance(module, torch.nn.Conv2d)
            and module.out_channels == class_count
            and module.kernel_size == (1, 1)
            and module.bias is not None
        ):
            names.extend([name + ".weight", name + ".bias"])
    if not names:
        raise ValueError("No verified classification output convolutions found")
    selected = set(names)
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name in selected)
    freeze_normalization_statistics(model)
    return sorted(names)



def freeze_normalization_statistics(model: Any) -> None:
    # The trainer calls model.train() at each epoch. Apply again before every
    # training batch; setting the whole detector to eval would change its output
    # contract and break the training loss.
    import torch

    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.eval()



def frozen_state_sha256(model: Any, mutable_names: list[str]) -> str:
    import torch

    h = hashlib.sha256()
    mutable = set(mutable_names)
    for name, value in sorted(model.state_dict().items()):
        if name in mutable:
            continue
        tensor = value.detach().cpu().contiguous()
        h.update(f"{name}\0{tensor.dtype}\0{list(tensor.shape)}\0".encode())
        h.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()



def synchronize_frozen_ema(model: Any, ema: Any, mutable_names: list[str]) -> None:
    """Keep fixed features exact instead of accumulating EMA rounding drift."""
    import torch

    mutable = set(mutable_names)
    state = model.state_dict()
    with torch.no_grad():
        for name, target in ema.state_dict().items():
            if name not in mutable:
                target.copy_(state[name].to(device=target.device, dtype=target.dtype))
