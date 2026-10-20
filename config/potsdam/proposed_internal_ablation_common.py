"""Shared builder for locked Potsdam internal-design ablations."""

import torch

from geoseg.models.proposed_method import ProposedGlobalLocalNetV3


def apply_internal_variant(namespace: dict, weights_name: str, **model_kwargs) -> None:
    namespace["weights_name"] = weights_name
    namespace["weights_path"] = f"model_weights/potsdam/{weights_name}"
    namespace["test_weights_name"] = weights_name
    namespace["log_name"] = f"potsdam/{weights_name}"
    namespace["resume_ckpt_path"] = None
    namespace["seed"] = 42

    net = ProposedGlobalLocalNetV3(
        num_classes=namespace["num_classes"],
        pretrained=True,
        **model_kwargs,
    )
    namespace["net"] = net

    backbone_params = []
    other_params = []
    for name, parameter in net.named_parameters():
        if "backbone" in name:
            backbone_params.append(parameter)
        else:
            other_params.append(parameter)

    optimizer = torch.optim.AdamW(
        [
            {
                "params": backbone_params,
                "lr": namespace["backbone_lr"],
                "weight_decay": namespace["weight_decay"],
            },
            {
                "params": other_params,
                "lr": namespace["lr"],
                "weight_decay": namespace["weight_decay"],
            },
        ],
        lr=namespace["lr"],
        weight_decay=namespace["weight_decay"],
    )
    namespace["optimizer"] = optimizer
    namespace["lr_scheduler"] = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=namespace["max_epoch"]
    )
