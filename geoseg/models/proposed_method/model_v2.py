"""V2 model using the same RegNet-Y backbone protocol as UrbanSSF-V3."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict

import timm
import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .cdfm import CDFM
from .dp_mpc import DPMPC
from .dgr_csmamba import DGRCSMamba
from .model import ConvNormAct, DecoderBlock


DEFAULT_REGNET_WEIGHTS = os.environ.get("REGNET_WEIGHTS")


def build_regnet_backbone(
    backbone_name: str,
    pretrained: bool,
    pretrained_path: str | None,
) -> nn.Module:
    kwargs = dict(
        features_only=True,
        output_stride=32,
        out_indices=(1, 2, 3, 4),
    )
    if pretrained:
        if pretrained_path:
            if not Path(pretrained_path).is_file():
                raise FileNotFoundError(
                    "REGNET_WEIGHTS points to a missing file: "
                    f"{pretrained_path!r}."
                )
            return timm.create_model(
                backbone_name,
                pretrained=True,
                pretrained_cfg_overlay={"file": pretrained_path},
                **kwargs,
            )
        # When REGNET_WEIGHTS is not set, timm downloads the same official
        # ImageNet checkpoint into its cache.
        return timm.create_model(backbone_name, pretrained=True, **kwargs)
    return timm.create_model(backbone_name, pretrained=False, **kwargs)


class ProposedGlobalLocalNetV2(nn.Module):
    """RegNet-Y + CDFM + multi-scale DP-MPC + DGR-CSMamba."""

    def __init__(
        self,
        num_classes: int = 6,
        backbone_name: str = "regnety_016.tv2_in1k",
        pretrained: bool = True,
        pretrained_path: str | None = DEFAULT_REGNET_WEIGHTS,
        fusion_dim: int = 256,
        polynomial_order: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.backbone = build_regnet_backbone(
            backbone_name=backbone_name,
            pretrained=pretrained,
            pretrained_path=pretrained_path,
        )
        c1, c2, c3, c4 = self.backbone.feature_info.channels()

        self.cdfm = CDFM(c4, hidden_channels=fusion_dim, rk_steps=2)
        self.dp_mpc1 = DPMPC(c1, kernel_size=5, polynomial_order=polynomial_order)
        self.dp_mpc2 = DPMPC(c2, kernel_size=7, polynomial_order=polynomial_order)
        self.dp_mpc3 = DPMPC(c3, kernel_size=9, polynomial_order=polynomial_order)

        self.local_projection1 = nn.Conv2d(c1, fusion_dim, 1, bias=False)
        self.local_projection2 = nn.Conv2d(c2, fusion_dim, 1, bias=False)
        self.local_projection3 = nn.Conv2d(c3, fusion_dim, 1, bias=False)
        self.local_fusion = nn.Sequential(
            ConvNormAct(fusion_dim * 3, fusion_dim, 1),
            ConvNormAct(fusion_dim, fusion_dim),
        )
        self.dgr_csmamba = DGRCSMamba(
            global_channels=c4,
            local_channels=fusion_dim,
            dim=fusion_dim,
            candidate_size=16,
            d_state=16,
            dropout=dropout,
        )

        self.decoder2 = DecoderBlock(fusion_dim, c2, 128)
        self.decoder1 = DecoderBlock(128, c1, 64)
        self.segmentation_head = nn.Sequential(
            ConvNormAct(64, 64),
            nn.Dropout2d(dropout),
            nn.Conv2d(64, num_classes, 1),
        )

    def _local_pyramid(self, p1: Tensor, p2: Tensor, p3: Tensor) -> Tensor:
        target_size = p3.shape[-2:]
        p1 = self.local_projection1(F.adaptive_avg_pool2d(p1, target_size))
        p2 = self.local_projection2(F.adaptive_avg_pool2d(p2, target_size))
        p3 = self.local_projection3(p3)
        return self.local_fusion(torch.cat((p1, p2, p3), dim=1))

    def forward(self, x: Tensor) -> Tensor | Dict[str, Tensor]:
        input_size = x.shape[-2:]
        e1, e2, e3, e4 = self.backbone(x)
        global_feature, flow_loss = self.cdfm(e4)
        p1 = self.dp_mpc1(e1)
        p2 = self.dp_mpc2(e2)
        p3 = self.dp_mpc3(e3)
        local_feature = self._local_pyramid(p1, p2, p3)

        fused = self.dgr_csmamba(global_feature, local_feature)
        decoded2 = self.decoder2(fused, p2)
        decoded1 = self.decoder1(decoded2, p1)
        logits = self.segmentation_head(decoded1)
        logits = F.interpolate(logits, size=input_size, mode="bilinear", align_corners=False)
        if self.training:
            return {"logits": logits, "flow_loss": flow_loss}
        return logits

