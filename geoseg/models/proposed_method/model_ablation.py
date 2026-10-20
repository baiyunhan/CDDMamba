"""Controlled core-module ablations of the V3-Detail architecture."""

from __future__ import annotations

from typing import Dict

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .cdfm import CDFM
from .dp_mpc import DPMPC
from .dgr_csmamba import DGRCSMamba, ResidualGatedDGRFusion
from .model import ConvNormAct, DecoderBlock
from .model_v2 import DEFAULT_REGNET_WEIGHTS, build_regnet_backbone


class ProposedCoreAblationNet(nn.Module):
    """V3-Detail with exactly one optional core module removed."""

    def __init__(
        self,
        num_classes: int = 6,
        backbone_name: str = "regnety_016.tv2_in1k",
        pretrained: bool = True,
        pretrained_path: str | None = DEFAULT_REGNET_WEIGHTS,
        fusion_dim: int = 256,
        polynomial_order: int = 3,
        dropout: float = 0.1,
        use_cdfm: bool = True,
        use_dp_mpc: bool = True,
        use_dgr_csmamba: bool = True,
        cdfm_antithetic_noise: bool = False,
        dgr_residual_gate: bool = False,
    ) -> None:
        super().__init__()
        disabled = 3 - sum((use_cdfm, use_dp_mpc, use_dgr_csmamba))
        if disabled != 1:
            raise ValueError("A controlled ablation must disable exactly one core module")

        self.num_classes = num_classes
        self.use_cdfm = use_cdfm
        self.use_dp_mpc = use_dp_mpc
        self.use_dgr_csmamba = use_dgr_csmamba
        self.use_stable_dgr_fusion = dgr_residual_gate
        self.backbone = build_regnet_backbone(
            backbone_name=backbone_name,
            pretrained=pretrained,
            pretrained_path=pretrained_path,
        )
        c1, c2, c3, c4 = self.backbone.feature_info.channels()

        if use_cdfm:
            self.cdfm = CDFM(
                c4,
                hidden_channels=fusion_dim,
                rk_steps=2,
                antithetic_noise=cdfm_antithetic_noise,
            )
        if use_dp_mpc:
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

        if dgr_residual_gate:
            self.dgr_csmamba = ResidualGatedDGRFusion(
                global_channels=c4,
                local_channels=fusion_dim,
                dim=fusion_dim,
                candidate_size=16,
                d_state=16,
                dropout=dropout,
                use_dgr=use_dgr_csmamba,
            )
        elif use_dgr_csmamba:
            self.dgr_csmamba = DGRCSMamba(
                global_channels=c4,
                local_channels=fusion_dim,
                dim=fusion_dim,
                candidate_size=16,
                d_state=16,
                dropout=dropout,
            )
        else:
            self.global_projection = nn.Conv2d(c4, fusion_dim, 1, bias=False)
            self.simple_fusion = nn.Sequential(
                ConvNormAct(fusion_dim * 2, fusion_dim, 1),
                ConvNormAct(fusion_dim, fusion_dim),
            )

        self.decoder2 = DecoderBlock(fusion_dim, c2, 128)
        self.decoder1 = DecoderBlock(128, c1, 64)
        self.detail_stem = nn.Sequential(
            ConvNormAct(3, 32, 3, stride=2),
            ConvNormAct(32, 32),
        )
        self.detail_decoder = DecoderBlock(64, 32, 64)
        self.segmentation_head = nn.Sequential(
            ConvNormAct(64, 64),
            nn.Dropout2d(dropout),
            nn.Conv2d(64, num_classes, 1),
        )
        # Keep the final V3 architecture identical outside the selected
        # ablation.  The formal protocol sets aux_weight=0, but retaining this
        # head avoids changing the module inventory or parameter accounting.
        self.auxiliary_head = nn.Sequential(
            ConvNormAct(c1, 64),
            nn.Dropout2d(dropout),
            nn.Conv2d(64, num_classes, 1),
        )

    def _local_pyramid(self, p1: Tensor, p2: Tensor, p3: Tensor) -> Tensor:
        target_size = p3.shape[-2:]
        p1_down = self.local_projection1(F.adaptive_avg_pool2d(p1, target_size))
        p2_down = self.local_projection2(F.adaptive_avg_pool2d(p2, target_size))
        p3_projected = self.local_projection3(p3)
        return self.local_fusion(torch.cat((p1_down, p2_down, p3_projected), dim=1))

    def forward(self, x: Tensor) -> Tensor | Dict[str, Tensor]:
        input_size = x.shape[-2:]
        detail = self.detail_stem(x)
        e1, e2, e3, e4 = self.backbone(x)

        if self.use_cdfm:
            global_feature, flow_loss = self.cdfm(e4)
        else:
            global_feature = e4
            flow_loss = e4.new_zeros(())

        if self.use_dp_mpc:
            p1 = self.dp_mpc1(e1)
            p2 = self.dp_mpc2(e2)
            p3 = self.dp_mpc3(e3)
        else:
            p1, p2, p3 = e1, e2, e3
        local_feature = self._local_pyramid(p1, p2, p3)

        if self.use_dgr_csmamba or self.use_stable_dgr_fusion:
            fused = self.dgr_csmamba(global_feature, local_feature)
        else:
            global_projected = self.global_projection(global_feature)
            global_projected = F.interpolate(
                global_projected,
                size=local_feature.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            fused = self.simple_fusion(
                torch.cat((global_projected, local_feature), dim=1)
            )

        decoded2 = self.decoder2(fused, p2)
        decoded1 = self.decoder1(decoded2, p1)
        detailed = self.detail_decoder(decoded1, detail)
        logits = self.segmentation_head(detailed)
        logits = F.interpolate(logits, size=input_size, mode="bilinear", align_corners=False)
        if self.training:
            aux_logits = self.auxiliary_head(p1)
            aux_logits = F.interpolate(
                aux_logits, size=input_size, mode="bilinear", align_corners=False
            )
            return {
                "logits": logits,
                "aux_logits": aux_logits,
                "flow_loss": flow_loss,
            }
        return logits
