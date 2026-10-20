"""Complete global-local collaborative segmentation network."""

from __future__ import annotations

from typing import Dict

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from mamba_ssm import Mamba

from .cdfm import CDFM
from .dp_mpc import DPMPC
from .dgr_csmamba import DGRCSMamba


class ConvNormAct(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
    ) -> None:
        padding = kernel_size // 2
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=padding,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )


class SpatialMambaBlock(nn.Module):
    def __init__(self, channels: int, d_state: int = 16) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(channels)
        self.mamba = Mamba(d_model=channels, d_state=d_state, d_conv=4, expand=2)

    def forward(self, x: Tensor) -> Tensor:
        batch, channels, height, width = x.shape
        tokens = x.flatten(2).transpose(1, 2)
        tokens = tokens + self.mamba(self.norm(tokens))
        return tokens.transpose(1, 2).reshape(batch, channels, height, width)


class EncoderStage(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, use_mamba: bool = False
    ) -> None:
        super().__init__()
        blocks: list[nn.Module] = [
            ConvNormAct(in_channels, out_channels, 3, stride=2),
            ConvNormAct(out_channels, out_channels, 3),
        ]
        if use_mamba:
            blocks.append(SpatialMambaBlock(out_channels))
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x: Tensor) -> Tensor:
        return self.blocks(x)


class DecoderBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.projection = nn.Sequential(
            ConvNormAct(in_channels + skip_channels, out_channels),
            ConvNormAct(out_channels, out_channels),
        )

    def forward(self, x: Tensor, skip: Tensor) -> Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.projection(torch.cat((x, skip), dim=1))


class ProposedGlobalLocalNet(nn.Module):
    """CDFM + DP-MPC + DGR-CSMamba semantic segmentation model."""

    def __init__(
        self,
        num_classes: int = 6,
        input_channels: int = 3,
        encoder_channels: tuple[int, int, int, int] = (64, 128, 256, 512),
        fusion_dim: int = 256,
        polynomial_order: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        c1, c2, c3, c4 = encoder_channels
        self.num_classes = num_classes

        self.encoder1 = EncoderStage(input_channels, c1, use_mamba=False)
        self.encoder2 = EncoderStage(c1, c2, use_mamba=False)
        self.encoder3 = EncoderStage(c2, c3, use_mamba=True)
        self.encoder4 = EncoderStage(c3, c4, use_mamba=True)

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
        self.decoder2 = DecoderBlock(fusion_dim, c2, c2)
        self.decoder1 = DecoderBlock(c2, c1, c1)
        self.segmentation_head = nn.Sequential(
            ConvNormAct(c1, c1),
            nn.Dropout2d(dropout),
            nn.Conv2d(c1, num_classes, 1),
        )

    def _local_pyramid(self, p1: Tensor, p2: Tensor, p3: Tensor) -> Tensor:
        target_size = p3.shape[-2:]
        p1 = self.local_projection1(
            F.adaptive_avg_pool2d(p1, output_size=target_size)
        )
        p2 = self.local_projection2(
            F.adaptive_avg_pool2d(p2, output_size=target_size)
        )
        p3 = self.local_projection3(p3)
        return self.local_fusion(torch.cat((p1, p2, p3), dim=1))

    def forward(self, x: Tensor) -> Tensor | Dict[str, Tensor]:
        input_size = x.shape[-2:]
        e1 = self.encoder1(x)       # 1/2
        e2 = self.encoder2(e1)      # 1/4
        e3 = self.encoder3(e2)      # 1/8
        e4 = self.encoder4(e3)      # 1/16

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


if __name__ == "__main__":
    model = ProposedGlobalLocalNet(num_classes=6)
    sample = torch.randn(1, 3, 256, 256)
    output = model(sample)
    print(output["logits"].shape, output["flow_loss"].item())

