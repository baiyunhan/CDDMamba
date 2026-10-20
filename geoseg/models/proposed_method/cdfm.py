"""Channel-Decoupled Flow Matching (CDFM).

The implementation follows the method chapter:
1. rank channels by their spatial response variation;
2. assign weak/strong perturbations to clear/ambiguous channels;
3. regress the velocity of a conditional feature-restoration path;
4. use a short Runge--Kutta solver for deterministic inference.
"""

from __future__ import annotations

import math
from typing import Tuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, t: Tensor) -> Tensor:
        half = self.dim // 2
        scale = math.log(10000.0) / max(half - 1, 1)
        frequencies = torch.exp(
            torch.arange(half, device=t.device, dtype=t.dtype) * -scale
        )
        angles = t[:, None] * frequencies[None, :]
        embedding = torch.cat((angles.sin(), angles.cos()), dim=-1)
        if embedding.shape[-1] < self.dim:
            embedding = F.pad(embedding, (0, self.dim - embedding.shape[-1]))
        return embedding


class VelocityBlock(nn.Module):
    def __init__(self, channels: int, hidden_channels: int) -> None:
        super().__init__()
        groups = max(1, min(32, hidden_channels // 8))
        while hidden_channels % groups:
            groups -= 1

        self.input_projection = nn.Conv2d(channels * 4, hidden_channels, 1)
        self.time_embedding = nn.Sequential(
            SinusoidalTimeEmbedding(hidden_channels),
            nn.Linear(hidden_channels, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, hidden_channels),
        )
        self.body = nn.Sequential(
            nn.GroupNorm(groups, hidden_channels),
            nn.SiLU(),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
            nn.GroupNorm(groups, hidden_channels),
            nn.SiLU(),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
        )
        self.output_projection = nn.Conv2d(hidden_channels, channels, 1)

        # A zero-initialized final layer makes the new branch safe at startup.
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)

    def forward(
        self,
        state: Tensor,
        condition: Tensor,
        t: Tensor,
        perturbation_scale: Tensor,
    ) -> Tensor:
        time_feature = self.time_embedding(t).unsqueeze(-1).unsqueeze(-1)
        network_input = torch.cat(
            (state, condition, state - condition, perturbation_scale), dim=1
        )
        hidden = self.input_projection(network_input)
        hidden = hidden + time_feature
        return self.output_projection(self.body(hidden) + hidden)


class CDFM(nn.Module):
    """Restore deep semantics with channel-dependent flow paths.

    Args:
        channels: Number of channels in the deepest encoder feature.
        clear_ratio: Fraction assigned to the semantic-clear group.
        lambda_clear: Perturbation strength for clear channels.
        lambda_ambiguous: Perturbation strength for ambiguous channels.
        rk_steps: Number of RK4 integration steps used during evaluation.
    """

    def __init__(
        self,
        channels: int,
        hidden_channels: int | None = None,
        clear_ratio: float = 0.5,
        lambda_clear: float = 0.15,
        lambda_ambiguous: float = 0.45,
        rk_steps: int = 2,
        antithetic_noise: bool = False,
        grouping_mode: str = "clarity",
        grouping_seed: int = 42,
    ) -> None:
        super().__init__()
        if not 0.0 < clear_ratio < 1.0:
            raise ValueError("clear_ratio must lie in (0, 1)")
        if not 0.0 <= lambda_clear <= lambda_ambiguous:
            raise ValueError("Expected 0 <= lambda_clear <= lambda_ambiguous")
        if rk_steps < 1:
            raise ValueError("rk_steps must be positive")
        if grouping_mode not in {"clarity", "random"}:
            raise ValueError("grouping_mode must be 'clarity' or 'random'")

        hidden_channels = hidden_channels or max(64, channels // 2)
        self.channels = channels
        self.clear_ratio = clear_ratio
        self.lambda_clear = lambda_clear
        self.lambda_ambiguous = lambda_ambiguous
        self.rk_steps = rk_steps
        self.antithetic_noise = antithetic_noise
        self.grouping_mode = grouping_mode

        # A dedicated generator keeps the random-grouping ablation fixed and
        # reproducible without changing model-initialisation RNG state.
        generator = torch.Generator(device="cpu")
        generator.manual_seed(grouping_seed)
        self.register_buffer(
            "random_channel_order",
            torch.randperm(channels, generator=generator),
            persistent=False,
        )

        self.velocity = VelocityBlock(channels, hidden_channels)
        self.output_projection = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
        )
        # The residual contribution grows only when it is useful.
        self.residual_gate = nn.Parameter(torch.tensor(-2.0))

    def _clarity_and_scale(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        # s_c = sqrt(mean_n (X_c,n - mu_c)^2)
        clarity = x.flatten(2).var(dim=-1, unbiased=False).add(1e-6).sqrt()
        clear_count = max(1, int(round(self.channels * self.clear_ratio)))
        if self.grouping_mode == "clarity":
            clear_indices = clarity.topk(clear_count, dim=1, largest=True).indices
        else:
            clear_indices = self.random_channel_order[:clear_count][None, :].expand(
                x.shape[0], -1
            )

        scale = x.new_full((x.shape[0], self.channels), self.lambda_ambiguous)
        scale.scatter_(1, clear_indices, self.lambda_clear)
        return clarity, scale[:, :, None, None]

    @staticmethod
    def _normalised_deterministic_noise(x: Tensor) -> Tensor:
        # Deterministic at evaluation time while retaining a noise-like source.
        noise = torch.tanh(x.detach() - x.detach().mean(dim=(2, 3), keepdim=True))
        std = noise.flatten(2).std(dim=-1, keepdim=True).unsqueeze(-1)
        return noise / std.clamp_min(1e-4)

    def _velocity(self, state: Tensor, condition: Tensor, t: Tensor, scale: Tensor) -> Tensor:
        scale_feature = scale.expand(-1, -1, state.shape[-2], state.shape[-1])
        return self.velocity(state, condition, t, scale_feature)

    def _rk4(self, source: Tensor, condition: Tensor, scale: Tensor) -> Tensor:
        state = source
        batch = source.shape[0]
        dt = 1.0 / self.rk_steps
        for step in range(self.rk_steps):
            t0 = source.new_full((batch,), step * dt)
            tm = source.new_full((batch,), step * dt + 0.5 * dt)
            t1 = source.new_full((batch,), (step + 1) * dt)
            k1 = self._velocity(state, condition, t0, scale)
            k2 = self._velocity(state + 0.5 * dt * k1, condition, tm, scale)
            k3 = self._velocity(state + 0.5 * dt * k2, condition, tm, scale)
            k4 = self._velocity(state + dt * k3, condition, t1, scale)
            state = state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        return state

    def forward(self, x0: Tensor) -> Tuple[Tensor, Tensor]:
        if x0.ndim != 4 or x0.shape[1] != self.channels:
            raise ValueError(
                f"CDFM expected [B, {self.channels}, H, W], got {tuple(x0.shape)}"
            )

        _, scale = self._clarity_and_scale(x0)
        batch = x0.shape[0]

        if self.training:
            epsilon = torch.randn_like(x0)
            t = torch.rand(batch, device=x0.device, dtype=x0.dtype)
            t_map = t[:, None, None, None]

            def estimate(noise: Tensor) -> tuple[Tensor, Tensor]:
                source = x0 + scale * noise
                state_t = (1.0 - t_map) * source + t_map * x0
                target_velocity = x0 - source
                predicted_velocity = self._velocity(state_t, x0, t, scale)
                loss = F.mse_loss(predicted_velocity, target_velocity)
                # Direct endpoint estimate keeps training inexpensive.
                endpoint = state_t + (1.0 - t_map) * predicted_velocity
                return endpoint, loss

            restored, flow_loss = estimate(epsilon)
            if self.antithetic_noise:
                mirrored, mirrored_loss = estimate(-epsilon)
                restored = 0.5 * (restored + mirrored)
                flow_loss = 0.5 * (flow_loss + mirrored_loss)
        else:
            epsilon = self._normalised_deterministic_noise(x0)
            source = x0 + scale * epsilon
            restored = self._rk4(source, x0, scale)
            flow_loss = x0.new_zeros(())

        residual = self.output_projection(restored - x0)
        enhanced = x0 + torch.sigmoid(self.residual_gate) * residual
        return enhanced, flow_loss
