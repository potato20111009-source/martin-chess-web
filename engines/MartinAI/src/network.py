"""Shared residual backbone with factorized policy and scalar value heads."""

from __future__ import annotations

import torch
from torch import nn

from .encoding import ACTION_SIZE, BOARD_PLANES


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.activation = nn.ReLU(inplace=True)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.activation(inputs + self.layers(inputs))


class ChessNetwork(nn.Module):
    """Predict action logits and a [-1, 1] value from the side-to-move's view."""

    def __init__(self, channels: int = 64, residual_blocks: int = 4) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(BOARD_PLANES, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            *(ResidualBlock(channels) for _ in range(residual_blocks)),
        )

        policy_width = 16
        self.policy_projection = nn.Conv2d(channels, policy_width * 5, kernel_size=1)
        self.policy_from = nn.Linear(policy_width, policy_width, bias=False)
        self.policy_to = nn.Linear(policy_width, policy_width, bias=False)

        self.value_head = nn.Sequential(
            nn.Conv2d(channels, 2, kernel_size=1),
            nn.BatchNorm2d(2),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(2 * 8 * 8, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),
            nn.Tanh(),
        )

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.backbone(inputs)
        batch_size = features.shape[0]
        projected = self.policy_projection(features).reshape(batch_size, 5, 16, 8, 8)
        projected = projected.permute(0, 1, 3, 4, 2).reshape(batch_size, 5, 64, 16)
        from_features = self.policy_from(projected)
        to_features = self.policy_to(projected)
        policy = torch.einsum("bpsd,bptd->bpst", from_features, to_features)
        policy = policy.permute(0, 2, 3, 1).reshape(batch_size, ACTION_SIZE)
        value = self.value_head(features).squeeze(-1)
        return policy, value
