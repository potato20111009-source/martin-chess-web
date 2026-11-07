"""Shared residual backbone with factorized policy and scalar value heads."""

from __future__ import annotations

import numpy as np
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

    def predict_numpy(
        self,
        features: np.ndarray,
        configured_device: torch.device | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Run inference for MCTS and return policy/value arrays as NumPy."""
        model_device = next(self.parameters()).device
        if (
            configured_device is not None
            and model_device.type != configured_device.type
        ):
            raise RuntimeError(
                "MCTS model device type does not match the configured device: "
                f"configured={configured_device.type}, model={model_device.type}."
            )
        tensors = torch.from_numpy(np.asarray(features, dtype=np.float32)).to(
            model_device, non_blocking=model_device.type == "cuda"
        )
        with torch.no_grad():
            if model_device.type == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    logits, values = self(tensors)
            else:
                logits, values = self(tensors)
            probabilities = torch.softmax(logits.float(), dim=1)
        return (
            probabilities.cpu().numpy(),
            values.float().reshape(-1).cpu().numpy(),
        )
