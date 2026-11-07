"""Export an in-memory Martin model for isolated Rust/ONNX inference."""

from __future__ import annotations

import inspect
import os
import tempfile
from pathlib import Path

import torch

from .encoding import BOARD_PLANES
from .network import ChessNetwork


class _InferenceOutputs(torch.nn.Module):
    def __init__(self, model: ChessNetwork) -> None:
        super().__init__()
        self.model = model

    def forward(self, boards: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logits, values = self.model(boards)
        return torch.softmax(logits.float(), dim=1), values.float()


def export_model_to_onnx(model: ChessNetwork, output_path: str | Path) -> Path:
    """Atomically export current weights with the dynamic shape expected by Rust."""
    destination = Path(output_path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f"{destination.stem}_", suffix=".onnx", dir=destination.parent
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    was_training = model.training
    original_device = next(model.parameters()).device
    try:
        model.to("cpu")
        inference_model = _InferenceOutputs(model).eval()
        export_options: dict[str, object] = {
            "input_names": ["input"],
            "output_names": ["policy", "value"],
            "dynamic_axes": {
                "input": {0: "batch"},
                "policy": {0: "batch"},
                "value": {0: "batch"},
            },
            "opset_version": 17,
            "do_constant_folding": True,
        }
        if "dynamo" in inspect.signature(torch.onnx.export).parameters:
            export_options["dynamo"] = False
        sample = torch.zeros(1, BOARD_PLANES, 8, 8, dtype=torch.float32)
        with torch.no_grad():
            torch.onnx.export(
                inference_model,
                (sample,),
                str(temporary_path),
                **export_options,
            )
        os.replace(temporary_path, destination)
    finally:
        model.to(original_device)
        model.train(was_training)
        if temporary_path.exists():
            temporary_path.unlink()
    return destination
