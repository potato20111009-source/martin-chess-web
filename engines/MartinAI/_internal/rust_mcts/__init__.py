"""Python package for Martin's optional Rust MCTS accelerator."""

from ._rust_mcts import RustMCTS, encode_board

__all__ = ["RustMCTS", "encode_board"]
