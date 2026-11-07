"""Chess board and move encodings shared by training and search."""

from __future__ import annotations

import chess
import numpy as np

BASE_BOARD_PLANES = 18
HISTORY_FRAMES = 5
BOARD_PLANES = BASE_BOARD_PLANES * HISTORY_FRAMES
PROMOTION_PIECES = (None, chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT)
PROMOTION_TO_INDEX = {piece: index for index, piece in enumerate(PROMOTION_PIECES)}
ACTION_SIZE = 64 * 64 * len(PROMOTION_PIECES)


def _encode_position(board: chess.Board) -> np.ndarray:
    encoded = np.zeros((BASE_BOARD_PLANES, 8, 8), dtype=np.float32)
    for square, piece in board.piece_map().items():
        plane = (piece.piece_type - 1) + (0 if piece.color == chess.WHITE else 6)
        encoded[plane, chess.square_rank(square), chess.square_file(square)] = 1.0

    encoded[12].fill(float(board.turn == chess.WHITE))
    encoded[13].fill(float(board.has_kingside_castling_rights(chess.WHITE)))
    encoded[14].fill(float(board.has_queenside_castling_rights(chess.WHITE)))
    encoded[15].fill(float(board.has_kingside_castling_rights(chess.BLACK)))
    encoded[16].fill(float(board.has_queenside_castling_rights(chess.BLACK)))
    if board.ep_square is not None:
        encoded[17, chess.square_rank(board.ep_square), chess.square_file(board.ep_square)] = 1.0
    return encoded


def encode_board(board: chess.Board) -> np.ndarray:
    """Encode the current position and four prior plies as 90 float32 planes.

    Missing history at the start of a game is padded with the earliest available
    position, so input shape remains fixed.
    """
    position_history = [_encode_position(board)]
    historical_board = board.copy(stack=True)
    for _ in range(HISTORY_FRAMES - 1):
        if historical_board.move_stack:
            historical_board.pop()
        position_history.append(_encode_position(historical_board))
    return np.concatenate(position_history, axis=0)


def move_to_action(move: chess.Move) -> int:
    """Map a legal chess move to a stable from/to/promotion action index."""
    try:
        promotion = PROMOTION_TO_INDEX[move.promotion]
    except KeyError as exc:
        raise ValueError(f"Unsupported promotion piece: {move.promotion}") from exc
    return ((move.from_square * 64 + move.to_square) * len(PROMOTION_PIECES)) + promotion


def action_to_move(action: int) -> chess.Move:
    """Decode an action index into a move; callers must still check legality."""
    if not 0 <= int(action) < ACTION_SIZE:
        raise ValueError(f"Action index out of range: {action}")
    from_to, promotion_index = divmod(int(action), len(PROMOTION_PIECES))
    from_square, to_square = divmod(from_to, 64)
    return chess.Move(
        from_square,
        to_square,
        promotion=PROMOTION_PIECES[promotion_index],
    )
