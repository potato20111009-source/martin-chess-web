"""PGN and Polyglot chess opening-book lookup with randomized exploration."""

from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import chess
import chess.pgn
import chess.polyglot
import numpy as np

MAX_BOOK_PLY = 5
DEFAULT_EXPLORATION_PROBABILITY = 0.2


class OpeningBookManager:
    """Index PGN openings and query standard Polyglot ``.bin`` book files."""

    def __init__(
        self,
        source: str | Path | Iterable[str | Path] | None = None,
        *,
        exploration_probability: float = DEFAULT_EXPLORATION_PROBABILITY,
        seed: int | None = None,
    ) -> None:
        if (
            isinstance(exploration_probability, bool)
            or not isinstance(exploration_probability, (int, float))
            or not math.isfinite(exploration_probability)
            or not 0.0 <= exploration_probability <= 1.0
        ):
            raise ValueError("exploration_probability must be finite and in [0, 1].")
        self.exploration_probability = exploration_probability
        self._rng = np.random.default_rng(seed)
        self._pgn_moves: dict[
            tuple[str, ...], dict[str, float]
        ] = defaultdict(lambda: defaultdict(float))
        self._polyglot_paths: list[Path] = []
        if source is not None:
            self.load(source)

    @staticmethod
    def _position_key(board: chess.Board) -> tuple[str, ...]:
        """Ignore move counters so equivalent opening positions share entries."""
        return tuple(board.fen().split()[:4])

    @staticmethod
    def _expand_sources(
        source: str | Path | Iterable[str | Path],
    ) -> list[Path]:
        sources = [source] if isinstance(source, (str, Path)) else list(source)
        if not sources:
            raise ValueError("At least one opening-book source must be provided.")

        paths: list[Path] = []
        for source_item in sources:
            path = Path(source_item)
            if not path.exists():
                raise FileNotFoundError(f"Opening-book source not found: {path}")
            if path.is_dir():
                paths.extend(
                    child
                    for child in sorted(path.iterdir(), key=lambda item: item.name.lower())
                    if child.is_file() and child.suffix.lower() in {".pgn", ".bin"}
                )
            elif path.is_file():
                paths.append(path)
            else:
                raise ValueError(f"Opening-book source is not a file or directory: {path}")

        unsupported = [path for path in paths if path.suffix.lower() not in {".pgn", ".bin"}]
        if unsupported:
            raise ValueError(
                "Unsupported opening-book file(s): "
                + ", ".join(str(path) for path in unsupported)
            )
        if not paths:
            raise FileNotFoundError("No .pgn or Polyglot .bin books found in source.")
        return paths

    def load(self, source: str | Path | Iterable[str | Path]) -> int:
        """Load all supplied books, returning the number of indexed PGN games."""
        games_loaded = 0
        for path in self._expand_sources(source):
            if path.suffix.lower() == ".pgn":
                games_loaded += self.load_pgn(path)
            else:
                self.load_polyglot(path)
        return games_loaded

    def load_pgn(self, path: str | Path) -> int:
        """Index mainline moves from a PGN file and return its game count."""
        pgn_path = Path(path)
        if not pgn_path.is_file():
            raise FileNotFoundError(f"PGN opening book not found: {pgn_path}")

        games_loaded = 0
        with pgn_path.open("r", encoding="utf-8-sig", errors="strict") as pgn_file:
            while True:
                game = chess.pgn.read_game(pgn_file)
                if game is None:
                    break
                games_loaded += 1
                if game.errors:
                    messages = "; ".join(str(error) for error in game.errors)
                    raise ValueError(
                        f"Invalid PGN game {games_loaded} in {pgn_path}: {messages}"
                    )

                board = game.board()
                for ply, move in enumerate(game.mainline_moves()):
                    if ply > MAX_BOOK_PLY:
                        break
                    if move not in board.legal_moves:
                        raise ValueError(
                            f"Illegal move {move.uci()} in game {games_loaded} "
                            f"of opening book {pgn_path}."
                        )
                    self._pgn_moves[self._position_key(board)][move.uci()] += 1.0
                    board.push(move)
        return games_loaded

    def load_polyglot(self, path: str | Path) -> None:
        """Register and validate a standard Polyglot binary opening book."""
        book_path = Path(path)
        if not book_path.is_file():
            raise FileNotFoundError(f"Polyglot opening book not found: {book_path}")
        if book_path.suffix.lower() != ".bin":
            raise ValueError(f"Polyglot book must use the .bin extension: {book_path}")
        try:
            with chess.polyglot.open_reader(str(book_path)):
                pass
        except (OSError, chess.polyglot.PolyglotError) as error:
            raise ValueError(f"Could not read Polyglot book {book_path}: {error}") from error
        if book_path not in self._polyglot_paths:
            self._polyglot_paths.append(book_path)

    def get_book_move(
        self,
        board: chess.Board,
        current_ply: int,
        temperature: float = 1.0,
    ) -> chess.Move | None:
        """Return a weighted opening move, or ``None`` for MCTS/exploration.

        The book is only consulted through ply five. With the default 20%
        epsilon, a legal move from the book is deliberately ignored on one
        fifth of eligible lookups to preserve free exploration.
        """
        if not isinstance(board, chess.Board):
            raise TypeError("board must be a python-chess Board.")
        if not isinstance(current_ply, int) or isinstance(current_ply, bool) or current_ply < 0:
            raise ValueError("current_ply must be a non-negative integer.")
        if (
            isinstance(temperature, bool)
            or not isinstance(temperature, (int, float))
            or not math.isfinite(temperature)
            or temperature < 0
        ):
            raise ValueError("temperature must be finite and non-negative.")
        if current_ply > MAX_BOOK_PLY or board.is_game_over(claim_draw=True):
            return None
        if self._rng.random() < self.exploration_probability:
            return None

        weights: dict[str, float] = defaultdict(float)
        for move_uci, weight in self._pgn_moves.get(self._position_key(board), {}).items():
            weights[move_uci] += weight

        for book_path in self._polyglot_paths:
            try:
                with chess.polyglot.open_reader(str(book_path)) as reader:
                    for entry in reader.find_all(board):
                        weights[entry.move.uci()] += max(1.0, float(entry.weight))
            except (OSError, chess.polyglot.PolyglotError) as error:
                raise RuntimeError(
                    f"Failed to query Polyglot opening book {book_path}: {error}"
                ) from error

        legal_weights = {
            chess.Move.from_uci(move_uci): weight
            for move_uci, weight in weights.items()
            if weight > 0 and chess.Move.from_uci(move_uci) in board.legal_moves
        }
        if not legal_weights:
            return None
        moves = list(legal_weights)
        move_weights = np.asarray([legal_weights[move] for move in moves], dtype=np.float64)
        if temperature == 0:
            return moves[int(np.argmax(move_weights))]

        logits = np.log(move_weights) / temperature
        logits -= logits.max()
        probabilities = np.exp(logits)
        probabilities /= probabilities.sum()
        return moves[int(self._rng.choice(len(moves), p=probabilities))]
