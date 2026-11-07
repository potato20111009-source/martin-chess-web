from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import chess
import numpy as np

from .encoding import ACTION_SIZE, encode_board, move_to_action

try:
    from rust_mcts._rust_mcts import RustMCTS as _RustMCTS
except ImportError:
    _RustMCTS = None

FPU_REDUCTION = 0.25
VALUE_DISCOUNT = 0.99
RESIGN_VALUE_THRESHOLD = -0.95
RESIGN_CONSECUTIVE_EVALUATIONS = 3


def sample_move_from_visits(
    visits: dict[chess.Move, int], temperature: float
) -> chess.Move:
    """Sample a move from MCTS visit counts, with temperature zero as argmax."""
    if not visits:
        raise ValueError("Cannot sample a move from an empty MCTS visit map.")
    if temperature < 0:
        raise ValueError("Move-sampling temperature must be non-negative.")

    moves = list(visits)
    counts = np.asarray([visits[move] for move in moves], dtype=np.float64)
    if np.any(counts < 0):
        raise ValueError("MCTS visit counts must be non-negative.")
    if temperature == 0:
        return moves[int(np.argmax(counts))]

    positive = counts > 0
    if not np.any(positive):
        raise ValueError("MCTS visit counts must contain at least one positive value.")
    moves = [move for move, is_positive in zip(moves, positive) if is_positive]
    counts = counts[positive]
    weights = np.power(counts / counts.max(), 1.0 / temperature)
    probabilities = weights / weights.sum()
    return moves[int(np.random.choice(len(moves), p=probabilities))]


@dataclass
class SearchNode:
    prior: float
    repetition_penalty: float = 0.0
    forced_value: float | None = None
    visits: int = 0
    value_sum: float = 0.0
    children: dict[chess.Move, "SearchNode"] = field(default_factory=dict)

    @property
    def mean_value(self) -> float:
        return self.value_sum / self.visits if self.visits else 0.0


class MCTS:
    def __init__(
        self,
        model: Any | None,
        device: Any,
        simulations: int = 64,
        c_puct: float = 1.5,
        dirichlet_alpha: float = 0.3,
        noise_fraction: float = 0.3,
        repetition_penalty: float = 0.8,
        inference_fn: Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]] | None = None,
        tactical_checks: bool = True,
        value_discount: float = VALUE_DISCOUNT,
        retain_search_tree: bool = False,
        rust_model_path: str | Path | None = None,
    ) -> None:
        if simulations < 1:
            raise ValueError("MCTS simulations must be at least 1.")
        if not 1.0 <= c_puct <= 2.5:
            raise ValueError("c_puct must be in the recommended range [1.0, 2.5].")
        if dirichlet_alpha <= 0:
            raise ValueError("dirichlet_alpha must be positive.")
        if not 0.0 <= noise_fraction <= 1.0:
            raise ValueError("noise_fraction must be in [0, 1].")
        if not 0.0 <= repetition_penalty <= 1.0:
            raise ValueError("repetition_penalty must be in [0, 1].")
        if not 0.0 < value_discount <= 1.0:
            raise ValueError("value_discount must be in (0, 1].")
        if model is None and inference_fn is None and _RustMCTS is None:
            raise ValueError("MCTS requires a model or an inference callback.")
        if rust_model_path is not None and _RustMCTS is None:
            raise RuntimeError(
                "An ONNX model path was provided, but the Rust MCTS extension "
                "is unavailable."
            )
        if inference_fn is not None:
            self._rust_backend = None
        elif rust_model_path is not None and _RustMCTS is not None:
            self._rust_backend = _RustMCTS(str(rust_model_path))
        else:
            self._rust_backend = _RustMCTS() if _RustMCTS is not None else None
        if model is not None and self._rust_backend is None:
            model.to(device)
        self.model = model
        self.device = device
        self.simulations = simulations
        self.c_puct = c_puct
        self.dirichlet_alpha = dirichlet_alpha
        self.noise_fraction = noise_fraction
        self.repetition_penalty = repetition_penalty
        self.inference_fn = inference_fn
        self.evaluation_cache: dict[str, tuple[np.ndarray, float]] = {}
        self._evaluation_cache_max_entries = 256
        self.tactical_checks = tactical_checks
        self.value_discount = value_discount
        self.retain_search_tree = retain_search_tree
        self._rust_search_diagnostics_printed = False
        self.last_root_value: float | None = None
        self.last_root_values: list[float | None] = []
        self.last_root_nodes: list[SearchNode | None] = []

    def _predict(self, board: chess.Board) -> tuple[np.ndarray, float]:
        probabilities, values = self._predict_batch([board])
        return probabilities[0], float(values[0])

    def _predict_batch(
        self, boards: list[chess.Board]
    ) -> tuple[np.ndarray, np.ndarray]:
        if not boards:
            raise ValueError("Cannot predict an empty batch of chess positions.")
        positions_by_fen: dict[str, list[int]] = {}
        for index, board in enumerate(boards):
            positions_by_fen.setdefault(board.fen(), []).append(index)

        probabilities = np.empty(
            (len(boards), ACTION_SIZE), dtype=np.float32
        )
        values = np.empty(len(boards), dtype=np.float32)
        missing_fens: list[str] = []
        missing_boards: list[chess.Board] = []
        for fen, indices in positions_by_fen.items():
            cached = self.evaluation_cache.get(fen)
            if cached is None:
                missing_fens.append(fen)
                missing_boards.append(boards[indices[0]])
                continue
            cached_probabilities, cached_value = cached
            for index in indices:
                probabilities[index] = cached_probabilities
                values[index] = cached_value

        if missing_boards:
            features = np.stack([encode_board(board) for board in missing_boards])
            inferred_probabilities, inferred_values = self._predict_features(features)
            for batch_index, fen in enumerate(missing_fens):
                cached_value = float(inferred_values[batch_index])
                cached_probabilities = inferred_probabilities[batch_index].copy()
                if len(self.evaluation_cache) >= self._evaluation_cache_max_entries:
                    self.evaluation_cache.pop(next(iter(self.evaluation_cache)))
                self.evaluation_cache[fen] = (
                    cached_probabilities,
                    cached_value,
                )
                for index in positions_by_fen[fen]:
                    probabilities[index] = cached_probabilities
                    values[index] = cached_value
        return probabilities, values

    def _predict_features(
        self, features: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        if features.ndim != 4 or features.shape[0] == 0:
            raise ValueError("Inference features must have shape (N, C, H, W).")
        if self.inference_fn is not None:
            probabilities, values = self.inference_fn(features)
            if (
                probabilities.shape[0] != features.shape[0]
                or values.reshape(-1).shape[0] != features.shape[0]
            ):
                raise RuntimeError("Inference callback returned a mismatched batch.")
            return probabilities, values.reshape(-1)
        if self.model is None:
            raise RuntimeError("MCTS model is unavailable for local inference.")
        predict_numpy = getattr(self.model, "predict_numpy", None)
        if not callable(predict_numpy):
            raise RuntimeError(
                "MCTS models must provide a predict_numpy() inference method."
            )
        return predict_numpy(features, self.device)

    @staticmethod
    def _terminal_value(board: chess.Board) -> float:
        outcome = board.outcome(claim_draw=True)
        if outcome is None or outcome.winner is None:
            return 0.0
        return 1.0 if outcome.winner == board.turn else -1.0

    @staticmethod
    def find_mate_in_one(board: chess.Board) -> chess.Move | None:
        """Return a legal move that checkmates immediately, if one exists."""
        for move in list(board.legal_moves):
            board.push(move)
            is_mate = board.is_checkmate()
            board.pop()
            if is_mate:
                return move
        return None

    @classmethod
    def find_forced_mate_in_two(cls, board: chess.Board) -> chess.Move | None:
        """Find a move that forces checkmate after every legal opponent reply."""
        attacker = board.turn
        for move in list(board.legal_moves):
            board.push(move)
            if board.is_checkmate():
                board.pop()
                return move
            if board.is_game_over(claim_draw=True):
                board.pop()
                continue

            forced = True
            replies = list(board.legal_moves)
            if not replies:
                forced = False
            for reply in replies:
                board.push(reply)
                if board.is_game_over(claim_draw=True):
                    forced = False
                elif not cls.find_mate_in_one(board):
                    forced = False
                board.pop()
                if not forced:
                    break
            board.pop()
            if forced and board.turn == attacker:
                return move
        return None

    @classmethod
    def _allows_forced_mate_in_two(
        cls, board: chess.Board, move: chess.Move
    ) -> bool:
        """Check whether a move permits the opponent to force mate within two moves."""
        board.push(move)
        try:
            return bool(
                cls.find_mate_in_one(board)
                or cls.find_forced_mate_in_two(board)
            )
        finally:
            board.pop()

    def _expand(
        self, node: SearchNode, board: chess.Board, detect_mate: bool = True
    ) -> float:
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return self._terminal_value(board)
        if detect_mate:
            mate_move = None
            if self.tactical_checks:
                mate_move = self.find_mate_in_one(board)
                if mate_move is None:
                    mate_move = self.find_forced_mate_in_two(board)
            if mate_move is not None:
                node.children = {mate_move: SearchNode(prior=1.0)}
                return 1.0

        priors, value = self._predict(board)
        return self._expand_with_prediction(node, board, priors, value)

    def _expand_with_prediction(
        self,
        node: SearchNode,
        board: chess.Board,
        priors: np.ndarray,
        value: float,
    ) -> float:
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return self._terminal_value(board)
        legal_priors = np.asarray(
            [priors[move_to_action(move)] for move in legal_moves], dtype=np.float64
        )
        total = float(legal_priors.sum())
        if not math.isfinite(total) or total <= 0:
            legal_priors.fill(1.0 / len(legal_moves))
        else:
            legal_priors /= total
        children: dict[chess.Move, SearchNode] = {}
        for move, prior in zip(legal_moves, legal_priors):
            repetition = False
            if board.move_stack:
                board.push(move)
                repetition = board.is_repetition(2)
                board.pop()
            children[move] = SearchNode(
                prior=float(prior),
                repetition_penalty=self.repetition_penalty if repetition else 0.0,
            )
        node.children = children
        return value

    def search_batch(
        self,
        boards: list[chess.Board],
        add_root_noise: bool = False,
        time_limit_seconds: float | None = None,
    ) -> list[tuple[dict[chess.Move, int], np.ndarray]]:
        """Search multiple independent roots while batching every neural inference."""
        
        if time_limit_seconds is not None and time_limit_seconds <= 0:
            raise ValueError("Search time limit must be positive.")
        if self._rust_backend is not None:
            return self._search_batch_rust(boards, add_root_noise)
        search_started = time.monotonic()
        if not boards:
            self.last_root_values = []
            self.last_root_nodes = []
            return []
        if self.model is not None:
            self.model.eval()
        self.last_root_values = [None] * len(boards)
        self.last_root_nodes = (
            [None] * len(boards) if self.retain_search_tree else []
        )
        results: list[tuple[dict[chess.Move, int], np.ndarray] | None] = [
            None
        ] * len(boards)
        roots: list[tuple[int, chess.Board, SearchNode]] = []
        for index, board in enumerate(boards):
            mate_move = None
            mate_distance = 0
            if self.tactical_checks:
                mate_move = self.find_mate_in_one(board)
                if mate_move is not None:
                    mate_distance = 1
                if mate_move is None:
                    mate_move = self.find_forced_mate_in_two(board)
                    if mate_move is not None:
                        mate_distance = 3
            if mate_move is not None:
                policy = np.zeros(ACTION_SIZE, dtype=np.float32)
                policy[move_to_action(mate_move)] = 1.0
                solved_root = SearchNode(prior=1.0)
                solved_root.children[mate_move] = SearchNode(
                    prior=1.0, visits=self.simulations
                )
                solved_root.visits = self.simulations
                if self.retain_search_tree:
                    self.last_root_nodes[index] = solved_root
                results[index] = ({mate_move: self.simulations}, policy)
                self.last_root_values[index] = self.value_discount**mate_distance
                continue
            roots.append((index, board, SearchNode(prior=1.0)))

        if roots:
            if self.retain_search_tree:
                for result_index, _, root in roots:
                    self.last_root_nodes[result_index] = root
            root_priors, values = self._predict_batch([board for _, board, _ in roots])
            for batch_index, (result_index, board, root) in enumerate(roots):
                self._expand_with_prediction(
                    root, board, root_priors[batch_index], float(values[batch_index])
                )
                if root.children and self.tactical_checks:
                    safe_moves = [
                        move
                        for move in root.children
                        if not self._allows_forced_mate_in_two(board, move)
                    ]
                    if safe_moves:
                        root.children = {
                            move: root.children[move] for move in safe_moves
                        }
                        prior_total = sum(
                            child.prior for child in root.children.values()
                        )
                        if prior_total > 0:
                            for child in root.children.values():
                                child.prior /= prior_total
                    else:
                        for child in root.children.values():
                            child.forced_value = -1.0
                if add_root_noise:
                    self._add_root_noise(root)

            for simulation_index in range(self.simulations):
                if (
                    simulation_index > 0
                    and time_limit_seconds is not None
                    and time.monotonic() - search_started >= time_limit_seconds
                ):
                    break
                pending: list[
                    tuple[SearchNode, chess.Board, list[SearchNode], int]
                ] = []
                for result_index, board, root in roots:
                    node = root
                    search_board = board
                    path = [node]
                    pushed_moves = 0
                    while node.children:
                        move, node = self._select_child(node)
                        search_board.push(move)
                        pushed_moves += 1
                        path.append(node)
                    if search_board.is_game_over(claim_draw=True):
                        leaf_value = self._terminal_value(search_board)
                        self._backup(path, leaf_value, self.value_discount)
                        for _ in range(pushed_moves):
                            search_board.pop()
                    else:
                        mate_move = (
                            self.find_mate_in_one(search_board)
                            if self.tactical_checks
                            else None
                        )
                        if mate_move is not None:
                            node.children = {mate_move: SearchNode(prior=1.0)}
                            self._backup(
                                path, self.value_discount, self.value_discount
                            )
                            for _ in range(pushed_moves):
                                search_board.pop()
                        else:
                            pending.append(
                                (node, search_board, path, pushed_moves)
                            )

                if pending:
                    leaf_priors, leaf_values = self._predict_batch(
                        [item[1] for item in pending]
                    )
                    for batch_index, (node, leaf_board, path, pushed_moves) in enumerate(
                        pending
                    ):
                        value = self._expand_with_prediction(
                            node,
                            leaf_board,
                            leaf_priors[batch_index],
                            float(leaf_values[batch_index]),
                        )
                        self._backup(path, value, self.value_discount)
                        for _ in range(pushed_moves):
                            leaf_board.pop()

                if all(
                    root.visits >= self.simulations
                    for _, _, root in roots
                ):
                    break

            for result_index, _, root in roots:
                visits = {
                    move: child.visits for move, child in root.children.items()
                }
                self.last_root_values[result_index] = root.mean_value
                policy = np.zeros(ACTION_SIZE, dtype=np.float32)
                for move, count in visits.items():
                    policy[move_to_action(move)] = count
                total_visits = float(policy.sum())
                if total_visits:
                    policy /= total_visits
                results[result_index] = (visits, policy)

        return [
            result
            if result is not None
            else ({}, np.zeros(ACTION_SIZE, dtype=np.float32))
            for result in results
        ]

    def _search_batch_rust(
        self, boards: list[chess.Board], add_root_noise: bool
    ) -> list[tuple[dict[chess.Move, int], np.ndarray]]:
        """Run Rust PUCT search and batched ONNX inference."""
        self.last_root_values = []
        self.last_root_nodes = []
        if not boards:
            return []
        fens: list[str] = []
        histories: list[list[str]] = []
        for board in boards:
            fens.append(board.fen(en_passant="fen"))
            history: list[str] = []
            historical_board = board.copy(stack=True)
            while historical_board.move_stack:
                historical_board.pop()
                history.append(historical_board.fen(en_passant="fen"))
            histories.append(history)

        rust_results = self._rust_backend.search_batch(
            fens,
            histories,
            self.simulations,
            self.c_puct,
            self.value_discount,
            add_root_noise=add_root_noise,
            dirichlet_alpha=self.dirichlet_alpha,
            noise_fraction=self.noise_fraction,
            return_policy=False,
        )
        if len(rust_results) != len(boards):
            raise RuntimeError(
                "Rust MCTS returned a mismatched root batch: "
                f"{len(rust_results)} results for {len(boards)} boards."
            )
        results: list[tuple[dict[chess.Move, int], np.ndarray]] = []
        for board, result in zip(boards, rust_results):
            visits = {
                chess.Move.from_uci(move_uci): int(count)
                for move_uci, count in result["visits"].items()
            }
            policy = np.zeros(ACTION_SIZE, dtype=np.float32)
            total_visits = sum(visits.values())
            if total_visits:
                for move, count in visits.items():
                    policy[move_to_action(move)] = count / total_visits
            if any(move not in board.legal_moves for move in visits):
                raise RuntimeError("Rust MCTS returned an illegal root move.")
            results.append((visits, policy))
            self.last_root_values.append(float(result["root_value"]))
        return results

    @staticmethod
    def _backup(
        path: list[SearchNode],
        leaf_value: float,
        discount: float = VALUE_DISCOUNT,
    ) -> None:
        for visited_node in reversed(path):
            visited_node.visits += 1
            visited_node.value_sum += leaf_value
            leaf_value = -leaf_value * discount

    def _add_root_noise(self, root: SearchNode) -> None:
        if not root.children or self.noise_fraction <= 0:
            return
        moves = list(root.children)
        noise = np.random.dirichlet([self.dirichlet_alpha] * len(moves))
        for move, noise_prior in zip(moves, noise):
            child = root.children[move]
            child.prior = (
                (1.0 - self.noise_fraction) * child.prior
                + self.noise_fraction * float(noise_prior)
            )

    def _selection_score(
        self, parent: SearchNode, child: SearchNode
    ) -> float:
        if child.forced_value is not None:
            return child.forced_value * 1.0e9
        q_value = (
            -child.mean_value
            if child.visits
            else parent.mean_value - FPU_REDUCTION
        )
        return (
            q_value
            - child.repetition_penalty
            + self.c_puct
            * child.prior
            * math.sqrt(max(1, parent.visits))
            / (1 + child.visits)
        )

    def _select_child(self, parent: SearchNode) -> tuple[chess.Move, SearchNode]:
        """Select a PUCT child with vectorized score arithmetic."""
        moves = list(parent.children)
        children = list(parent.children.values())
        visits = np.fromiter((child.visits for child in children), dtype=np.int64)
        priors = np.fromiter((child.prior for child in children), dtype=np.float64)
        value_sums = np.fromiter(
            (child.value_sum for child in children), dtype=np.float64
        )
        penalties = np.fromiter(
            (child.repetition_penalty for child in children), dtype=np.float64
        )
        q_values = np.full(len(children), parent.mean_value - FPU_REDUCTION)
        visited = visits > 0
        np.divide(-value_sums, visits, out=q_values, where=visited)
        scores = (
            q_values
            - penalties
            + self.c_puct
            * priors
            * math.sqrt(max(1, parent.visits))
            / (1 + visits)
        )
        forced_values = np.fromiter(
            (
                np.nan if child.forced_value is None else child.forced_value
                for child in children
            ),
            dtype=np.float64,
        )
        forced = ~np.isnan(forced_values)
        scores[forced] = forced_values[forced] * 1.0e9
        selected = int(np.argmax(scores))
        return moves[selected], children[selected]

    def search(
        self,
        board: chess.Board,
        add_root_noise: bool = False,
    ) -> tuple[dict[chess.Move, int], np.ndarray]:
        if self._rust_backend is not None:
            result = self.search_batch([board], add_root_noise=add_root_noise)[0]
            self.last_root_value = self.last_root_values[0]
            return result
        if self.model is not None:
            self.model.eval()
        self.last_root_value = None
        self.last_root_values = []
        self.last_root_nodes = []
        mate_move = None
        mate_distance = 0
        if self.tactical_checks:
            mate_move = self.find_mate_in_one(board)
            if mate_move is not None:
                mate_distance = 1
            if mate_move is None:
                mate_move = self.find_forced_mate_in_two(board)
                if mate_move is not None:
                    mate_distance = 3
        if mate_move is not None:
            self.last_root_value = self.value_discount**mate_distance
            policy = np.zeros(ACTION_SIZE, dtype=np.float32)
            policy[move_to_action(mate_move)] = 1.0
            solved_root = SearchNode(prior=1.0, visits=self.simulations)
            solved_root.children[mate_move] = SearchNode(
                prior=1.0, visits=self.simulations
            )
            if self.retain_search_tree:
                self.last_root_nodes = [solved_root]
            return {mate_move: self.simulations}, policy

        root = SearchNode(prior=1.0)
        self.last_root_nodes = [root] if self.retain_search_tree else []
        self._expand(root, board, detect_mate=False)

        if root.children and self.tactical_checks:
            safe_moves = [
                move
                for move in root.children
                if not self._allows_forced_mate_in_two(board, move)
            ]
            if safe_moves:
                root.children = {
                    move: root.children[move] for move in safe_moves
                }
                prior_total = sum(child.prior for child in root.children.values())
                if prior_total > 0:
                    for child in root.children.values():
                        child.prior /= prior_total
            else:
                for child in root.children.values():
                    child.forced_value = -1.0
        # Self-play opts in to AlphaZero root noise; evaluation remains deterministic.
        if add_root_noise:
            self._add_root_noise(root)

        for _ in range(self.simulations):
            node = root
            search_board = board
            path = [node]
            pushed_moves = 0
            while node.children:
                move, node = self._select_child(node)
                search_board.push(move)
                pushed_moves += 1
                path.append(node)

            if search_board.is_game_over(claim_draw=True):
                leaf_value = self._terminal_value(search_board)
            else:
                leaf_value = self._expand(node, search_board)
            self._backup(path, leaf_value, self.value_discount)
            for _ in range(pushed_moves):
                search_board.pop()
        self.last_root_value = root.mean_value

        visits = {move: child.visits for move, child in root.children.items()}
        policy = np.zeros(ACTION_SIZE, dtype=np.float32)
        for move, count in visits.items():
            policy[move_to_action(move)] = count
        total_visits = float(policy.sum())
        if total_visits:
            policy /= total_visits
        return visits, policy
