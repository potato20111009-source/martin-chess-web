"""Incremental Elo evaluation against archived Martin model anchors."""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any

import chess
import torch
from tqdm.auto import tqdm

from src.evaluation import (
    evaluate_candidate_vs_incumbent,
    evaluate_candidate_vs_incumbent_isolated,
)
from src.mcts import _RustMCTS
from src.network import ChessNetwork
from src.persistence import (
    ARCHIVED_MODELS_DIR,
    PROJECT_ROOT,
    load_archived_model,
    update_model_archive_metadata,
)


def score_to_elo_delta(score: float, games: int) -> float:
    """Convert match score to rating difference, clipping all-win/loss samples."""
    if not 0.0 <= score <= 1.0:
        raise ValueError("Match score must be in [0, 1].")
    if games < 1:
        raise ValueError("Game count must be positive.")
    smoothed_score = min(max(score, 0.5 / games), 1.0 - 0.5 / games)
    return 400.0 * math.log10(smoothed_score / (1.0 - smoothed_score))


def _archive_iteration(path: Path) -> int | None:
    match = re.fullmatch(r"model_iter_(\d+)\.pt", path.name)
    return int(match.group(1)) if match else None


def _find_model_anchors(iteration: int) -> list[tuple[str, Path]]:
    ARCHIVED_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    archives = [
        (archive_iteration, path)
        for path in ARCHIVED_MODELS_DIR.glob("model_iter_*.pt")
        if (archive_iteration := _archive_iteration(path)) is not None
        and archive_iteration < iteration
    ]
    archives.sort(key=lambda entry: entry[0])
    if not archives:
        return []

    baseline_iteration, baseline_path = archives[0]
    incumbent = next(
        (entry for entry in reversed(archives) if entry[0] < iteration),
        None,
    )
    anchors = [(f"baseline_iter_{baseline_iteration}", baseline_path)]
    if incumbent is not None and incumbent[1] != baseline_path:
        anchors.append((f"incumbent_iter_{incumbent[0]}", incumbent[1]))
    return anchors


def evaluate_promoted_candidate(
    candidate: ChessNetwork,
    incumbent: ChessNetwork,
    device: torch.device,
    iteration: int,
    state: dict[str, Any],
    games: int = 10,
    simulations: int = 110,
    max_moves: int = 400,
    batch_size: int = 32,
    pgn_directory: str | Path | None = None,
    c_puct: float = 1.5,
    value_discount: float = 0.99,
    isolated: bool = False,
) -> dict[str, Any]:
    """Estimate a promoted model's incremental rating from historical-model matches."""
    if games < 2 or games % 2:
        raise ValueError("Elo games must be an even number of at least 2.")
    if simulations < 1 or max_moves < 1 or batch_size < 1:
        raise ValueError("Elo search settings must be positive.")

    current_rating = state.get("current_elo")
    if not isinstance(current_rating, (int, float)):
        current_rating = None
    anchors = _find_model_anchors(iteration)
    if not anchors:
        anchors = [("incumbent", None)]

    if isolated:
        if _RustMCTS is None:
            raise RuntimeError(
                "Isolated Elo evaluation requires the Rust MCTS extension."
            )

    estimates: list[dict[str, Any]] = []
    total_games = 0
    total_points = 0.0
    for anchor_name, archive_path in anchors:
        if archive_path is None:
            anchor_model = incumbent
        else:
            anchor_model, _ = load_archived_model(
                archive_path,
                torch.device("cpu") if isolated else device,
            )

        match_directory = (
            Path(pgn_directory)
            if pgn_directory is not None
            else PROJECT_ROOT / "pgn_logs" / "elo"
        )
        if isolated:
            match = evaluate_candidate_vs_incumbent_isolated(
                candidate,
                anchor_model,
                device,
                iteration=iteration,
                games=games,
                simulations=simulations,
                max_moves=max_moves,
                pgn_directory=match_directory,
                batch_size=batch_size,
                c_puct=c_puct,
                value_discount=value_discount,
            )
        else:
            match = evaluate_candidate_vs_incumbent(
                candidate,
                anchor_model,
                device,
                iteration=iteration,
                games=games,
                simulations=simulations,
                max_moves=max_moves,
                pgn_directory=match_directory,
                batch_size=batch_size,
                c_puct=c_puct,
                value_discount=value_discount,
            )
        score = float(match["score_rate"])
        match_games = int(match["games"])
        match_points = int(match["wins"]) + 0.5 * int(match["draws"])
        total_games += match_games
        total_points += match_points
        estimates.append(
            {
                "anchor": anchor_name,
                "archive": None if archive_path is None else str(archive_path),
                "games": match_games,
                "wins": int(match["wins"]),
                "draws": int(match["draws"]),
                "losses": int(match["losses"]),
                "score_rate": score,
                "delta_elo": score_to_elo_delta(score, match_games),
            }
        )

    if total_games < 1:
        raise RuntimeError("Historical-model Elo evaluation produced no games.")
    delta_elo = score_to_elo_delta(total_points / total_games, total_games)
    estimated_rating = (
        float(current_rating) + delta_elo
        if current_rating is not None
        else None
    )

    result: dict[str, Any] = {
        "iteration": iteration,
        "games_per_anchor": games,
        "anchors": estimates,
        "delta_elo": delta_elo,
        "estimated_elo": estimated_rating,
        "total_games": total_games,
        "calibrated": current_rating is not None,
    }
    state.setdefault("elo_delta_history", []).append(result)
    if estimated_rating is not None:
        state["current_elo"] = estimated_rating

    if estimated_rating is None:
        tqdm.write(
            "[Multi-anchor Elo] Finished: rating uncalibrated "
            f"(ΔElo: {delta_elo:+.0f}, {total_games} games/"
            f"{len(estimates)} anchors); run eval_absolute_elo.py to calibrate."
        )
    else:
        assert delta_elo is not None
        tqdm.write(
            "[Multi-anchor Elo] Finished: "
            f"rating {estimated_rating:.0f} (ΔElo: {delta_elo:+.0f}, "
            f"{total_games} games/{len(estimates)} anchors)."
        )

    return result


def record_archive_elo(path: Path, elo_result: dict[str, Any]) -> None:
    """Attach the latest incremental Elo assessment to its model archive."""
    update_model_archive_metadata(
        path,
        {
            "current_elo": elo_result.get("estimated_elo"),
            "elo_delta": elo_result.get("delta_elo"),
            "elo_evaluation": elo_result,
        },
    )
