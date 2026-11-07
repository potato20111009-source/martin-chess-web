"""Run a multi-anchor Maia gauntlet and estimate MartinAI's absolute Elo."""

from __future__ import annotations

import argparse
import concurrent.futures
import math
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import chess
import chess.engine
import chess.pgn
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.mcts import MCTS
from src.network import ChessNetwork
from src.persistence import (
    ARCHIVED_MODELS_DIR,
    MODEL_PATH,
    PROJECT_ROOT,
    load_model_weights,
    load_training_state,
    save_training_state,
    update_model_archive_metadata,
)


ENGINE_DIRECTORY = PROJECT_ROOT / "benchmark_engines" / "maia"
DEFAULT_LC0 = ENGINE_DIRECTORY / "lc0.exe"
DEFAULT_MARTIN_EXE = PROJECT_ROOT / "build" / "MartinAI.exe"


@dataclass(frozen=True)
class MaiaAnchor:
    rating: int
    weights: Path


@dataclass
class GameResult:
    rating: int
    game_number: int
    result: str
    pgn: str


def discover_maia_anchors(directory: Path = ENGINE_DIRECTORY) -> list[MaiaAnchor]:
    """Find all maia-<rating>.pb.gz weights in ascending rating order."""
    anchors: list[MaiaAnchor] = []
    pattern = re.compile(r"^maia-(\d+)\.pb\.gz$", re.IGNORECASE)
    for path in directory.glob("maia-*.pb.gz"):
        match = pattern.match(path.name)
        if match:
            anchors.append(MaiaAnchor(int(match.group(1)), path))
    return sorted(anchors, key=lambda anchor: anchor.rating)


def weighted_absolute_elo(
    results: list[dict[str, int | float]],
) -> tuple[float, list[dict[str, int | float]]]:
    """Fit one Elo rating to all anchor scores using weighted logistic regression."""
    if not results:
        raise ValueError("At least one Maia result is required.")
    estimates: list[dict[str, int | float]] = []
    for result in results:
        games = int(result["games"])
        score_rate = float(result["score_rate"])
        opponent_elo = int(result["opponent_elo"])
        if games < 1 or not 0.0 <= score_rate <= 1.0:
            raise ValueError("Each anchor needs games and a score rate in [0, 1].")
        smoothed_score = min(max(score_rate, 0.5 / games), 1.0 - 0.5 / games)
        delta = 400.0 * math.log10(smoothed_score / (1.0 - smoothed_score))
        estimates.append(
            {
                **result,
                "delta_elo": delta,
                "estimated_elo": opponent_elo + delta,
            }
        )

    def expected_score(rating: float, opponent_elo: int) -> float:
        exponent = min(
            20.0,
            max(-20.0, (opponent_elo - rating) * math.log(10.0) / 400.0),
        )
        return 1.0 / (1.0 + math.exp(exponent))

    def gradient(rating: float) -> float:
        return sum(
            int(result["games"])
            * (
                expected_score(rating, int(result["opponent_elo"]))
                - min(
                    max(
                        float(result["score_rate"]),
                        0.5 / int(result["games"]),
                    ),
                    1.0 - 0.5 / int(result["games"]),
                )
            )
            for result in results
        )

    low = min(int(result["opponent_elo"]) for result in results) - 2000.0
    high = max(int(result["opponent_elo"]) for result in results) + 2000.0
    if gradient(low) >= 0.0:
        combined = low
    elif gradient(high) <= 0.0:
        combined = high
    else:
        for _ in range(100):
            midpoint = (low + high) / 2.0
            if gradient(midpoint) > 0.0:
                high = midpoint
            else:
                low = midpoint
        combined = (low + high) / 2.0
    return combined, estimates


def adaptive_exit_reason(
    anchor_rating: int,
    wins: int,
    games: int,
    score_rate: float,
    consecutive_low_score_anchors: int,
) -> tuple[str | None, int]:
    """Stop after a winless tier or two successive tiers below 15% score."""
    if games < 1:
        raise ValueError("games must be positive.")
    if wins == 0:
        return (
            f"MartinAI recorded no wins at Maia {anchor_rating} "
            f"({games} games; score {score_rate:.1%}).",
            0,
        )
    if score_rate < 0.15:
        consecutive_low_score_anchors += 1
    else:
        consecutive_low_score_anchors = 0
    if consecutive_low_score_anchors >= 2:
        return (
            f"Performance dropped below threshold at Maia {anchor_rating} "
            "for two consecutive anchors; stopping further tests.",
            consecutive_low_score_anchors,
        )
    return None, consecutive_low_score_anchors


def _parse_anchors(value: str) -> list[int] | None:
    if value.strip().lower() == "all":
        return None
    try:
        ratings = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError:
        raise argparse.ArgumentTypeError(
            "Anchors must be 'all' or comma-separated Maia Elo ratings."
        ) from None
    if not ratings:
        raise argparse.ArgumentTypeError("At least one Maia anchor is required.")
    return ratings


def _play_gauntlet_game(
    anchor: MaiaAnchor,
    game_number: int,
    martin_executable: Path | None,
    lc0_executable: Path,
    model: ChessNetwork | None,
    device: torch.device | None,
    simulations: int,
    movetime_ms: int,
    max_plies: int,
) -> GameResult:
    martin_is_white = game_number % 2 == 0
    board = chess.Board()
    game = chess.pgn.Game()
    game.headers["Event"] = f"MartinAI vs Maia {anchor.rating} Gauntlet"
    game.headers["Round"] = f"{anchor.rating}.{game_number + 1}"
    game.headers["White"] = "MartinAI" if martin_is_white else f"Maia-{anchor.rating}"
    game.headers["Black"] = f"Maia-{anchor.rating}" if martin_is_white else "MartinAI"
    game.headers["MaiaElo"] = str(anchor.rating)
    game.headers["TimeControl"] = f"{movetime_ms / 1000:g}+0"
    node: chess.pgn.GameNode = game

    martin_engine: chess.engine.SimpleEngine | None = None
    maia_engine: chess.engine.SimpleEngine | None = None
    search: MCTS | None = None
    try:
        if martin_executable is not None:
            martin_engine = chess.engine.SimpleEngine.popen_uci(
                str(martin_executable),
                timeout=max(30.0, movetime_ms / 1000.0 + 10.0),
                cwd=str(PROJECT_ROOT),
                stderr=subprocess.DEVNULL,
            )
        else:
            if model is None or device is None:
                raise RuntimeError("Python model or Martin UCI executable is unavailable.")
            search = MCTS(
                model,
                device,
                simulations=simulations,
                tactical_checks=False,
            )

        maia_engine = chess.engine.SimpleEngine.popen_uci(
            [str(lc0_executable), f"--weights={anchor.weights}"],
            timeout=max(30.0, movetime_ms / 1000.0 + 10.0),
            cwd=str(ENGINE_DIRECTORY),
            stderr=subprocess.DEVNULL,
        )

        while board.ply() < max_plies and not board.is_game_over(claim_draw=True):
            martin_to_move = board.turn == martin_is_white
            if martin_to_move and martin_engine is not None:
                engine_move = martin_engine.play(
                    board, chess.engine.Limit(time=movetime_ms / 1000.0)
                ).move
            elif martin_to_move:
                assert search is not None
                visits, _ = search.search_batch(
                    [board],
                    time_limit_seconds=movetime_ms / 1000.0,
                )[0]
                engine_move = (
                    max(visits.items(), key=lambda item: item[1])[0]
                    if visits
                    else None
                )
            else:
                assert maia_engine is not None
                engine_move = maia_engine.play(
                    board, chess.engine.Limit(time=movetime_ms / 1000.0)
                ).move
            if engine_move is None:
                raise RuntimeError(
                    "An engine returned no move for a non-terminal position."
                )
            node = node.add_variation(engine_move)
            board.push(engine_move)

        outcome = board.outcome(claim_draw=True)
        if outcome is None or outcome.winner is None:
            result = "1/2-1/2"
        else:
            martin_won = outcome.winner == martin_is_white
            result = ("1-0" if martin_is_white else "0-1") if martin_won else (
                "0-1" if martin_is_white else "1-0"
            )
        game.headers["Result"] = result
        if board.ply() >= max_plies and outcome is None:
            node.comment = "Draw adjudicated at configured ply limit."
        pgn = game.accept(
            chess.pgn.StringExporter(headers=True, variations=False, comments=True)
        )
        return GameResult(anchor.rating, game_number, result, pgn)
    finally:
        if martin_engine is not None:
            martin_engine.quit()
        if maia_engine is not None:
            maia_engine.quit()


def run_gauntlet(args: argparse.Namespace) -> dict[str, Any]:
    anchors = discover_maia_anchors(args.engine_directory)
    if not anchors:
        raise FileNotFoundError(
            f"No Maia weights matching maia-<rating>.pb.gz in {args.engine_directory}"
        )
    if args.anchors is not None:
        selected_ratings = set(args.anchors)
        anchors = [anchor for anchor in anchors if anchor.rating in selected_ratings]
        missing = selected_ratings - {anchor.rating for anchor in anchors}
        if missing:
            raise FileNotFoundError(
                "Requested Maia anchor weights not found: "
                + ", ".join(str(rating) for rating in sorted(missing))
            )
    if not args.lc0.is_file():
        raise FileNotFoundError(f"lc0 executable not found: {args.lc0}")
    if not 10 <= args.games <= 20 or args.games % 2:
        raise ValueError("--games must be an even number between 10 and 20.")
    if args.workers < 1 or args.max_plies < 1 or args.movetime < 1:
        raise ValueError("--workers, --max-plies and --movetime must be positive.")

    martin_executable: Path | None
    model: ChessNetwork | None = None
    device: torch.device | None = None
    use_martin_executable = (
        args.martin_exe.is_file()
        and args.checkpoint.resolve() == MODEL_PATH.resolve()
        and (args.martin_exe.parent / "latest_model.pt").is_file()
    )
    if use_martin_executable:
        martin_executable = args.martin_exe
    elif args.checkpoint.is_file():
        martin_executable = None
        device = torch.device(
            "cuda" if args.device == "auto" and torch.cuda.is_available()
            else "cpu" if args.device == "auto"
            else args.device
        )
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                f"CUDA device {device} requested, but CUDA is unavailable."
            )
        model = ChessNetwork().to(device)
        load_model_weights(model, device, args.checkpoint)
    else:
        raise FileNotFoundError(
            f"Neither Martin executable nor checkpoint exists: "
            f"{args.martin_exe}, {args.checkpoint}"
        )

    completed: list[GameResult] = []
    per_anchor: list[dict[str, int | float]] = []
    low_score_streak = 0
    exit_message: str | None = None
    for anchor in anchors:
        anchor_results: list[GameResult] = []
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=min(args.workers, args.games)
        ) as executor:
            jobs = [
                executor.submit(
                    _play_gauntlet_game,
                    anchor,
                    game_number,
                    martin_executable,
                    args.lc0,
                    model,
                    device,
                    args.simulations,
                    args.movetime,
                    args.max_plies,
                )
                for game_number in range(args.games)
            ]
            for future in concurrent.futures.as_completed(jobs):
                result = future.result()
                completed.append(result)
                anchor_results.append(result)
                print(
                    f"[Gauntlet] Maia {result.rating}: "
                    f"{len(anchor_results)}/{args.games} ({result.result})",
                    flush=True,
                )

        wins = sum(
            result.result
            == ("1-0" if result.game_number % 2 == 0 else "0-1")
            for result in anchor_results
        )
        losses = sum(
            result.result
            == ("0-1" if result.game_number % 2 == 0 else "1-0")
            for result in anchor_results
        )
        draws = len(anchor_results) - wins - losses
        score_rate = (wins + 0.5 * draws) / len(anchor_results)
        anchor_summary: dict[str, int | float] = {
            "opponent_elo": anchor.rating,
            "games": len(anchor_results),
            "wins": wins,
            "draws": draws,
            "losses": losses,
            "score_rate": score_rate,
        }
        per_anchor.append(anchor_summary)
        exit_message, low_score_streak = adaptive_exit_reason(
            anchor.rating,
            wins,
            len(anchor_results),
            score_rate,
            low_score_streak,
        )
        if exit_message is not None:
            print(
                f"[Adaptive Exit] {exit_message}",
                flush=True,
            )
            break

    pgn_directory = PROJECT_ROOT / "pgn_logs"
    pgn_directory.mkdir(parents=True, exist_ok=True)
    pgn_path = pgn_directory / "absolute_elo_gauntlet.pgn"
    with pgn_path.open("w", encoding="utf-8") as pgn_file:
        for result in sorted(completed, key=lambda item: (item.rating, item.game_number)):
            pgn_file.write(result.pgn + "\n\n")

    absolute_elo, estimates = weighted_absolute_elo(per_anchor)
    print(f"[Result] Real Elo: {absolute_elo:.0f}")
    print(
        f"[Result] Completed {len(per_anchor)}/{len(anchors)} Maia tiers; "
        f"PGN: {pgn_path}"
    )

    state = load_training_state()
    state["base_elo"] = absolute_elo
    state["current_elo"] = absolute_elo
    state["absolute_elo_history"] = state.get("absolute_elo_history", []) + [
        {
            "iteration": int(state.get("iteration", 0)) - 1,
            "elo": absolute_elo,
            "anchors": estimates,
        }
    ]
    save_training_state(state)

    archive_paths = sorted(
        ARCHIVED_MODELS_DIR.glob("model_iter_*.pt"),
        key=lambda path: int(_archive_rating_from_name(path)),
    )
    if archive_paths:
        latest_archive = archive_paths[-1]
        update_model_archive_metadata(
            latest_archive,
            {
                "base_elo": absolute_elo,
                "current_elo": absolute_elo,
                "absolute_elo_evaluation": estimates,
            },
        )
    return {
        "elo": absolute_elo,
        "anchors": estimates,
        "pgn_path": pgn_path,
    }


def _archive_rating_from_name(path: Path) -> int:
    match = re.fullmatch(r"model_iter_(\d+)\.pt", path.name)
    return int(match.group(1)) if match else -1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MartinAI Maia Elo gauntlet")
    parser.add_argument("--engine-directory", type=Path, default=ENGINE_DIRECTORY)
    parser.add_argument("--lc0", type=Path, default=DEFAULT_LC0)
    parser.add_argument("--martin-exe", type=Path, default=DEFAULT_MARTIN_EXE)
    parser.add_argument("--checkpoint", type=Path, default=MODEL_PATH)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--anchors",
        type=_parse_anchors,
        default=None,
        help="Comma-separated Maia Elo anchors or 'all' (default: every detected anchor)",
    )
    parser.add_argument("--games", type=int, default=10)
    parser.add_argument("--movetime", type=int, default=1000, help="Milliseconds per move")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--simulations", type=int, default=10000)
    parser.add_argument("--max-plies", type=int, default=400)
    return parser.parse_args()


def main() -> None:
    try:
        run_gauntlet(parse_args())
    except (FileNotFoundError, ValueError, RuntimeError, OSError) as error:
        print(f"[Gauntlet] Error: {error}", file=sys.stderr)
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
