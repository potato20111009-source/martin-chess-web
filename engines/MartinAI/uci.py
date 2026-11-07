"""UCI protocol adapter for Martin ONNX anchors and PyTorch checkpoints."""

from __future__ import annotations

import argparse
import importlib.util
import os
import shutil
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import Any

ENGINE_DIR = Path(__file__).resolve().parent
if str(ENGINE_DIR) not in sys.path:
    sys.path.insert(0, str(ENGINE_DIR))
MODEL_PATH = ENGINE_DIR / "model_anchor.onnx"


def _runtime_paths() -> tuple[
    Path, Path, Path, list[Path], list[Path], list[Path]
]:
    frozen = bool(getattr(sys, "frozen", False))
    executable_dir = Path(sys.executable).resolve().parent
    bundle_dir = Path(getattr(sys, "_MEIPASS", executable_dir)).resolve()
    project_root = executable_dir if frozen else Path(__file__).resolve().parent
    import_roots = list(dict.fromkeys((bundle_dir, executable_dir, project_root)))
    for root in import_roots:
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))

    runtime_roots = set(import_roots)
    runtime_roots.update(
        Path(entry).resolve()
        for entry in sys.path
        if entry and Path(entry).is_dir()
    )
    onnxruntime_dirs: set[Path] = set()
    rust_package_dirs: set[Path] = set()
    for root in runtime_roots:
        capi = root / "onnxruntime" / "capi"
        if capi.is_dir():
            onnxruntime_dirs.add(capi.resolve())
        elif any(root.glob("onnxruntime*.dll")):
            onnxruntime_dirs.add(root.resolve())
        site_packages = root / "Lib" / "site-packages"
        capi = site_packages / "onnxruntime" / "capi"
        if capi.is_dir():
            onnxruntime_dirs.add(capi.resolve())
        for relative in (
            Path("rust_mcts") / "python" / "rust_mcts",
            Path("rust_mcts"),
        ):
            rust_dir = root / relative
            if rust_dir.is_dir() and any(
                child.suffix.lower() in {".dll", ".pyd"}
                for child in rust_dir.iterdir()
                if child.is_file()
            ):
                rust_package_dirs.add(rust_dir.resolve())

    try:
        rust_spec = importlib.util.find_spec("rust_mcts")
    except (ImportError, ValueError):
        rust_spec = None
    if rust_spec is not None and rust_spec.submodule_search_locations:
        rust_package_dirs.update(
            Path(location).resolve()
            for location in rust_spec.submodule_search_locations
            if Path(location).is_dir()
        )

    for capi_dir in onnxruntime_dirs:
        if capi_dir.parent.name == "capi":
            runtime_roots.add(capi_dir.parent.parent)

    dll_dirs: set[Path] = set(onnxruntime_dirs | rust_package_dirs)
    for root in runtime_roots:
        for relative in (
            Path("torch") / "lib",
        ):
            candidate = root / relative
            if candidate.is_dir():
                dll_dirs.add(candidate.resolve())
        nvidia_root = root / "nvidia"
        if nvidia_root.is_dir():
            dll_dirs.update(
                path.resolve()
                for path in nvidia_root.rglob("*")
                if path.is_dir() and path.name.lower() in {"bin", "lib"}
            )
    return (
        project_root,
        executable_dir,
        bundle_dir,
        sorted(dll_dirs, key=str),
        sorted(onnxruntime_dirs, key=str),
        sorted(rust_package_dirs, key=str),
    )


def _copy_missing_onnx_dlls(
    onnxruntime_dirs: list[Path], rust_package_dirs: list[Path]
) -> None:
    for rust_dir in rust_package_dirs:
        for source_dir in onnxruntime_dirs:
            for source in source_dir.glob("onnxruntime*.dll"):
                destination = rust_dir / source.name
                if destination.exists():
                    continue
                try:
                    shutil.copy2(source, destination)
                except OSError as error:
                    raise OSError(
                        f"Could not bundle ONNX Runtime DLL {source} into "
                        f"the Rust MCTS package directory {rust_dir}: {error}"
                    ) from error


(
    PROJECT_ROOT,
    LAUNCH_DIRECTORY,
    BUNDLE_DIRECTORY,
    _DLL_DIRS,
    _ONNXRUNTIME_DIRS,
    _RUST_PACKAGE_DIRS,
) = _runtime_paths()
_copy_missing_onnx_dlls(_ONNXRUNTIME_DIRS, _RUST_PACKAGE_DIRS)
_DLL_DIRECTORY_HANDLES: list[Any] = []
for _dll_dir in _DLL_DIRS:
    _dll_path = str(_dll_dir)
    if _dll_path not in os.environ.get("PATH", "").split(os.pathsep):
        os.environ["PATH"] = _dll_path + os.pathsep + os.environ.get("PATH", "")
    if sys.platform == "win32" and hasattr(os, "add_dll_directory"):
        try:
            _DLL_DIRECTORY_HANDLES.append(os.add_dll_directory(_dll_path))
        except OSError as error:
            raise RuntimeError(
                f"Could not register DLL search directory {_dll_dir}: {error}"
            ) from error

def _model_candidates(
    *,
    frozen: bool,
    bundle_dir: Path,
    executable_dir: Path,
    project_root: Path,
) -> tuple[Path, ...]:
    candidates: list[Path] = []
    candidates.append(MODEL_PATH)
    if frozen:
        internal_dir = executable_dir / "_internal"
        candidates.extend(
            (
                bundle_dir / "model_anchor.onnx",
                bundle_dir / "exports" / "model_anchor.onnx",
                internal_dir / "model_anchor.onnx",
                internal_dir / "exports" / "model_anchor.onnx",
            )
        )
    candidates.extend(
        (
            project_root / "model.onnx",
            project_root / "_internal" / "exports" / "model_anchor.onnx",
            project_root / "_internal" / "exports" / "model.onnx",
            executable_dir / "_internal" / "exports" / "model_anchor.onnx",
            executable_dir / "model_anchor.onnx",
            executable_dir / "latest_model.pt",
            project_root / "exports" / "model_anchor.onnx",
            project_root / "build" / "latest_model.pt",
        )
    )
    return tuple(dict.fromkeys(candidate.resolve() for candidate in candidates))


DEFAULT_MODEL_CANDIDATES = _model_candidates(
    frozen=bool(getattr(sys, "frozen", False)),
    bundle_dir=BUNDLE_DIRECTORY,
    executable_dir=LAUNCH_DIRECTORY,
    project_root=PROJECT_ROOT,
)

import chess
import torch

from src.mcts import MCTS
from src.persistence import load_archived_model


HARD_SEARCH_TIME_SECONDS = 5.0


def resolve_model_path(path: str | Path | None = None) -> Path:
    """Resolve the explicit model path or the ordered local/default candidates."""
    if path is not None:
        candidate = Path(path).expanduser()
        if not candidate.is_absolute():
            candidate = LAUNCH_DIRECTORY / candidate
        if not candidate.is_file():
            raise FileNotFoundError(f"Model file not found: {candidate}")
        return candidate.resolve()
    for candidate in DEFAULT_MODEL_CANDIDATES:
        if candidate.is_file():
            return candidate.resolve()
    searched = "\n".join(f"  - {candidate}" for candidate in DEFAULT_MODEL_CANDIDATES)
    raise FileNotFoundError(
        f"ONNX/checkpoint model not found. Expected {MODEL_PATH} or one of:\n{searched}"
    )


def parse_position_command(command: str) -> chess.Board:
    """Parse a UCI position command and replay its legal move history."""
    tokens = command.split()
    if not tokens or tokens[0] != "position":
        raise ValueError("Expected a UCI position command.")

    if len(tokens) >= 2 and tokens[1] == "startpos":
        board = chess.Board()
        moves_start = 2
    elif len(tokens) >= 8 and tokens[1] == "fen":
        try:
            moves_start = tokens.index("moves", 2)
        except ValueError:
            moves_start = len(tokens)
        fen_fields = tokens[2:moves_start]
        if len(fen_fields) != 6:
            raise ValueError("FEN position must contain exactly six fields.")
        try:
            board = chess.Board(" ".join(fen_fields))
        except ValueError as error:
            raise ValueError(f"Invalid FEN in position command: {error}") from error
    else:
        raise ValueError("Position must use 'startpos' or 'fen <six fields>'.")

    if moves_start < len(tokens):
        if tokens[moves_start] != "moves":
            raise ValueError("Unexpected data after UCI position.")
        for uci_move in tokens[moves_start + 1 :]:
            try:
                move = chess.Move.from_uci(uci_move)
            except ValueError as error:
                raise ValueError(f"Invalid UCI move {uci_move!r}.") from error
            if move not in board.legal_moves:
                raise ValueError(f"Illegal move {uci_move} in position command.")
            board.push(move)
    return board


def go_time_budget(command: str, side_to_move: chess.Color) -> float:
    """Return a conservative per-move search budget in seconds."""
    tokens = command.split()
    if not tokens or tokens[0] != "go":
        raise ValueError("Expected a UCI go command.")
    fields: dict[str, int] = {}
    index = 1
    while index < len(tokens):
        key = tokens[index]
        if key in {"wtime", "btime", "winc", "binc", "movetime", "movestogo"}:
            if index + 1 >= len(tokens):
                raise ValueError(f"Missing integer for UCI go field {key}.")
            try:
                fields[key] = max(0, int(tokens[index + 1]))
            except ValueError:
                raise ValueError(f"Invalid integer for UCI go field {key}.") from None
            index += 2
        else:
            index += 1

    if "movetime" in fields:
        return min(
            HARD_SEARCH_TIME_SECONDS,
            max(0.01, fields["movetime"] / 1000.0 * 0.9),
        )

    clock_key = "wtime" if side_to_move == chess.WHITE else "btime"
    increment_key = "winc" if side_to_move == chess.WHITE else "binc"
    if clock_key not in fields:
        return min(HARD_SEARCH_TIME_SECONDS, 1.0)
    moves_to_go = max(1, fields.get("movestogo", 30))
    clock_seconds = fields[clock_key] / 1000.0
    increment_seconds = fields.get(increment_key, 0) / 1000.0
    allocation = (clock_seconds / moves_to_go + increment_seconds * 0.8) * 0.8
    return min(
        HARD_SEARCH_TIME_SECONDS,
        max(0.01, min(clock_seconds * 0.8, allocation)),
    )


def _go_simulations(command: str, configured_simulations: int) -> int:
    fields = command.split()[1:]
    limits: list[int] = []
    for index, key in enumerate(fields[:-1]):
        if key in {"depth", "nodes"}:
            try:
                value = int(fields[index + 1])
            except ValueError:
                raise ValueError(f"Invalid integer for UCI go field {key}.") from None
            if value < 1:
                raise ValueError(f"UCI go field {key} must be positive.")
            limits.append(value)
    return min([configured_simulations, *limits]) if limits else configured_simulations


def _fallback_uci_move(board: chess.Board) -> str:
    legal_moves = list(board.legal_moves)
    return legal_moves[0].uci() if legal_moves else "0000"


class MartinUCI:
    def __init__(
        self,
        model_path: Path | None = None,
        device_option: str = "auto",
        simulations: int = 128,
    ) -> None:
        if simulations < 1:
            raise ValueError("MCTS simulations must be positive.")
        self.model_path = resolve_model_path(model_path)
        self.device_option = device_option
        self.simulations = simulations
        self.device = torch.device("cpu")
        self.model: torch.nn.Module | None = None
        self.search: MCTS | None = None
        self.board = chess.Board()
        self._output_lock = threading.Lock()
        self._search_thread: threading.Thread | None = None
        self._stop_requested = threading.Event()
        self._suppress_bestmove = threading.Event()

    def _emit(self, line: str) -> None:
        with self._output_lock:
            print(line, flush=True)

    def _load_model(self) -> None:
        if self.search is not None:
            return
        if not self.model_path.is_file():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        if self.model_path.suffix.lower() == ".onnx":
            self.search = MCTS(
                None,
                torch.device("cpu"),
                simulations=self.simulations,
                tactical_checks=False,
                rust_model_path=self.model_path,
            )
            self.device = torch.device("cpu")
            return
        if self.device_option == "auto":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(self.device_option)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(f"CUDA device {device} requested, but CUDA is unavailable.")
        model, _ = load_archived_model(self.model_path, device)
        self.model = model
        self.device = device
        self.search = MCTS(
            model,
            device,
            simulations=self.simulations,
            tactical_checks=False,
        )

    def _new_game(self) -> None:
        self.board = chess.Board()
        if self.search is not None:
            if self.model_path.suffix.lower() == ".onnx":
                self.search = MCTS(
                    None,
                    torch.device("cpu"),
                    simulations=self.simulations,
                    tactical_checks=False,
                    rust_model_path=self.model_path,
                )
            elif self.model is not None:
                self.search = MCTS(
                    self.model,
                    self.device,
                    simulations=self.simulations,
                    tactical_checks=False,
                )

    def _finish_search(self, command: str) -> None:
        position = self.board.copy(stack=True)
        legal_moves = list(position.legal_moves)
        fallback_move = legal_moves[0].uci() if legal_moves else "0000"
        try:
            if self.search is None:
                raise RuntimeError("Model is not loaded; send isready before go.")
            search = self.search
            time_budget = go_time_budget(command, self.board.turn)
            simulations = _go_simulations(command, self.simulations)
            self.search.simulations = simulations
            started = time.monotonic()
            search_result: dict[str, Any] = {}
            search_finished = threading.Event()
            visits: dict[Any, Any] = {}

            def run_search() -> None:
                try:
                    search_result["results"] = search.search_batch(
                        [position],
                        time_limit_seconds=time_budget,
                    )
                except Exception as error:
                    search_result["error"] = error
                finally:
                    search_finished.set()

            search_thread = threading.Thread(
                target=run_search,
                name="martin-mcts-search",
                daemon=True,
            )
            search_thread.start()
            if not search_finished.wait(timeout=time_budget):
                bestmove = fallback_move
                self._emit(
                    f"info string search exceeded {time_budget:.2f}s; "
                    "returning a legal fallback move"
                )
            else:
                if "error" in search_result:
                    raise search_result["error"]
                visits, _ = search_result["results"][0]
                if visits:
                    best_move = max(visits.items(), key=lambda entry: entry[1])[0]
                    candidate = best_move.uci()
                    bestmove = (
                        candidate
                        if any(candidate == move.uci() for move in legal_moves)
                        else fallback_move
                    )
                else:
                    bestmove = fallback_move
            elapsed_ms = max(1, int((time.monotonic() - started) * 1000))
            if visits:
                self._emit(f"info nodes {sum(visits.values())} time {elapsed_ms}")
            if not self._suppress_bestmove.is_set():
                self._emit(f"bestmove {bestmove}")
        except Exception as e:
            print(f"[ERROR] MCTS search failed: {e}", file=sys.stderr)
            print(traceback.format_exc(), file=sys.stderr)
            sys.stderr.flush()
            bestmove = fallback_move
            self._emit(f"info string error: {e}")
            if not self._suppress_bestmove.is_set():
                self._emit(f"bestmove {bestmove}")
        finally:
            self._search_thread = None

    def _start_search(self, command: str) -> None:
        self._stop_search(suppress_output=True)
        if self.search is None:
            self._emit("info string error: Model is not loaded; send isready first.")
            self._emit(f"bestmove {_fallback_uci_move(self.board)}")
            return
        self._stop_requested.clear()
        self._suppress_bestmove.clear()
        self._search_thread = threading.Thread(
            target=self._finish_search,
            args=(command,),
            name="martin-uci-search",
            daemon=True,
        )
        self._search_thread.start()

    def _stop_search(self, *, suppress_output: bool = False) -> None:
        thread = self._search_thread
        if thread is None:
            return
        if suppress_output:
            self._suppress_bestmove.set()
        self._stop_requested.set()
        thread.join()
        self._search_thread = None

    def run(self) -> None:
        for raw_command in sys.stdin:
            command = raw_command.strip()
            if not command:
                continue
            try:
                if command == "uci":
                    self._emit("id name MartinAI")
                    self._emit("id author Martin")
                    self._emit("uciok")
                elif command == "isready":
                    try:
                        self._load_model()
                    except Exception as error:
                        self._emit(f"info string error: {error}")
                    self._emit("readyok")
                elif command == "ucinewgame":
                    self._stop_search(suppress_output=True)
                    self._new_game()
                elif command.startswith("position "):
                    self._stop_search(suppress_output=True)
                    self.board = parse_position_command(command)
                elif command == "go" or command.startswith("go "):
                    self._start_search(command)
                elif command == "stop":
                    self._stop_search()
                elif command == "quit":
                    self._stop_search(suppress_output=True)
                    return
                elif command == "ponderhit":
                    continue
            except Exception as error:
                self._emit(f"info string error: {error}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MartinAI UCI chess engine")
    parser.add_argument(
        "--model-path",
        "--checkpoint",
        dest="model_path",
        type=Path,
        default=None,
        help=(
            "Model file; default priority: local model_anchor.onnx, "
            "local latest_model.pt, exports/model_anchor.onnx"
        ),
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Inference device for .pt models: auto, cpu, cuda, or cuda:N",
    )
    parser.add_argument(
        "--simulations",
        type=int,
        default=128,
        help="Maximum MCTS simulations per move",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        engine = MartinUCI(args.model_path, args.device, args.simulations)
        engine.run()
    except Exception as error:
        print(f"MartinAI startup failed: {error}", file=sys.stderr, flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
