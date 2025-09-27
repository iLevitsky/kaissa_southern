"""
Utilities to provide a stronger best‑move prediction for the Kaissa game.

This module implements an iterative‑deepening alpha–beta search with a
transposition table and simple move ordering.  The goal is to search
deeper than the existing minimax implementation in ``kaissa_engine``
while avoiding re‑computing identical positions.  It also includes a
slightly richer evaluation function that takes into account piece
placement and mobility, not just material.

The functions defined here do not modify the existing engine classes;
they operate on a ``KaissaGameEngine`` instance provided by the caller.

Example usage::

    from kaissa_engine.kaissa_engine import KaissaGameEngine
    from kaissa_ai.stronger_predictor import predict_best_move_strong

    engine = KaissaGameEngine()
    move = predict_best_move_strong(engine, max_depth=4, time_limit=5.0)
    print("Suggested move:", move)

This code is intended as a starting point; feel free to experiment with
the evaluation weights, transposition table size and search depth.
"""

from __future__ import annotations

import random
import time
from typing import Dict, Optional, Tuple, List

from kaissa_engine.kaissa_engine import (
    KaissaGameEngine,
    BOARD_ROWS,
    BOARD_COLS,
    NUM_SQUARES,
    get_piece_type,
    is_yellow,
)


# -----------------------------------------------------------------------------
# Zobrist hashing support
#
# A Zobrist hash assigns a random 64‑bit integer to each possible piece at
# each square.  The hash of a board is the XOR of all random values for the
# pieces currently on the board.  Using a transposition table keyed on this
# hash allows the search to recognise repeated positions.

_rng = random.Random(0xC0FFEE)

# Dimensions: [piece_type (0..10)][color_bit (0 or 1)][square_idx (0..99)]
_ZOBRIST_TABLE: List[List[List[int]]] = [
    [
        [_rng.getrandbits(64) for _ in range(NUM_SQUARES)] for _ in range(2)
    ]
    for _ in range(11)
]

def zobrist_hash(engine: KaissaGameEngine) -> int:
    """Compute a 64‑bit Zobrist hash of the current board state."""
    h = 0
    board = engine.board
    for idx in range(NUM_SQUARES):
        piece_val = board.squares[idx]
        if piece_val == 0:
            continue
        pt = get_piece_type(piece_val)
        col = 1 if is_yellow(piece_val) else 0
        h ^= _ZOBRIST_TABLE[pt][col][idx]
    return h

# -----------------------------------------------------------------------------
# Evaluation function
#
# A basic evaluation that combines material, piece‑square values and mobility.

# Base material values (same as kaissa_engine.evaluate)
_PIECE_VALUES = {
    0: 0,
    1: 900,   # Ubar
    2: 700,   # Ubara
    3: 500,   # Tarn
    4: 550,   # Builder
    5: 330,   # Initiate
    6: 420,   # Scribe
    7: 300,   # Assassin
    8: 150,   # Rider
    9: 50,    # Spearman
    10: 10000, # Homestone
}

# A simple piece‑square table.  For now all pieces share the same values;
# ideally you would tune these per piece type.  Central squares are more
# valuable.  The board is 10×10; indices are 0 (top/left) to 9 (bottom/right).
_PIECE_SQUARE_TABLE = [
    [0, 5, 5, 5, 5, 5, 5, 5, 5, 0],
    [5,10,10,10,10,10,10,10,10, 5],
    [5,10,15,15,15,15,15,15,10, 5],
    [5,10,15,20,20,20,20,15,10, 5],
    [5,10,15,20,25,25,20,15,10, 5],
    [5,10,15,20,25,25,20,15,10, 5],
    [5,10,15,20,20,20,20,15,10, 5],
    [5,10,15,15,15,15,15,15,10, 5],
    [5,10,10,10,10,10,10,10,10, 5],
    [0, 5, 5, 5, 5, 5, 5, 5, 5, 0],
]

def evaluate_position(engine: KaissaGameEngine) -> int:
    """Return a heuristic evaluation of the current position.

    Positive values favour Yellow; negative values favour Red.
    This evaluation adds a simple piece‑square bonus and mobility term to
    the base material counts.
    """
    material = 0
    mobility = 0
    pst_bonus = 0
    board = engine.board
    for idx in range(NUM_SQUARES):
        piece_val = board.squares[idx]
        if piece_val == 0:
            continue
        pt = get_piece_type(piece_val)
        col = 1 if is_yellow(piece_val) else -1
        # material
        material += col * _PIECE_VALUES.get(pt, 0)
        # piece‑square bonus: convert idx to row/col
        r = idx // BOARD_COLS
        c = idx % BOARD_COLS
        pst_bonus += col * _PIECE_SQUARE_TABLE[r][c]
    # mobility: difference in legal move counts
    # We temporarily compute mobility by switching turn
    is_yellow_turn = engine.is_yellow_turn
    engine.is_yellow_turn = True
    yellow_moves = len(engine.get_legal_moves())
    engine.is_yellow_turn = False
    red_moves = len(engine.get_legal_moves())
    engine.is_yellow_turn = is_yellow_turn
    mobility = (yellow_moves - red_moves) * 10
    return material + pst_bonus + mobility

# -----------------------------------------------------------------------------
# Transposition table entry

class TTEntry:
    __slots__ = ("depth", "value", "best_move")
    def __init__(self, depth: int, value: int, best_move):
        self.depth = depth
        self.value = value
        self.best_move = best_move


def _minimax_search(
    engine: KaissaGameEngine,
    depth: int,
    alpha: int,
    beta: int,
    maximizing: bool,
    tt: Dict[int, TTEntry],
    start_time: float,
    time_limit: Optional[float],
) -> Tuple[int, Optional[tuple]]:
    """Internal recursive minimax search with alpha‑beta pruning and transposition.

    Returns a tuple (score, best_move).  Uses a transposition table ``tt``
    keyed by Zobrist hash.  If ``time_limit`` is provided, the search
    aborts early and returns an evaluation if the limit is exceeded.
    """
    # Check time limit
    if time_limit is not None and (time.time() - start_time) > time_limit:
        return evaluate_position(engine), None

    # Check transposition table
    zhash = zobrist_hash(engine)
    tt_entry = tt.get(zhash)
    if tt_entry is not None and tt_entry.depth >= depth:
        return tt_entry.value, tt_entry.best_move

    # Terminal or depth limit
    if depth == 0 or engine.is_game_over():
        return evaluate_position(engine), None

    legal_moves = engine.get_legal_moves()
    if not legal_moves:
        return evaluate_position(engine), None

    # Move ordering: prefer captures first by checking if the end square is occupied
    # We'll approximate by sorting moves where the destination contains a piece.
    state = engine.get_state()
    def move_key(mv):
        if len(mv) >= 2:
            (_, end) = mv[:2]
            er, ec = end
            if 0 <= er < BOARD_ROWS and 0 <= ec < BOARD_COLS and state[er][ec] is not None:
                return 0  # capture moves first
        return 1
    legal_moves.sort(key=move_key)

    best_value = -10**12 if maximizing else 10**12
    best_move = None

    for mv in legal_moves:
        # clone engine and apply move
        new_engine = engine.clone()
        new_engine.apply_move(mv)
        # recurse
        val, _ = _minimax_search(
            new_engine,
            depth - 1,
            alpha,
            beta,
            not maximizing,
            tt,
            start_time,
            time_limit,
        )
        if maximizing:
            if val > best_value:
                best_value, best_move = val, mv
            alpha = max(alpha, val)
        else:
            if val < best_value:
                best_value, best_move = val, mv
            beta = min(beta, val)
        if beta <= alpha:
            break
    # Store in transposition table
    tt[zhash] = TTEntry(depth, best_value, best_move)
    return best_value, best_move


def predict_best_move_strong(
    engine: KaissaGameEngine,
    max_depth: int = 4,
    time_limit: Optional[float] = None,
) -> Optional[tuple]:
    """Compute a stronger best move using iterative deepening search.

    :param engine: game state to evaluate (will not be modified).
    :param max_depth: maximum search depth in plies.  Depth 4 or 5 is a
      reasonable starting point; higher values may be very slow.
    :param time_limit: optional time limit in seconds.  If provided, the
      search will abort when the time budget is exceeded and return the
      best move found so far.
    :return: a move tuple compatible with ``apply_move``, or ``None`` if no
      legal move is available.
    """
    best_move_overall: Optional[tuple] = None
    tt: Dict[int, TTEntry] = {}
    start_time = time.time()
    # Iterative deepening: increase depth gradually, reusing transposition table
    for depth in range(1, max_depth + 1):
        val, move = _minimax_search(
            engine,
            depth,
            -10**12,
            10**12,
            maximizing=engine.is_yellow_turn,
            tt=tt,
            start_time=start_time,
            time_limit=time_limit,
        )
        if move is not None:
            best_move_overall = move
        # If time limit exceeded, break
        if time_limit is not None and (time.time() - start_time) > time_limit:
            break
    return best_move_overall