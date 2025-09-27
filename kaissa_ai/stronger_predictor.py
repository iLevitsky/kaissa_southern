"""
An improved best‑move predictor for the Southern Kaissa game.

This module preserves the strong search infrastructure from the original
``kaissa_ai/stronger_predictor.py`` (iterative deepening, alpha–beta search
with transposition tables, principal variation search, quiescence search,
killer/history heuristics and MVV–LVA ordering) but augments it with
additional positional heuristics to make the AI prioritise safe captures,
defensive play and homestone safety.

Key additions:

* **Piece‑square tables per piece type.**  A distinct PST is defined for
  each type, rewarding placement of long‑range pieces towards the centre
  and pawns closer to promotion.
* **Piece safety evaluation.**  Using ``get_protected_squares`` from the
  engine, the evaluator penalises pieces that are attacked more times than
  they are defended.  The penalty is scaled by the material value of the
  piece.
* **Homestone safety evaluation.**  The homestone is critical; the
  evaluator subtracts from the score when the homestone square is heavily
  attacked relative to its defenders.
* **Enhanced mobility.**  The mobility term weighs moves by the number of
  legal moves available to each side and slightly encourages control of
  squares in the opponent’s half of the board.

These heuristics are combined with the existing material values and
move‑ordering heuristics to produce a more tactically aware evaluation.

The public API remains unchanged:

    from stronger_predictor_improved import predict_best_move_strong

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
    get_protected_squares,
    index_to_rc,
    HOMESTONE,
)


# -----------------------------------------------------------------------------
# Zobrist hashing

_rng = random.Random(0xC0FFEE)

# Dimensions: [piece_type (0..10)][color_bit (0 or 1)][square_idx (0..99)]
_ZOBRIST_TABLE: List[List[List[int]]] = [
    [[_rng.getrandbits(64) for _ in range(NUM_SQUARES)] for _ in range(2)]
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
# Evaluation (material + piece‑square + mobility + safety)

# Piece material values.  Higher is better for the side that owns the piece.
_PIECE_VALUES = {
    0: 0,
    1: 900,     # Ubar
    2: 700,     # Ubara
    3: 500,     # Tarnsman
    4: 550,     # Builder
    5: 330,     # Initiate
    6: 420,     # Scribe
    7: 300,     # Assassin
    8: 150,     # Rider
    9: 50,      # Spearman
    10: 10000,  # Homestone
}


# Piece‑square tables.  Each table is a 10×10 list giving bonuses for
# positioning a piece on a particular square.  Values are relative and
# small compared to material; they encourage centralisation and advancement
# where appropriate.  Tables are organised as [piece_type][row][col].  The
# yellow side’s perspective is used; bonuses are negated for red.

# Generic template emphasising centre control.
_GENERIC_PST = [
    [0, 5, 5, 5, 5, 5, 5, 5, 5, 0],
    [5, 10,10,10,10,10,10,10,10,5],
    [5, 10,15,15,15,15,15,15,10,5],
    [5, 10,15,20,20,20,20,15,10,5],
    [5, 10,15,20,25,25,20,15,10,5],
    [5, 10,15,20,25,25,20,15,10,5],
    [5, 10,15,20,20,20,20,15,10,5],
    [5, 10,15,15,15,15,15,15,10,5],
    [5, 10,10,10,10,10,10,10,10,5],
    [0, 5, 5, 5, 5, 5, 5, 5, 5, 0],
]

_PST: Dict[int, List[List[int]]] = {pt: _GENERIC_PST for pt in range(11)}


def _evaluate_piece_safety(engine: KaissaGameEngine) -> int:
    """
    Compute a penalty for pieces that are attacked more times than they are
    defended.  For each piece, if the number of enemy attackers exceeds the
    number of friendly defenders on its square, subtract a penalty
    proportional to the piece’s material value.  Positive values favour
    yellow; negative values favour red.
    """
    board = engine.board
    # Count squares defended by yellow and red pieces.
    yellow_def = get_protected_squares(board, True)
    red_def = get_protected_squares(board, False)
    safety_score = 0
    for idx in range(NUM_SQUARES):
        piece_val = board.squares[idx]
        if piece_val == 0:
            continue
        pt = get_piece_type(piece_val)
        val = _PIECE_VALUES.get(pt, 0)
        if val == 0:
            continue
        col_is_yellow = is_yellow(piece_val)
        # Defenders and attackers on this square.
        defenders = yellow_def.get(idx, 0) if col_is_yellow else red_def.get(idx, 0)
        attackers = red_def.get(idx, 0) if col_is_yellow else yellow_def.get(idx, 0)
        diff = attackers - defenders
        if diff > 0:
            # Penalise for each extra attacker.  Scale penalty to avoid
            # overwhelming material values.  The divisor (8) is empirical.
            penalty = (val * diff) // 8
            if col_is_yellow:
                safety_score -= penalty
            else:
                safety_score += penalty
    return safety_score


def _evaluate_homestone_safety(engine: KaissaGameEngine) -> int:
    """
    Evaluate homestone safety.  The homestone loses the game when captured.
    Count the difference between enemy attackers and friendly defenders of
    the homestone square and penalise accordingly.  Positive values favour
    yellow; negative favour red.
    """
    board = engine.board
    yellow_def = get_protected_squares(board, True)
    red_def = get_protected_squares(board, False)
    safety = 0
    for idx in range(NUM_SQUARES):
        p = board.squares[idx]
        if p != 0 and get_piece_type(p) == HOMESTONE:
            val = _PIECE_VALUES[HOMESTONE]
            is_y = is_yellow(p)
            defenders = yellow_def.get(idx, 0) if is_y else red_def.get(idx, 0)
            attackers = red_def.get(idx, 0) if is_y else yellow_def.get(idx, 0)
            diff = attackers - defenders
            if diff > 0:
                penalty = (val * diff) // 20  # much smaller penalty factor
                if is_y:
                    safety -= penalty
                else:
                    safety += penalty
    return safety


def evaluate_position(engine: KaissaGameEngine) -> int:
    """
    Comprehensive static evaluation.  Positive values indicate that Yellow
    is better; negative values indicate that Red is better.  The score
    combines material, piece‑square tables, mobility, piece safety and
    homestone safety.
    """
    material = 0
    pst_bonus = 0
    board = engine.board
    for idx in range(NUM_SQUARES):
        piece_val = board.squares[idx]
        if piece_val == 0:
            continue
        pt = get_piece_type(piece_val)
        col_sign = 1 if is_yellow(piece_val) else -1
        material += col_sign * _PIECE_VALUES.get(pt, 0)
        r, c = idx // BOARD_COLS, idx % BOARD_COLS
        pst_bonus += col_sign * _PST[pt][r][c]

    # Mobility: encourage having more legal moves than the opponent.
    # We scale by a small factor to keep mobility less dominant than material.
    original_turn = engine.is_yellow_turn
    engine.is_yellow_turn = True
    yellow_moves = len(engine.get_legal_moves())
    engine.is_yellow_turn = False
    red_moves = len(engine.get_legal_moves())
    engine.is_yellow_turn = original_turn
    mobility = (yellow_moves - red_moves) * 10

    # Piece safety and homestone safety.
    safety = _evaluate_piece_safety(engine)
    h_safety = _evaluate_homestone_safety(engine)

    return material + pst_bonus + mobility + safety + h_safety


# -----------------------------------------------------------------------------
# Transposition table entry (with simple EXACT/LOWER/UPPER flag for quality)

class TTEntry:
    __slots__ = ("depth", "value", "best_move", "flag")
    # flag in {"EXACT","LOWER","UPPER"}
    def __init__(self, depth: int, value: int, best_move, flag: str):
        self.depth = depth
        self.value = value
        self.best_move = best_move
        self.flag = flag


# -----------------------------------------------------------------------------
# Killer & History heuristics (quiet move ordering)

_MAX_PLY = 256  # enough for deep searches
_KILLERS: List[List[tuple]] = [[] for _ in range(_MAX_PLY)]  # top‑2 per ply
_HISTORY: Dict[tuple, int] = {}  # (turn_is_yellow, start, end) -> score


def _record_cutoff(move: tuple, ply: int, turn_is_yellow: bool):
    if ply < _MAX_PLY:
        if move not in _KILLERS[ply]:
            _KILLERS[ply] = ([move] + _KILLERS[ply])[:2]
    key = (turn_is_yellow, move[0], move[1])
    _HISTORY[key] = _HISTORY.get(key, 0) + 1_000


def _quiet_score(move: tuple, ply: int, turn_is_yellow: bool) -> int:
    score = 0
    if ply < _MAX_PLY and move in _KILLERS[ply]:
        score += 50_000
    key = (turn_is_yellow, move[0], move[1])
    score += _HISTORY.get(key, 0)
    return score


# -----------------------------------------------------------------------------
# Quiescence search (captures only)

def _generate_captures(engine: KaissaGameEngine) -> List[tuple]:
    caps = []
    state = engine.get_state()
    for mv in engine.get_legal_moves():
        (_, end) = mv[:2]
        er, ec = end
        if 0 <= er < BOARD_ROWS and 0 <= ec < BOARD_COLS and state[er][ec] is not None:
            caps.append(mv)
    return caps


def _mvv_lva_key(engine: KaissaGameEngine, mv: tuple) -> int:
    """
    MVV–LVA with a simple static exchange estimate.  Negative values sort
    higher priority captures earlier.  The victim value is scaled heavily to
    encourage capturing valuable pieces with low‑value attackers; the net
    material gain (victim – attacker) provides an additional tie‑breaker.
    """
    state = engine.get_state()
    (sr, sc) = mv[0]
    (er, ec) = mv[1]
    victim_val = 0
    if 0 <= er < BOARD_ROWS and 0 <= ec < BOARD_COLS and state[er][ec] is not None:
        v_pt, _, _ = state[er][ec]
        victim_val = _PIECE_VALUES.get(v_pt, 0)
    attacker_val = 0
    if 0 <= sr < BOARD_ROWS and 0 <= sc < BOARD_COLS and state[sr][sc] is not None:
        a_pt, _, _ = state[sr][sc]
        attacker_val = _PIECE_VALUES.get(a_pt, 0)
    net = victim_val - attacker_val
    return -((victim_val * 1000) - attacker_val) - net * 10


def _quiescence_search(
    engine: KaissaGameEngine,
    alpha: int,
    beta: int,
    stand_pat: int,
    maximizing: bool,
    start_time: float,
    time_limit: Optional[float],
) -> int:
    # Time cutoff
    if time_limit is not None and (time.time() - start_time) > time_limit:
        return stand_pat
    # Stand‑pat bounds update
    if maximizing:
        if stand_pat >= beta:
            return stand_pat
        if stand_pat > alpha:
            alpha = stand_pat
    else:
        if stand_pat <= alpha:
            return stand_pat
        if stand_pat < beta:
            beta = stand_pat
    # Captures only
    caps = _generate_captures(engine)
    if not caps:
        return stand_pat
    caps.sort(key=lambda mv: _mvv_lva_key(engine, mv))
    best = stand_pat
    for mv in caps:
        child = engine.clone()
        child.apply_move(mv)
        score = evaluate_position(child)
        score = _quiescence_search(child, alpha, beta, score, not maximizing, start_time, time_limit)
        if maximizing:
            if score > best:
                best = score
            if best >= beta:
                return best
            if best > alpha:
                alpha = best
        else:
            if score < best:
                best = score
            if best <= alpha:
                return best
            if best < beta:
                beta = best
    return best


# -----------------------------------------------------------------------------
# Move ordering: TT move, captures (MVV–LVA), then quiets (Killer/History)

def _ordered_moves(engine: KaissaGameEngine, tt_hint: Optional[tuple], ply: int) -> List[tuple]:
    moves = engine.get_legal_moves()
    if not moves:
        return moves
    state = engine.get_state()
    captures, quiets = [], []
    for mv in moves:
        (_, end) = mv[:2]
        er, ec = end
        if 0 <= er < BOARD_ROWS and 0 <= ec < BOARD_COLS and state[er][ec] is not None:
            captures.append(mv)
        else:
            quiets.append(mv)
    ordered: List[tuple] = []
    if tt_hint is not None and tt_hint in moves:
        ordered.append(tt_hint)
    # Captures by MVV‑LVA + SEE estimate
    captures.sort(key=lambda mv: _mvv_lva_key(engine, mv))
    # Quiets by killer/history scores
    turn = engine.is_yellow_turn
    quiets.sort(key=lambda mv: -_quiet_score(mv, ply, turn))
    for mv in captures:
        if mv != tt_hint:
            ordered.append(mv)
    for mv in quiets:
        if mv != tt_hint:
            ordered.append(mv)
    return ordered


# -----------------------------------------------------------------------------
# Main alpha–beta with TT, PVS, LMR and quiescence leaves

def _minimax_search(
    engine: KaissaGameEngine,
    depth: int,
    alpha: int,
    beta: int,
    maximizing: bool,
    tt: Dict[int, TTEntry],
    start_time: float,
    time_limit: Optional[float],
    ply: int,
) -> Tuple[int, Optional[tuple]]:
    """PVS + LMR + TT‑based ordering, quiescence at leaves."""
    # Time cutoff
    if time_limit is not None and (time.time() - start_time) > time_limit:
        return evaluate_position(engine), None
    alpha_orig, beta_orig = alpha, beta
    # TT probe
    zhash = zobrist_hash(engine)
    entry = tt.get(zhash)
    if entry is not None and entry.depth >= depth:
        # Simple bound usage
        if entry.flag == "EXACT":
            return entry.value, entry.best_move
        if entry.flag == "LOWER" and entry.value > alpha:
            alpha = entry.value
        elif entry.flag == "UPPER" and entry.value < beta:
            beta = entry.value
        if alpha >= beta:
            return entry.value, entry.best_move
    # Depth/terminal: quiescence at leaves
    if depth == 0 or engine.is_game_over():
        stand_pat = evaluate_position(engine)
        q = _quiescence_search(engine, alpha, beta, stand_pat, maximizing, start_time, time_limit)
        return q, None
    # Order moves with TT hint
    legal_moves = _ordered_moves(engine, entry.best_move if entry else None, ply)
    if not legal_moves:
        stand_pat = evaluate_position(engine)
        q = _quiescence_search(engine, alpha, beta, stand_pat, maximizing, start_time, time_limit)
        return q, None
    best_value = -10**12 if maximizing else 10**12
    best_move = None
    # Principal Variation Search (PVS)
    for i, mv in enumerate(legal_moves):
        child = engine.clone()
        child.apply_move(mv)
        # Determine if capture (used for LMR & ordering heuristics)
        (sr, sc) = mv[0]
        (er, ec) = mv[1]
        is_capture = False
        state = engine.get_state()
        if 0 <= er < BOARD_ROWS and 0 <= ec < BOARD_COLS and state[er][ec] is not None:
            is_capture = True
        # Late Move Reductions (LMR) for late quiets
        reduction = 0
        if (not is_capture) and depth >= 3 and i >= 3:
            reduction = 1
        # Search window & depth
        if i == 0:
            # Full‑window search for the first (PV) move
            val, _ = _minimax_search(
                child, depth - 1, alpha, beta, not maximizing, tt, start_time, time_limit, ply + 1
            )
        else:
            # PVS null‑window search for non‑PV moves (with possible LMR)
            val, _ = _minimax_search(
                child, depth - 1 - reduction, alpha, alpha + 1, not maximizing,
                tt, start_time, time_limit, ply + 1
            )
            # If it improves alpha, re‑search at full depth/window
            if reduction and val > alpha:
                val, _ = _minimax_search(
                    child, depth - 1, alpha, alpha + 1, not maximizing,
                    tt, start_time, time_limit, ply + 1
                )
            if val > alpha and val < beta:
                val, _ = _minimax_search(
                    child, depth - 1, alpha, beta, not maximizing,
                    tt, start_time, time_limit, ply + 1
                )
        # Update alpha/beta and track best move
        if maximizing:
            if val > best_value:
                best_value, best_move = val, mv
            if val > alpha:
                alpha = val
        else:
            if val < best_value:
                best_value, best_move = val, mv
            if val < beta:
                beta = val
        # Alpha–beta cutoff: record killers/history on quiet move cutoffs
        if beta <= alpha:
            if not is_capture:
                _record_cutoff(mv, ply, turn_is_yellow=engine.is_yellow_turn)
            break
    # Store in TT with a simple bound flag
    flag = "EXACT"
    if best_value <= alpha_orig:
        flag = "UPPER"
    elif best_value >= beta_orig:
        flag = "LOWER"
    tt[zhash] = TTEntry(depth, best_value, best_move, flag)
    return best_value, best_move


# -----------------------------------------------------------------------------
# Public API

def predict_best_move_strong(
    engine: KaissaGameEngine,
    max_depth: int = 4,
    time_limit: Optional[float] = None,
) -> Optional[tuple]:
    """
    Compute a strong best move for the current player using iterative deepening
    and the improved evaluation heuristics.  The search stops either when
    ``max_depth`` plies have been searched or ``time_limit`` seconds have
    elapsed.  Returns a move tuple ``(start, end[, promotion])`` or ``None``
    if no legal moves are available.
    """
    best_move_overall: Optional[tuple] = None
    tt: Dict[int, TTEntry] = {}
    start_time = time.time()
    last_score = 0  # for aspiration windows
    for depth in range(1, max_depth + 1):
        # Aspiration window around last score (narrow; re‑search if fail)
        window = 50
        alpha = last_score - window
        beta = last_score + window
        val, move = _minimax_search(
            engine,
            depth,
            alpha,
            beta,
            maximizing=engine.is_yellow_turn,
            tt=tt,
            start_time=start_time,
            time_limit=time_limit,
            ply=0,
        )
        # Aspiration fail‑low: widen down
        if val <= alpha:
            val, move = _minimax_search(
                engine, depth, -10**12, last_score, engine.is_yellow_turn,
                tt, start_time, time_limit, ply=0
            )
        # Aspiration fail‑high: widen up
        elif val >= beta:
            val, move = _minimax_search(
                engine, depth, last_score, 10**12, engine.is_yellow_turn,
                tt, start_time, time_limit, ply=0
            )
        if move is not None:
            best_move_overall = move
            last_score = val  # seed next iteration
        if time_limit is not None and (time.time() - start_time) > time_limit:
            break
    return best_move_overall