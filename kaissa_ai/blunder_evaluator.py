# kaissa_ai/blunder_evaluator.py
from __future__ import annotations
from typing import Dict, List, Set, Tuple

from kaissa_engine.kaissa_engine import KaissaGameEngine, BOARD_ROWS, BOARD_COLS

Move = tuple                    # ((sr,sc),(er,ec)) or ((sr,sc),(er,ec),promo)
Square = Tuple[int, int]
Arrow = Tuple[Square, Square]   # (from_enemy_square, to_our_square)

def _immediate_attackers_after_move(child: KaissaGameEngine) -> List[Arrow]:
    """
    On position 'child' (already after OUR move), list opponent captures in ONE move:
      returns [(enemy_from_sq, our_victim_sq), ...]
    """
    arrows: List[Arrow] = []
    st = child.get_state()
    opp_is_yellow = child.is_yellow_turn  # opponent to move

    for mv in child.get_legal_moves():
        start, end = mv[:2]
        er, ec = end
        if 0 <= er < BOARD_ROWS and 0 <= ec < BOARD_COLS:
            occ = st[er][ec]
            if occ is not None:
                _ptype, is_yellow, _moved = occ
                # If occupant is OUR color (i.e., not opponent's), then this move captures us
                if is_yellow != opp_is_yellow:
                    arrows.append((start, end))
    return arrows


def evaluate_immediate_blunders_for_moves(
    engine: KaissaGameEngine,
    candidate_moves: List[Move],
) -> Dict[Move, List[Arrow]]:
    """
    For each candidate move, return the list of immediate opponent capture arrows:
      { move -> [(enemy_from, our_victim), ...] }
    """
    results: Dict[Move, List[Arrow]] = {}
    for mv in candidate_moves:
        child = engine.clone()
        child.apply_move(mv)
        if child.is_game_over():
            results[mv] = []
            continue
        results[mv] = _immediate_attackers_after_move(child)
    return results


def classify_destinations_for_selected_piece(
    engine: KaissaGameEngine,
    selected_moves: List[Move],
):
    """
    For the SELECTED piece's legal moves, compute:
      - red_dests: set[(r,c)]   -> destination squares where the moved piece dies immediately
      - orange_dests: set[(r,c)]-> destination squares where the moved piece survives but some other piece dies
      - red_arrows: set[((fr,fc),(tr,tc))]    -> enemy arrows that capture the moved piece (destination)
      - orange_arrows: set[((fr,fc),(tr,tc))] -> enemy arrows that capture other pieces

    Arrows are shown as a de-duplicated union across all candidate moves.
    """
    per_move_attacks = evaluate_immediate_blunders_for_moves(engine, selected_moves)

    red_dests: Set[Square] = set()
    orange_dests: Set[Square] = set()
    red_arrows: Set[Arrow] = set()
    orange_arrows: Set[Arrow] = set()

    for mv, arrows in per_move_attacks.items():
        if not arrows:
            continue
        end_sq = mv[1]
        # Track whether moved piece dies for this move
        moved_piece_dies = False
        other_piece_dies = False

        for enemy_from, our_victim in arrows:
            if our_victim == end_sq:
                moved_piece_dies = True
                red_arrows.add((enemy_from, our_victim))
            else:
                other_piece_dies = True
                orange_arrows.add((enemy_from, our_victim))

        if moved_piece_dies:
            red_dests.add(end_sq)
        elif other_piece_dies:
            orange_dests.add(end_sq)

    return {
        "red_dests": red_dests,
        "orange_dests": orange_dests,
        "red_arrows": red_arrows,
        "orange_arrows": orange_arrows,
    }
