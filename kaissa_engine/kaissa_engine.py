# kaissa_engine/kaissa_engine.py

import copy
from .piece_constants import (
    UBAR, UBARA, TARNSMAN, BUILDER, INITIATE,
    SCRIBE, ASSASSIN, RIDER, SPEARMAN, HOMESTONE
)

BOARD_ROWS = 10
BOARD_COLS = 10

class Piece:
    """Basic piece structure: type, color, 'has_moved' for Spearmen, etc."""
    def __init__(self, piece_type: int, is_yellow: bool):
        self.piece_type = piece_type
        self.is_yellow = is_yellow
        self.has_moved = False

    def __repr__(self):
        c = "Y" if self.is_yellow else "R"
        return f"Piece({self.piece_type}, {c})"

class Board:
    """
    10x10 array: self.grid[r][c] = Piece or None.
    """
    def __init__(self):
        self.grid = [[None for _ in range(BOARD_COLS)] for _ in range(BOARD_ROWS)]
        self.setup_initial_position()

    def setup_initial_position(self):
        """
        Standard layout:
         - Yellow on rows 9 & 8
         - Red on rows 0 & 1
        """
        yellow_back =  [INITIATE, BUILDER, SCRIBE, TARNSMAN, UBAR, UBARA, TARNSMAN, SCRIBE, BUILDER, INITIATE]
        yellow_front = [ASSASSIN, RIDER, SPEARMAN, SPEARMAN, SPEARMAN, SPEARMAN, SPEARMAN, SPEARMAN, RIDER, ASSASSIN]
        red_back    =  [INITIATE, BUILDER, SCRIBE, TARNSMAN, UBAR, UBARA, TARNSMAN, SCRIBE, BUILDER, INITIATE]
        red_front   =  [ASSASSIN, RIDER, SPEARMAN, SPEARMAN, SPEARMAN, SPEARMAN, SPEARMAN, SPEARMAN, RIDER, ASSASSIN]

        # Place Yellow (rows 9 & 8)
        for c in range(BOARD_COLS):
            self.grid[9][c] = Piece(yellow_back[c], True)
            self.grid[8][c] = Piece(yellow_front[c], True)

        # Place Red (rows 0 & 1)
        for c in range(BOARD_COLS):
            self.grid[0][c] = Piece(red_back[c], False)
            self.grid[1][c] = Piece(red_front[c], False)

    def clone(self):
        new_board = Board.__new__(Board)
        new_board.grid = [[None for _ in range(BOARD_COLS)] for _ in range(BOARD_ROWS)]
        for r in range(BOARD_ROWS):
            for c in range(BOARD_COLS):
                p = self.grid[r][c]
                if p:
                    cp = Piece(p.piece_type, p.is_yellow)
                    cp.has_moved = p.has_moved
                    new_board.grid[r][c] = cp
        return new_board

    def in_bounds(self, r: int, c: int) -> bool:
        return (0 <= r < BOARD_ROWS) and (0 <= c < BOARD_COLS)

    def get_piece(self, r, c):
        if not self.in_bounds(r, c):
            return None
        return self.grid[r][c]

    def set_piece(self, r, c, piece):
        self.grid[r][c] = piece

    def move_piece(self, start, end):
        sr, sc = start
        er, ec = end
        piece = self.get_piece(sr, sc)
        if not piece:
            return
        self.grid[er][ec] = piece
        self.grid[sr][sc] = None
        piece.has_moved = True


def king_dirs():
    return [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]

def rook_dirs():
    return [(1,0),(-1,0),(0,1),(0,-1)]

def bishop_dirs():
    return [(1,1),(1,-1),(-1,1),(-1,-1)]

def queen_dirs():
    return rook_dirs() + bishop_dirs()


def gen_sliding(board, r, c, max_dist, directions):
    """Used by rooks, bishops, queens, or short-range variants."""
    moves = []
    p = board.get_piece(r, c)
    if not p:
        return moves
    color = p.is_yellow

    for (dr, dc) in directions:
        steps = 0
        nr, nc = r, c
        while True:
            nr += dr
            nc += dc
            steps += 1
            if not board.in_bounds(nr, nc):
                break
            occupant = board.get_piece(nr, nc)
            if occupant:
                # capture if enemy
                if occupant.is_yellow != color:
                    moves.append((nr, nc))
                break
            else:
                moves.append((nr, nc))
            if (max_dist is not None) and (steps >= max_dist):
                break
    return moves

def gen_assassin(board, r, c):
    """Can move 1 or 2 squares in any direction, can't jump if 2 squares."""
    moves = []
    p = board.get_piece(r, c)
    if not p:
        return moves
    color = p.is_yellow
    for (dr, dc) in queen_dirs():
        for dist in [1, 2]:
            nr = r + dr*dist
            nc = c + dc*dist
            if dist == 2:
                # check the intermediate
                mid_r = r + dr
                mid_c = c + dc
                if board.in_bounds(mid_r, mid_c):
                    if board.get_piece(mid_r, mid_c) is not None:
                        continue
            if board.in_bounds(nr, nc):
                occupant = board.get_piece(nr, nc)
                if occupant is None or occupant.is_yellow != color:
                    moves.append((nr, nc))
    return moves

def gen_rider(board, r, c):
    """Moves exactly 1 square in any direction (like a King, but no special rules)."""
    moves = []
    p = board.get_piece(r, c)
    if not p:
        return moves
    color = p.is_yellow
    for (dr, dc) in king_dirs():
        nr = r + dr
        nc = c + dc
        if board.in_bounds(nr, nc):
            occupant = board.get_piece(nr, nc)
            if occupant is None or occupant.is_yellow != color:
                moves.append((nr, nc))
    return moves

def gen_tarnsman(board, r, c):
    """Knight-like leaps (3+2) plus 1-square shifts with no capture for shift."""
    moves = []
    p = board.get_piece(r, c)
    if not p:
        return moves
    color = p.is_yellow

    # Knight leaps
    offsets = [(3,2),(3,-2),(-3,2),(-3,-2),(2,3),(2,-3),(-2,3),(-2,-3)]
    for (dr, dc) in offsets:
        nr = r + dr
        nc = c + dc
        if board.in_bounds(nr, nc):
            occupant = board.get_piece(nr, nc)
            if occupant is None or occupant.is_yellow != color:
                moves.append((nr, nc))

    # 1-square shift (cannot capture)
    for (dr, dc) in king_dirs():
        nr, nc = r + dr, c + dc
        if board.in_bounds(nr, nc):
            occupant = board.get_piece(nr, nc)
            if occupant is None:
                moves.append((nr, nc))

    return moves

def gen_homestone(board, r, c):
    """Home Stone can move 1 square in any direction if empty. No captures."""
    moves = []
    for (dr, dc) in king_dirs():
        nr, nc = r+dr, c+dc
        if board.in_bounds(nr, nc):
            if board.get_piece(nr, nc) is None:
                moves.append((nr, nc))
    return moves

def gen_spearman(board, r, c):
    """
    Spearman: forward 1..3 if not moved, else 1. Side or diag forward if empty, diag forward capture.
    Promotion if last rank => can remain or become Rider or Tarnsman.
    We'll store promotions as 3-tuple: (er, ec, promotion_type).
    """
    moves = []
    p = board.get_piece(r, c)
    if not p:
        return moves
    color = p.is_yellow
    direction = -1 if color else 1
    first_cap = 3 if not p.has_moved else 1

    last_rank = 0 if color else 9

    def add(er, ec):
        if er == last_rank:
            # allow remain or promote
            moves.append((er, ec, None))
            moves.append((er, ec, RIDER))
            moves.append((er, ec, TARNSMAN))
        else:
            moves.append((er, ec))

    # forward
    for step in range(1, first_cap + 1):
        nr = r + step*direction
        if not board.in_bounds(nr, c):
            break
        occ = board.get_piece(nr, c)
        if occ is None:
            add(nr, c)
        else:
            break

    # sideways & diag forward (empty only)
    for dc in [-1, 1]:
        # sideways
        nr, nc = r, c+dc
        if board.in_bounds(nr, nc):
            if board.get_piece(nr, nc) is None:
                add(nr, nc)
        # diag forward
        nr, nc = r+direction, c+dc
        if board.in_bounds(nr, nc):
            if board.get_piece(nr, nc) is None:
                add(nr, nc)

    # capturing diagonally forward
    for dc in [-1, 1]:
        nr, nc = r+direction, c+dc
        if board.in_bounds(nr, nc):
            occ = board.get_piece(nr, nc)
            if occ and occ.is_yellow != color:
                add(nr, nc)

    return moves


def gen_moves_for_piece(board: Board, r: int, c: int):
    p = board.get_piece(r, c)
    if not p:
        return []
    t = p.piece_type
    if t == UBAR:
        return gen_sliding(board, r, c, None, queen_dirs())
    elif t == UBARA:
        return gen_sliding(board, r, c, 3, queen_dirs())
    elif t == BUILDER:
        return gen_sliding(board, r, c, None, rook_dirs())
    elif t == INITIATE:
        return gen_sliding(board, r, c, None, bishop_dirs())
    elif t == SCRIBE:
        return gen_sliding(board, r, c, 5, bishop_dirs())
    elif t == ASSASSIN:
        return gen_assassin(board, r, c)
    elif t == RIDER:
        return gen_rider(board, r, c)
    elif t == SPEARMAN:
        return gen_spearman(board, r, c)
    elif t == TARNSMAN:
        return gen_tarnsman(board, r, c)
    elif t == HOMESTONE:
        return gen_homestone(board, r, c)
    return []

def generate_legal_moves(engine, board, is_yellow_turn):
    """Return all possible moves for current side. Format may be 2-tuple or 3-tuple if promotion."""
    moves = []
    # place homestone if needed
    if is_yellow_turn and not engine.yellow_homestone_placed:
        row = 9
        for cc in range(BOARD_COLS):
            if board.get_piece(row, cc) is None:
                moves.append(((-1,-1),(row, cc)))  # place HS
    if (not is_yellow_turn) and not engine.red_homestone_placed:
        row = 0
        for cc in range(BOARD_COLS):
            if board.get_piece(row, cc) is None:
                moves.append(((-1,-1),(row, cc)))

    # normal piece moves
    for rr in range(BOARD_ROWS):
        for cc in range(BOARD_COLS):
            piece = board.get_piece(rr, cc)
            if piece and piece.is_yellow == is_yellow_turn:
                piece_moves = gen_moves_for_piece(board, rr, cc)
                for pm in piece_moves:
                    # pm can be (er, ec) or (er, ec, promo)
                    if len(pm) == 2:
                        er, ec = pm
                        moves.append(((rr, cc), (er, ec)))
                    else:
                        er, ec, promo = pm
                        moves.append(((rr, cc), (er, ec), promo))
    return moves


class KaissaGameEngine:
    """
    Maintains current board, turn, game_over, homestone placement, etc.
    """
    def __init__(self):
        self.board = Board()
        self.is_yellow_turn = True
        self.game_over = False
        self.winner = None

        self.yellow_moves = 0
        self.red_moves = 0
        self.yellow_homestone_placed = False
        self.red_homestone_placed = False

        self.move_history = []

    def reset(self):
        self.__init__()

    def clone(self):
        new_eng = KaissaGameEngine.__new__(KaissaGameEngine)
        new_eng.board = self.board.clone()
        new_eng.is_yellow_turn = self.is_yellow_turn
        new_eng.game_over = self.game_over
        new_eng.winner = self.winner
        new_eng.yellow_moves = self.yellow_moves
        new_eng.red_moves = self.red_moves
        new_eng.yellow_homestone_placed = self.yellow_homestone_placed
        new_eng.red_homestone_placed = self.red_homestone_placed
        new_eng.move_history = copy.deepcopy(self.move_history)
        return new_eng

    def get_state(self):
        """
        2D array of (ptype, is_yellow, has_moved) or None.
        """
        out = []
        for r in range(BOARD_ROWS):
            row_data = []
            for c in range(BOARD_COLS):
                p = self.board.get_piece(r, c)
                if p:
                    row_data.append((p.piece_type, p.is_yellow, p.has_moved))
                else:
                    row_data.append(None)
            out.append(row_data)
        return out

    def is_game_over(self):
        return self.game_over

    def get_winner(self):
        return self.winner  # "Yellow" or "Red" or None

    def get_legal_moves(self):
        return generate_legal_moves(self, self.board, self.is_yellow_turn)

    def apply_move(self, move):
        if self.game_over:
            return
        if len(move) == 2:
            (start, end) = move
            promo = None
        elif len(move) == 3:
            (start, end, promo) = move
        else:
            return

        # place homestone
        if start == (-1, -1):
            (er, ec) = end
            self._place_homestone(er, ec)
            self._end_turn()
            return

        (sr, sc) = start
        (er, ec) = end
        moving_piece = self.board.get_piece(sr, sc)
        if not moving_piece:
            return
        target = self.board.get_piece(er, ec)
        # capturing homestone => immediate game over
        if target and target.piece_type == HOMESTONE:
            self._record_move(moving_piece, start, end)
            self.board.move_piece(start, end)
            self.game_over = True
            self.winner = "Yellow" if moving_piece.is_yellow else "Red"
            return
        # normal move
        self._record_move(moving_piece, start, end)
        self.board.move_piece(start, end)

        # handle promotion if needed
        if promo is not None:
            new_p = self.board.get_piece(er, ec)
            if new_p and new_p.piece_type == SPEARMAN:
                new_p.piece_type = promo

        self._end_turn()

    def _record_move(self, piece, start, end):
        sr, sc = start
        er, ec = end
        self.move_history.append({
            "piece_type": piece.piece_type,
            "is_yellow": piece.is_yellow,
            "start": [sr, sc],
            "end": [er, ec]
        })

    def _end_turn(self):
        if self.is_yellow_turn:
            self.yellow_moves += 1
            if self.yellow_moves == 10 and not self.yellow_homestone_placed:
                self.game_over = True
                self.winner = "Red"
        else:
            self.red_moves += 1
            if self.red_moves == 10 and not self.red_homestone_placed:
                self.game_over = True
                self.winner = "Yellow"
        if not self.game_over:
            self.is_yellow_turn = not self.is_yellow_turn

    def _place_homestone(self, r, c):
        p = Piece(HOMESTONE, self.is_yellow_turn)
        self.board.set_piece(r, c, p)
        if self.is_yellow_turn:
            self.yellow_homestone_placed = True
        else:
            self.red_homestone_placed = True
        self.move_history.append({
            "piece_type": HOMESTONE,
            "is_yellow": self.is_yellow_turn,
            "start": [-1, -1],
            "end": [r, c]
        })
