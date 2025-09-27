###############################
# Kaissa Engine (Optimized)
###############################

# Piece type constants:
EMPTY     = 0
UBAR      = 1
UBARA     = 2
TARNSMAN  = 3
BUILDER   = 4
INITIATE  = 5
SCRIBE    = 6
ASSASSIN  = 7
RIDER     = 8
SPEARMAN  = 9
HOMESTONE = 10


BOARD_ROWS = 10
BOARD_COLS = 10
NUM_SQUARES = BOARD_ROWS * BOARD_COLS

# Bitwise layout (6 bits total):
#   bit 0..3: piece_type (0..9)
#   bit 4:    color (0 => Red, 1 => Yellow)
#   bit 5:    has_moved flag
HAS_MOVED_MASK = 1 << 5
COLOR_MASK     = 1 << 4
TYPE_MASK      = 0b1111  # bits 0..3

def encode_piece(piece_type: int, is_yellow: bool, has_moved: bool=False) -> int:
    val = piece_type & TYPE_MASK
    if is_yellow:
        val |= COLOR_MASK
    if has_moved:
        val |= HAS_MOVED_MASK
    return val

def get_piece_type(piece_val: int) -> int:
    return piece_val & TYPE_MASK

def get_color_bit(piece_val: int) -> int:
    return (piece_val & COLOR_MASK) >> 4

def is_yellow(piece_val: int) -> bool:
    return (piece_val & COLOR_MASK) != 0

def has_moved(piece_val: int) -> bool:
    return bool(piece_val & HAS_MOVED_MASK)

def set_has_moved(piece_val: int) -> int:
    return piece_val | HAS_MOVED_MASK

def is_empty(piece_val: int) -> bool:
    return piece_val == 0

def square_index(r: int, c: int) -> int:
    return r * BOARD_COLS + c

def index_to_rc(idx: int):
    return (idx // BOARD_COLS, idx % BOARD_COLS)

def in_bounds(r: int, c: int) -> bool:
    return 0 <= r < BOARD_ROWS and 0 <= c < BOARD_COLS


####################################
# Board Class (Single-Array Storage)
####################################
class Board:
    def __init__(self):
        # 0 means empty. Otherwise holds encoded piece.
        self.squares = [0] * NUM_SQUARES
        self.setup_initial_position()

    def setup_initial_position(self):
        """
        Standard Kaissa layout, 10x10.
         - Yellow on rows 9 & 8
         - Red on rows 0 & 1
        """
        # Arrays for the back + front rows
        yellow_back =  [INITIATE, BUILDER, SCRIBE, TARNSMAN, UBAR, UBARA, TARNSMAN, SCRIBE, BUILDER, INITIATE]
        yellow_front = [ASSASSIN, RIDER, SPEARMAN, SPEARMAN, SPEARMAN, SPEARMAN, SPEARMAN, SPEARMAN, RIDER, ASSASSIN]
        red_back    =  [INITIATE, BUILDER, SCRIBE, TARNSMAN, UBAR, UBARA, TARNSMAN, SCRIBE, BUILDER, INITIATE]
        red_front   =  [ASSASSIN, RIDER, SPEARMAN, SPEARMAN, SPEARMAN, SPEARMAN, SPEARMAN, SPEARMAN, RIDER, ASSASSIN]

        # Place Yellow (rows 9,8)
        for c in range(BOARD_COLS):
            self.set_piece(9, c, encode_piece(yellow_back[c], True, False))
            self.set_piece(8, c, encode_piece(yellow_front[c], True, False))

        # Place Red (rows 0,1)
        for c in range(BOARD_COLS):
            self.set_piece(0, c, encode_piece(red_back[c],  False, False))
            self.set_piece(1, c, encode_piece(red_front[c], False, False))

    def clone_into(self, other_board: 'Board'):
        """Copy all squares into another Board object (fast)."""
        other_board.squares = self.squares[:]

    def get_piece(self, r: int, c: int) -> int:
        """Returns encoded piece int or 0 if empty/out of bounds."""
        if not in_bounds(r, c):
            return 0
        return self.squares[square_index(r,c)]

    def set_piece(self, r: int, c: int, piece_val: int):
        self.squares[square_index(r,c)] = piece_val

    def move_piece(self, start_idx: int, end_idx: int) -> None:
        """Move piece from start_idx to end_idx. Clears the start square."""
        p = self.squares[start_idx]
        # set has_moved bit
        p = set_has_moved(p)
        self.squares[end_idx] = p
        self.squares[start_idx] = 0


############################################
# Move Structure for Make/Undo
############################################
class MoveRecord:
    """
    Stores enough info to undo a move.
    """
    __slots__ = (
        'start_idx', 'end_idx', 'moved_piece', 
        'captured_piece', 'old_has_moved',
        'promotion_new_type',
    )

    def __init__(self, start_idx, end_idx, moved_piece, captured_piece, old_has_moved, promotion_new_type=None):
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.moved_piece = moved_piece
        self.captured_piece = captured_piece
        self.old_has_moved = old_has_moved
        self.promotion_new_type = promotion_new_type


############################################
# Movement Generation Utils
############################################
def king_dirs():
    return [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]

def rook_dirs():
    return [(1,0),(-1,0),(0,1),(0,-1)]

def bishop_dirs():
    return [(1,1),(1,-1),(-1,1),(-1,-1)]

def queen_dirs():
    return rook_dirs() + bishop_dirs()

def gen_sliding(board: Board, r: int, c: int, max_dist, directions):
    """For rooks, bishops, queens, or short-range variants (Ubara/Scribe)."""
    start_idx = square_index(r,c)
    p_val = board.squares[start_idx]
    color_yellow = is_yellow(p_val)
    moves = []

    for (dr, dc) in directions:
        steps = 0
        nr, nc = r, c
        while True:
            nr += dr
            nc += dc
            steps += 1
            if not in_bounds(nr, nc):
                break
            sq_idx = square_index(nr, nc)
            occupant = board.squares[sq_idx]
            if occupant != 0:
                # Occupied - can capture if enemy
                if is_yellow(occupant) != color_yellow:
                    moves.append(sq_idx)
                break
            else:
                # Empty
                moves.append(sq_idx)
            if max_dist is not None and steps >= max_dist:
                break
    return moves

def gen_assassin(board: Board, r: int, c: int):
    moves = []
    start_idx = square_index(r,c)
    p_val = board.squares[start_idx]
    color_yellow = is_yellow(p_val)
    for (dr, dc) in queen_dirs():
        for dist in [1, 2]:
            nr = r + dr*dist
            nc = c + dc*dist
            if dist == 2:
                # can't jump
                mid_r = r + dr
                mid_c = c + dc
                if in_bounds(mid_r, mid_c):
                    if board.squares[square_index(mid_r, mid_c)] != 0:
                        # blocked
                        continue
            if in_bounds(nr, nc):
                sq_idx = square_index(nr, nc)
                occupant = board.squares[sq_idx]
                if occupant == 0 or is_yellow(occupant) != color_yellow:
                    moves.append(sq_idx)
    return moves

def gen_rider(board: Board, r: int, c: int):
    moves = []
    start_idx = square_index(r,c)
    p_val = board.squares[start_idx]
    color_yellow = is_yellow(p_val)
    for (dr, dc) in king_dirs():
        nr = r + dr
        nc = c + dc
        if in_bounds(nr, nc):
            sq_idx = square_index(nr, nc)
            occupant = board.squares[sq_idx]
            if occupant == 0 or is_yellow(occupant) != color_yellow:
                moves.append(sq_idx)
    return moves

def gen_tarnsman(board: Board, r: int, c: int):
    moves = []
    start_idx = square_index(r,c)
    p_val = board.squares[start_idx]
    color_yellow = is_yellow(p_val)
    # Knight-like leaps
    offsets = [(3,2),(3,-2),(-3,2),(-3,-2),(2,3),(2,-3),(-2,3),(-2,-3)]
    for (dr, dc) in offsets:
        nr = r + dr
        nc = c + dc
        if in_bounds(nr, nc):
            sq_idx = square_index(nr, nc)
            occupant = board.squares[sq_idx]
            if occupant == 0 or is_yellow(occupant) != color_yellow:
                moves.append(sq_idx)
    # 1-square shift no capture
    for (dr, dc) in king_dirs():
        nr = r + dr
        nc = c + dc
        if in_bounds(nr, nc):
            sq_idx = square_index(nr, nc)
            occupant = board.squares[sq_idx]
            if occupant == 0:
                moves.append(sq_idx)
    return moves

def gen_homestone(board: Board, r: int, c: int):
    """Home Stone: can move 1 square in any direction if empty. No captures."""
    moves = []
    for (dr, dc) in king_dirs():
        nr, nc = r+dr, c+dc
        if in_bounds(nr, nc):
            sq_idx = square_index(nr, nc)
            if board.squares[sq_idx] == 0:
                moves.append(sq_idx)
    return moves

def gen_spearman(board: Board, r: int, c: int):
    """
    Spearman:
     - forward 1..3 if not moved, else 1
     - sideways or diag forward if empty
     - diag forward capture
     - can promote if reaching last rank
    """
    moves = []
    start_idx = square_index(r,c)
    p_val = board.squares[start_idx]
    color_yellow = is_yellow(p_val)

    direction = -1 if color_yellow else 1
    if not has_moved(p_val):
        forward_cap = 3
    else:
        forward_cap = 1

    # last rank
    last_rank = 0 if color_yellow else 9

    def add_move(er, ec, capture=False):
        # Check for promotion if on last rank
        if er == last_rank:
            # remain Spearman or become Rider/Tarnsman
            # We'll store these as tuples: (end_idx, new_type)
            end_idx = square_index(er, ec)
            # moves.append((end_idx, None))       # remain Spearman
            moves.append((end_idx, RIDER))      # promote to Rider
            moves.append((end_idx, TARNSMAN))   # promote to Tarnsman
        else:
            # normal move, no promotion
            end_idx = square_index(er, ec)
            moves.append((end_idx, None))

    # forward
    for step in range(1, forward_cap + 1):
        nr = r + step*direction
        nc = c
        if not in_bounds(nr, nc):
            break
        occupant = board.squares[square_index(nr, nc)]
        if occupant == 0:
            add_move(nr, nc)
        else:
            break

    # sideways & diag forward if empty
    for dc in [-1, 1]:
        # sideways
        nr, nc = r, c + dc
        if in_bounds(nr, nc):
            if board.squares[square_index(nr, nc)] == 0:
                add_move(nr, nc)
        # diag forward
        nr, nc = r + direction, c + dc
        if in_bounds(nr, nc):
            if board.squares[square_index(nr, nc)] == 0:
                add_move(nr, nc)

    # capturing diagonally forward
    for dc in [-1, 1]:
        nr = r + direction
        nc = c + dc
        if in_bounds(nr, nc):
            occ = board.squares[square_index(nr, nc)]
            if occ != 0 and is_yellow(occ) != color_yellow:
                add_move(nr, nc, capture=True)

    return moves

def gen_moves_for_piece(board: Board, r: int, c: int):
    start_idx = square_index(r,c)
    piece_val = board.squares[start_idx]
    if piece_val == EMPTY:
        return []

    t = get_piece_type(piece_val)
    if t == UBAR:
        # infinite range queen
        dsts = gen_sliding(board, r, c, None, queen_dirs())
        return [(d, None) for d in dsts]
    elif t == UBARA:
        # queen but max distance=3
        dsts = gen_sliding(board, r, c, 3, queen_dirs())
        return [(d, None) for d in dsts]
    elif t == BUILDER:
        # rook infinite
        dsts = gen_sliding(board, r, c, None, rook_dirs())
        return [(d, None) for d in dsts]
    elif t == INITIATE:
        # bishop infinite
        dsts = gen_sliding(board, r, c, None, bishop_dirs())
        return [(d, None) for d in dsts]
    elif t == SCRIBE:
        # bishop range=5
        dsts = gen_sliding(board, r, c, 5, bishop_dirs())
        return [(d, None) for d in dsts]
    elif t == ASSASSIN:
        dsts = gen_assassin(board, r, c)
        return [(d, None) for d in dsts]
    elif t == RIDER:
        dsts = gen_rider(board, r, c)
        return [(d, None) for d in dsts]
    elif t == SPEARMAN:
        # This returns possible squares plus optional promotions
        return gen_spearman(board, r, c)
    elif t == TARNSMAN:
        dsts = gen_tarnsman(board, r, c)
        return [(d, None) for d in dsts]
    elif t == HOMESTONE:
        dsts = gen_homestone(board, r, c)
        return [(d, None) for d in dsts]
    return []






############################################
# Legal Move Generation for Current Side
############################################
def generate_legal_moves(engine, board: Board, is_yellow_turn: bool):
    moves = []
    # place homestone if needed
    if is_yellow_turn and not engine.yellow_homestone_placed:
        row = 9
        for cc in range(BOARD_COLS):
            if board.squares[square_index(row, cc)] == 0:
                # We represent "placing homestone" as a special move:
                # (start=(-1,-1), end=(row,cc))
                moves.append(((-1,-1),(row,cc)))
    if (not is_yellow_turn) and not engine.red_homestone_placed:
        row = 0
        for cc in range(BOARD_COLS):
            if board.squares[square_index(row, cc)] == 0:
                moves.append(((-1,-1),(row,cc)))

    # normal piece moves
    for r in range(BOARD_ROWS):
        for c in range(BOARD_COLS):
            sq_idx = square_index(r, c)
            piece_val = board.squares[sq_idx]
            if piece_val != 0 and is_yellow(piece_val) == is_yellow_turn:
                # generate moves
                all_moves = gen_moves_for_piece(board, r, c)
                for (end_idx, promo) in all_moves:
                    moves.append(((r,c), index_to_rc(end_idx), promo))
    return moves


def get_protected_squares(board: Board, color_flag: bool):
    protected_counts = {}

    for r in range(BOARD_ROWS):
        for c in range(BOARD_COLS):
            p_val = board.get_piece(r, c)
            if p_val == 0 or is_yellow(p_val) != color_flag:
                continue

            t = get_piece_type(p_val)

            # Get all directions or move targets (include blocked ones)
            directions = []
            max_range = None

            if t in [UBAR, BUILDER]:  # Queen or Rook
                directions = queen_dirs() if t == UBAR else rook_dirs()
            elif t in [UBARA, SCRIBE]:  # Limited queen/bishop
                directions = queen_dirs() if t == UBARA else bishop_dirs()
                max_range = 3 if t == UBARA else 5
            elif t == INITIATE:
                directions = bishop_dirs()
            elif t == ASSASSIN:
                directions = queen_dirs()
                max_range = 2
            elif t == RIDER or t == HOMESTONE:
                directions = king_dirs()
                max_range = 1
            elif t == TARNSMAN:
                directions = king_dirs()
                max_range = 1

            if directions:
                for (dr, dc) in directions:
                    steps = 0
                    nr, nc = r, c
                    while True:
                        nr += dr
                        nc += dc
                        steps += 1
                        if not in_bounds(nr, nc):
                            break
                        sq_idx = square_index(nr, nc)
                        target = board.get_piece(nr, nc)
                        if target != 0:
                            if is_yellow(target) == color_flag:
                                protected_counts[sq_idx] = protected_counts.get(sq_idx, 0) + 1
                            break
                        if max_range and steps >= max_range:
                            break

            # Knight-leaps (Tarnsman)
            if t == TARNSMAN:
                for (dr, dc) in [(3,2),(3,-2),(-3,2),(-3,-2),(2,3),(2,-3),(-2,3),(-2,-3)]:
                    nr, nc = r+dr, c+dc
                    if in_bounds(nr, nc):
                        sq_idx = square_index(nr, nc)
                        target = board.get_piece(nr, nc)
                        if target != 0 and is_yellow(target) == color_flag:
                            protected_counts[sq_idx] = protected_counts.get(sq_idx, 0) + 1

    return protected_counts



############################################
# KaissaGameEngine
############################################
class KaissaGameEngine:
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
        # For make/undo
        self.move_stack = []

    def reset(self):
        self.__init__()

    def clone(self):
        """Create a new KaissaGameEngine with the same state."""
        new_eng = KaissaGameEngine.__new__(KaissaGameEngine)
        new_eng.board = Board()
        self.board.clone_into(new_eng.board)

        new_eng.is_yellow_turn = self.is_yellow_turn
        new_eng.game_over = self.game_over
        new_eng.winner = self.winner
        new_eng.yellow_moves = self.yellow_moves
        new_eng.red_moves = self.red_moves
        new_eng.yellow_homestone_placed = self.yellow_homestone_placed
        new_eng.red_homestone_placed = self.red_homestone_placed

        new_eng.move_history = list(self.move_history)
        new_eng.move_stack = list(self.move_stack)
        return new_eng

    def get_state(self):
        """Return a 2D array of (ptype, is_yellow, has_moved) or None for easy display."""
        out = []
        for r in range(BOARD_ROWS):
            row_data = []
            for c in range(BOARD_COLS):
                p_val = self.board.get_piece(r, c)
                if p_val == 0:
                    row_data.append(None)
                else:
                    row_data.append((
                        get_piece_type(p_val),
                        is_yellow(p_val),
                        has_moved(p_val)
                    ))
            out.append(row_data)
        return out

    def is_game_over(self):
        return self.game_over

    def get_winner(self):
        """Returns 'Yellow', 'Red', or None (draw)."""
        return self.winner

    def get_legal_moves(self):
        return generate_legal_moves(self, self.board, self.is_yellow_turn)

    ########################################
    # Make/Undo Move
    ########################################
    def make_move(self, start_idx, end_idx, promotion_type=None):
        """
        Execute a move on the board, capturing if needed.
        Return a MoveRecord so we can undo if we want.
        """
        moved_piece = self.board.squares[start_idx]
        captured_piece = self.board.squares[end_idx]
        old_has_moved = has_moved(moved_piece)

        # Move the piece
        self.board.move_piece(start_idx, end_idx)

        # Promotion?
        if promotion_type is not None:
            # Overwrite piece_type in end square
            p_val = self.board.squares[end_idx]
            # must be a spearman piece originally
            # Clear the old type bits, set new type
            new_val = (p_val & ~TYPE_MASK) | (promotion_type & TYPE_MASK)
            self.board.squares[end_idx] = new_val

        mr = MoveRecord(
            start_idx, end_idx, moved_piece, 
            captured_piece, old_has_moved, 
            promotion_type
        )
        self.move_stack.append(mr)
        return mr

    def undo_move(self):
        """
        Undo the last move made. Restores captured pieces, old flags, etc.
        """
        if not self.move_stack:
            return
        mr = self.move_stack.pop()

        # Move piece back
        self.board.squares[mr.start_idx] = mr.moved_piece
        self.board.squares[mr.end_idx] = mr.captured_piece

        # If it had old_has_moved==False, we need to clear that bit
        if mr.old_has_moved is False:
            # clear the has_moved bit
            val = self.board.squares[mr.start_idx]
            val &= ~HAS_MOVED_MASK
            self.board.squares[mr.start_idx] = val

    ########################################
    # apply_move() to match your old API
    ########################################
    def apply_move(self, move):
        """
        move can be:
          - ((-1,-1),(r,c)) => place Homestone
          - ((sr,sc),(er,ec)) => normal move
          - ((sr,sc),(er,ec),promo)
        """
        if self.game_over:
            return

        if len(move) == 2:
            (start, end) = move
            promo = None
        elif len(move) == 3:
            (start, end, promo) = move
        else:
            return

        # Place homestone special
        if start == (-1, -1):
            (er, ec) = end
            self._place_homestone(er, ec)
            self._end_turn()
            return

        (sr, sc) = start
        (er, ec) = end
        start_idx = square_index(sr, sc)
        end_idx = square_index(er, ec)

        moving_piece = self.board.squares[start_idx]
        if moving_piece == 0:
            # invalid move
            return

        target = self.board.squares[end_idx]
        # capturing homestone => immediate game over
        if target != 0 and get_piece_type(target) == HOMESTONE:
            # record move, then game over
            self._record_move(moving_piece, (sr,sc), (er,ec))
            self.make_move(start_idx, end_idx, promo)
            self.game_over = True
            self.winner = "Yellow" if is_yellow(moving_piece) else "Red"
            return

        # normal move
        self._record_move(moving_piece, (sr,sc), (er,ec))
        self.make_move(start_idx, end_idx, promo)

        self._end_turn()

    def _record_move(self, piece_val, start, end):
        (sr, sc) = start
        (er, ec) = end
        self.move_history.append({
            "piece_type": get_piece_type(piece_val),
            "is_yellow": is_yellow(piece_val),
            "start": [sr, sc],
            "end": [er, ec]
        })

    def _place_homestone(self, r, c):
        p_val = encode_piece(HOMESTONE, self.is_yellow_turn, False)
        self.board.set_piece(r, c, p_val)
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

    def _end_turn(self):
        if self.is_yellow_turn:
            self.yellow_moves += 1
        else:
            self.red_moves += 1

        # After Red's 9th move, check Homestone placement
        if (
            not self.game_over
            and not self.is_yellow_turn
            and self.red_moves == 9
        ):
            if self.yellow_homestone_placed and self.red_homestone_placed:
                # both placed => continue
                pass
            elif (not self.yellow_homestone_placed) and (not self.red_homestone_placed):
                # neither => draw
                self.game_over = True
                self.winner = None
            elif self.yellow_homestone_placed:
                # only Yellow => Yellow wins
                self.game_over = True
                self.winner = "Yellow"
            else:
                # only Red => Red wins
                self.game_over = True
                self.winner = "Red"

        if not self.game_over:
            self.is_yellow_turn = not self.is_yellow_turn
            # print("\n=== Turn: {} ===".format("Yellow" if self.is_yellow_turn else "Red"))
            #print_board(self.board) #PRINTING BOARD


########################################
# (Optional) Board-to-Tensor for ML
########################################
def board_to_tensor(board: Board):
    """
    Example: return a float tensor of shape [20, 10, 10].
    20 channels = 10 piece types × 2 colors
    Each cell is 1 if that piece is present, else 0.
    """
    import numpy as np
    tensor = np.zeros((20, BOARD_ROWS, BOARD_COLS), dtype=np.float32)
    for idx in range(NUM_SQUARES):
        p = board.squares[idx]
        if p == 0:
            continue
        pt = get_piece_type(p)
        c  = get_color_bit(p)  # 0 or 1
        r, cc = index_to_rc(idx)
        channel = pt + 10*c
        tensor[channel, r, cc] = 1.0
    return tensor


def print_board(board: Board):
    symbol_map = {
        EMPTY: ".",
        UBAR: "U",
        UBARA: "A",
        TARNSMAN: "T",
        BUILDER: "B",
        INITIATE: "I",
        SCRIBE: "S",
        ASSASSIN: "X",
        RIDER: "R",
        SPEARMAN: "P",
        HOMESTONE: "H",
    }

    def piece_to_str(p):
        if p == 0:
            return ". "
        symbol = symbol_map.get(get_piece_type(p), "?")
        return symbol.lower() + " " if is_yellow(p) else symbol.upper() + " "

    print("   " + " ".join(str(c) for c in range(BOARD_COLS)))
    for r in range(BOARD_ROWS):
        row_str = f"{r}  " + "".join(piece_to_str(board.get_piece(r, c)) for c in range(BOARD_COLS))
        print(row_str)
    print()



############################################
# Minimax Search and Best Move Prediction
############################################
def evaluate(engine: 'KaissaGameEngine') -> int:
    """
    A simple evaluation function that scores the board based on material.
    Positive scores favor Yellow; negative scores favor Red.
    """
    piece_values = {
        EMPTY:     0,     # No value for empty squares.
        UBAR:      900,   # Ubar acts as a full queen.
        UBARA:     700,   # Ubara is a limited queen.
        TARNSMAN:  500,   # Tarn acts like a knight.
        BUILDER:   550,   # Builder is comparable to a rook.
        INITIATE:  330,   # Initiate is like a bishop, on the weaker side.
        SCRIBE:    420,   # Scribe is like a bishop, but a bit stronger.
        ASSASSIN:  300,   # Assassin is an advanced pawn—more potent than a normal pawn.
        RIDER:     150,   # Rider is a better pawn.
        SPEARMAN:   50,   # Spearman is like a regular pawn.
        HOMESTONE: 10000,  # Homestone is critical; losing it loses the game.
    }

    score = 0
    for idx in range(NUM_SQUARES):
        p = engine.board.squares[idx]
        if p != 0:
            val = piece_values.get(get_piece_type(p), 0)
            score += val if is_yellow(p) else -val
    return score

def minimax(engine: 'KaissaGameEngine', depth: int, alpha: float, beta: float, maximizingPlayer: bool):
    """
    Minimax search with alpha-beta pruning.
    Returns a tuple (score, best_move).
    """
    if depth == 0 or engine.is_game_over():
        return evaluate(engine), None

    legal_moves = engine.get_legal_moves()
    best_move = None

    if maximizingPlayer:
        max_eval = -float('inf')
        for move in legal_moves:
            new_engine = engine.clone()
            new_engine.apply_move(move)
            eval_score, _ = minimax(new_engine, depth - 1, alpha, beta, False)
            if eval_score > max_eval:
                max_eval = eval_score
                best_move = move
            alpha = max(alpha, eval_score)
            if beta <= alpha:
                break
        return max_eval, best_move
    else:
        min_eval = float('inf')
        for move in legal_moves:
            new_engine = engine.clone()
            new_engine.apply_move(move)
            eval_score, _ = minimax(new_engine, depth - 1, alpha, beta, True)
            if eval_score < min_eval:
                min_eval = eval_score
                best_move = move
            beta = min(beta, eval_score)
            if beta <= alpha:
                break
        return min_eval, best_move

def predict_best_move(engine: 'KaissaGameEngine', search_depth: int = 3):
    """
    Predicts and prints the best move for the current player.
    """
    maximizingPlayer = engine.is_yellow_turn
    score, best_move = minimax(engine, search_depth, -float('inf'), float('inf'), maximizingPlayer)
    # print(f"\n[Best Move Prediction] Score: {score}")
    # print(f"[Best Move Prediction] Move: {best_move}\n")
    return best_move


def player_turn(engine: KaissaGameEngine):
    """
    Displays the board and best move suggestion, then prompts the user for a move.
    You can modify this function to automatically use the predicted move or to accept user input.
    """
    print_board(engine.board)
    
    # Print best move suggestion BEFORE moving a piece:
    best_move = predict_best_move(engine, search_depth=3)
    
    # Option 1: Automatically use the predicted move:
    # Uncomment the following two lines if you want to auto-play the predicted move.
    # print("Auto-playing predicted move...")
    # engine.apply_move(best_move)
    
    # Option 2: Let the user enter a move (this is just an example placeholder).
    # You can replace this part with your own input mechanism.
    move_input = input("Enter your move (or press Enter to use the predicted move): ")
    if move_input.strip() == "":
        # Use predicted move if no input given
        print("Using predicted move.")
        engine.apply_move(best_move)
    else:
        # Here you would parse move_input into a move tuple.
        # For now, we'll assume the predicted move is chosen.
        print("Custom moves not implemented; using predicted move.")
        engine.apply_move(best_move)