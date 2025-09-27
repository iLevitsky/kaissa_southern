# kaissa_encoding/board_encoder.py

import torch
import numpy as np

BOARD_ROWS = 10
BOARD_COLS = 10

def encode_board(state):
    """
    state is a 10x10 array of (piece_type, is_yellow, has_moved) or None.
    Return a torch.FloatTensor of shape [channels=3, rows=10, cols=10].
    """
    tensor = np.zeros((3, BOARD_ROWS, BOARD_COLS), dtype=np.float32)
    for r in range(BOARD_ROWS):
        for c in range(BOARD_COLS):
            cell = state[r][c]
            if cell is not None:
                (ptype, is_yellow, has_moved) = cell
                tensor[0, r, c] = float(ptype)       # piece type 0..9
                tensor[1, r, c] = 1.0 if is_yellow else 0.0
                tensor[2, r, c] = 1.0 if has_moved else 0.0
    return torch.from_numpy(tensor)
