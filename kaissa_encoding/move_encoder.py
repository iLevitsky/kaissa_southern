# kaissa_encoding/move_encoder.py

PROMO_NONE = 0
PROMO_RIDER = 1
PROMO_TARNSMAN = 2

ACTION_SPACE_SIZE = 10 * 10 * 10 * 10 * 3  # 30,000

def encode_move(sr, sc, er, ec, promo_id):
    """
    Convert move to a single integer index in [0..ACTION_SPACE_SIZE-1].
    promo_id should be 0 (none), 1 (Rider), 2 (Tarnsman).
    """
    return (
        sr * 10 * 10 * 10 * 3 +
        sc * 10 * 10 * 3 +
        er * 10 * 3 +
        ec * 3 +
        promo_id
    )

def decode_move(action_index):
    """
    Convert integer index back to (sr, sc, er, ec, promo_id).
    """
    sr_block = 10 * 10 * 10 * 3  # 3000
    sc_block = 10 * 10 * 3       # 300
    er_block = 10 * 3           # 30
    ec_block = 3                # 3

    sr = action_index // sr_block
    remainder = action_index % sr_block

    sc = remainder // sc_block
    remainder = remainder % sc_block

    er = remainder // er_block
    remainder = remainder % er_block

    ec = remainder // ec_block
    promo_id = remainder % ec_block

    return (sr, sc, er, ec, promo_id)

def to_action_index(move):
    """
    Converts a move like ((sr, sc), (er, ec)) or ((sr, sc), (er, ec), promo_type)
    into a flat action index for AlphaZero.
    """
    if len(move) == 2:
        (sr, sc), (er, ec) = move
        promo_id = 0
    else:
        (sr, sc), (er, ec), promo_type = move
        if promo_type is None:
            promo_id = 0
        elif promo_type == 7:  # Rider
            promo_id = 1
        elif promo_type == 2:  # Tarnsman
            promo_id = 2
        else:
            promo_id = 0  # fallback if unknown

    return encode_move(sr, sc, er, ec, promo_id)

def from_action_index(action_index):
    """
    Decodes action_index back into (sr, sc, er, ec, promo_id)
    """
    sr_block = 10 * 10 * 10 * 3  # 3000
    sc_block = 10 * 10 * 3       # 300
    er_block = 10 * 3            # 30
    ec_block = 3                 # 3

    sr = action_index // sr_block
    remainder = action_index % sr_block

    sc = remainder // sc_block
    remainder = remainder % sc_block

    er = remainder // er_block
    remainder = remainder % er_block

    ec = remainder // ec_block
    promo_id = remainder % ec_block

    return sr, sc, er, ec, promo_id


__all__ = [
    "ACTION_SPACE_SIZE",
    "encode_move",
    "decode_move",
    "to_action_index",
    "from_action_index"
]


