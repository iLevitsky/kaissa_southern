# kaissa_engine/__init__.py

"""
Initialize the kaissa_engine package.
Imports constants and the main engine.
"""

from .piece_constants import (
    UBAR, UBARA, TARNSMAN, BUILDER, INITIATE, SCRIBE, ASSASSIN, RIDER, SPEARMAN, HOMESTONE
)
from .kaissa_engine import KaissaGameEngine, BOARD_ROWS, BOARD_COLS
