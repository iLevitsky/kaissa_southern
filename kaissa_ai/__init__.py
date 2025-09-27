"""
Utilities to provide a stronger best-move prediction for the Kaissa game.

Expose `predict_best_move_strong` at the package level:

    from kaissa_ai import predict_best_move_strong
"""

from .stronger_predictor import predict_best_move_strong

__all__ = ["predict_best_move_strong"]
