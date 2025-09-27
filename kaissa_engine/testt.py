import unittest
import threading
import sys
import time

# Import your Kaissa engine here.
from kaissa_engine import KaissaGameEngine  # Adjust as per your file/module

# Assuming the KaissaGameEngine is defined in the same script for now
# (Otherwise, import from your module)

class TestRecursionSafety(unittest.TestCase):
    def setUp(self):
        self.engine = KaissaGameEngine()

    def simulate_moves(self):
        move_count = 0
        max_moves = 200  # reasonable upper bound to catch looping

        while not self.engine.is_game_over() and move_count < max_moves:
            legal_moves = self.engine.get_legal_moves()
            if not legal_moves:
                break  # no legal moves = stuck = not infinite
            move = legal_moves[0]
            self.engine.apply_move(move)
            move_count += 1

    def run_with_timeout(self, timeout_seconds=3):
        """Helper to run simulate_moves() with timeout"""
        thread = threading.Thread(target=self.simulate_moves)
        thread.start()
        thread.join(timeout_seconds)

        if thread.is_alive():
            raise RecursionError("Possible infinite loop or recursion in game engine.")

    def test_no_infinite_recursion(self):
        try:
            self.run_with_timeout()
        except RecursionError as e:
            self.fail(f"Engine caused infinite recursion: {e}")

    def test_recursion_depth_limit(self):
        """Ensure we don't hit Python's recursion depth limit"""
        try:
            sys.setrecursionlimit(2000)  # Increase just to test
            self.run_with_timeout()
        except RecursionError:
            self.fail("Recursion limit exceeded — infinite recursion likely.")
        finally:
            sys.setrecursionlimit(1000)  # Reset to default

if __name__ == '__main__':
    unittest.main()
