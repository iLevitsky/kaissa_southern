# scripts/evaluate.py

import torch
import numpy as np
from kaissa_net.model import KaissaNet
from kaissa_engine.kaissa_engine import KaissaGameEngine
from kaissa_ai.mcts import MCTS

def evaluate_models(model_a_path, model_b_path, games=10, device="cpu"):
    """
    Let model A and B play 'games/2' times with each color, total = games
    Return the stats.
    """
    model_a = KaissaNet()
    model_a.load_state_dict(torch.load(model_a_path, map_location=device))
    model_a.eval().to(device)

    model_b = KaissaNet()
    model_b.load_state_dict(torch.load(model_b_path, map_location=device))
    model_b.eval().to(device)

    def play_one_game(model_yellow, model_red):
        eng = KaissaGameEngine()
        mcts_yellow = MCTS(model_yellow, simulations=200, device=device)
        mcts_red = MCTS(model_red, simulations=200, device=device)

        while not eng.is_game_over():
            if eng.is_yellow_turn:
                visits, root_q = mcts_yellow.run(eng)
            else:
                visits, root_q = mcts_red.run(eng)

            policy = visits / (np.sum(visits) + 1e-8)
            action_index = np.argmax(policy)
            mv = mcts_red.from_action_index(action_index)  # same from_action_index
            eng.apply_move(mv)

        return eng.get_winner()

    a_wins = 0
    b_wins = 0
    draws = 0
    half_games = games // 2

    # half with A=yellow, B=red
    for _ in range(half_games):
        w = play_one_game(model_a, model_b)
        if w == "Yellow":
            a_wins += 1
        elif w == "Red":
            b_wins += 1
        else:
            draws += 1

    # half with B=yellow, A=red
    for _ in range(half_games):
        w = play_one_game(model_b, model_a)
        if w == "Yellow":
            b_wins += 1
        elif w == "Red":
            a_wins += 1
        else:
            draws += 1

    print(f"After {games} games: A wins={a_wins}, B wins={b_wins}, draws={draws}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python evaluate.py <model_a> <model_b> [games=10]")
        sys.exit(1)
    model_a = sys.argv[1]
    model_b = sys.argv[2]
    g = 10
    if len(sys.argv) >= 4:
        g = int(sys.argv[3])
    evaluate_models(model_a, model_b, g)
