# scripts/play_vs_ai.py

import torch
import numpy as np

from kaissa_engine.kaissa_engine import KaissaGameEngine
from kaissa_net.model import KaissaNet
from kaissa_ai.mcts import MCTS, from_action_index
from kaissa_encoding.move_encoder import ACTION_SPACE_SIZE
from kaissa_engine.__version__ import __version__ as AI_VERSION

def play_vs_ai(model_path, simulations=200, device="cpu"):
    model = KaissaNet()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    model.to(device)

    mcts = MCTS(model, simulations=simulations, c_puct=1.5, device=device)
    engine = KaissaGameEngine()

    while not engine.is_game_over():
        if engine.is_yellow_turn:
            # human move
            print(engine.get_state())
            moves = engine.get_legal_moves()
            print(f"Legal moves for Yellow: {moves}")

            choice = int(input("Select move index (0-based): "))
            chosen_move = moves[choice]
            engine.apply_move(chosen_move)
        else:
            # AI move
            visits, root_q = mcts.run(engine)
            if np.sum(visits) > 0:
                policy = visits / np.sum(visits)
            else:
                policy = visits

            action_index = np.argmax(policy)
            move = from_action_index(action_index)
            engine.apply_move(move)
            print(f"AI (Red) moves {move}")

    winner = engine.get_winner()
    print("Game over! Winner =", winner, "| AI version =", AI_VERSION)

if __name__ == "__main__":
    # Example usage:
    # python play_vs_ai.py checkpoints/model_v0.0.2.pth
    import sys
    if len(sys.argv) < 2:
        print("Usage: python play_vs_ai.py <model_path>")
        sys.exit(1)
    model_path = sys.argv[1]
    play_vs_ai(model_path)
