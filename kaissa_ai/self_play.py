import os
import json
import time
import datetime
import numpy as np

import torch

from kaissa_engine.kaissa_engine import KaissaGameEngine
from kaissa_engine.__version__ import __version__ as AI_VERSION

# We'll import separate MCTS
from kaissa_ai.mcts import MCTS
from kaissa_encoding.board_encoder import encode_board
from kaissa_encoding.move_encoder import (
    ACTION_SPACE_SIZE,
    to_action_index,
    from_action_index
)
from kaissa_net.model import KaissaNet

def self_play_games(
    model_path,
    output_dir,
    num_games=10,
    simulations=800,
    device="cpu",
    move_limit=80
):
    """
    Loads a neural net from model_path, plays num_games self-play,
    saves JSON in output_dir/v{AI_VERSION}/game_XXX.json.

    We've added a move_limit to avoid infinite games on a large board.
    """
    # Prepare output folder
    version_folder = os.path.join(output_dir, f"v{AI_VERSION}")
    os.makedirs(version_folder, exist_ok=True)

    # Load model once
    model = KaissaNet()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # Create two MCTS instances (they can share the same model),
    # or we can do just one MCTS. But having two clarifies the color logic.
    mcts_yellow = MCTS(model, simulations=simulations, c_puct=1.5, device=device)
    mcts_red    = MCTS(model, simulations=simulations, c_puct=1.5, device=device)

    for g_i in range(num_games):
        game_engine = KaissaGameEngine()
        moves_data = []
        game_id = f"kaissa_{time.strftime('%Y%m%d_%H%M%S')}_{g_i:03d}"
        print(f"Starting self-play game {game_id}")

        while not game_engine.is_game_over():
            # Safety check: If we exceed move_limit, call it a draw
            if len(moves_data) >= move_limit:
                print(f"Move limit ({move_limit}) reached. Ending as a draw.")
                game_engine.game_over = True
                break

            move_num = len(moves_data) + 1
            current_player = "Yellow" if game_engine.is_yellow_turn else "Red"
            print(f"[Turn {move_num}] {current_player} thinking...")

            # Depending on who is to move, pick from the corresponding MCTS
            if game_engine.is_yellow_turn:
                visits, root_q = mcts_yellow.run(game_engine.clone())
            else:
                visits, root_q = mcts_red.run(game_engine.clone())

            print(f"[Turn {move_num}] MCTS completed. Selecting move...")

            # Turn visits into a normalized policy
            visit_sum = np.sum(visits)
            if visit_sum > 0:
                policy = visits / visit_sum
            else:
                # if no visits, fallback
                policy = np.zeros(ACTION_SPACE_SIZE, dtype=np.float32)

            # pick an action via argmax for simplicity
            action_index = np.argmax(policy)

            # decode action
            move = from_action_index(action_index)

            # record data
            current_state = game_engine.get_state()
            moves_data.append({
                "player": current_player,
                "state": current_state,  # raw state
                "policy": policy.tolist(),
                "move": list(move_to_list(move)),  # [sr, sc, er, ec, promo]
                "root_value": float(root_q),
                "search_count": int(visit_sum)
            })

            # apply to engine
            game_engine.apply_move(move)

        # if the loop ends, either game_over or move_limit
        winner = game_engine.get_winner()  # "Yellow", "Red", or None
        if not winner and len(moves_data) >= move_limit:
            # Force a draw if no winner
            winner = None  # or "Draw" if you prefer
            game_engine.game_over = True

        print(f"Game {game_id} ended, winner = {winner}")

        # build final JSON
        game_json = {
            "game_id": game_id,
            "timestamp": datetime.datetime.now().isoformat(),
            "ai_version": f"v{AI_VERSION}",
            "winner": winner,
            "moves": moves_data
        }

        # save
        outfile = os.path.join(version_folder, f"{game_id}.json")
        with open(outfile, "w") as f:
            json.dump(game_json, f, indent=2)

def move_to_list(move):
    """
    Converts engine move -> [sr, sc, er, ec, promo_id].
    Handles ( (sr,sc),(er,ec) ), ( (sr,sc),(er,ec), promo ), or flat.
    """
    if isinstance(move, tuple) and len(move) == 2:
        (sr, sc), (er, ec) = move
        promo_id = 0
    elif isinstance(move, tuple) and len(move) == 3:
        (sr, sc), (er, ec), promo = move
        promo_id = promo if promo is not None else 0
    elif isinstance(move, tuple) and len(move) == 5:
        # Already flat format, from from_action_index
        sr, sc, er, ec, promo_id = move
    else:
        raise ValueError(f"Invalid move format: {move}")
    return [int(sr), int(sc), int(er), int(ec), int(promo_id)]
