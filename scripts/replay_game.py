# scripts/replay_game.py

import json
import sys
import time
from kaissa_engine.kaissa_engine import KaissaGameEngine

def replay_game(json_path, delay=1.0):
    with open(json_path, "r") as f:
        game_data = json.load(f)

    game_id = game_data["game_id"]
    winner = game_data["winner"]
    moves = game_data["moves"]

    print(f"Replaying game {game_id}, winner={winner}")
    engine = KaissaGameEngine()

    for i, move_item in enumerate(moves):
        move_list = move_item["move"]  # [sr, sc, er, ec, promo]
        # Convert to engine's format
        if move_list[4] == 0:
            # no promo
            mv = ((move_list[0], move_list[1]), (move_list[2], move_list[3]))
        else:
            mv = ((move_list[0], move_list[1]), (move_list[2], move_list[3]), move_list[4])

        print(f"Move {i+1}: {move_item['player']} -> {mv}")
        engine.apply_move(mv)
        time.sleep(delay)

    print("Final board state:")
    print(engine.get_state())

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python replay_game.py <game.json> [delay]")
        sys.exit(1)

    json_path = sys.argv[1]
    d = 1.0 if len(sys.argv) < 3 else float(sys.argv[2])
    replay_game(json_path, delay=d)
