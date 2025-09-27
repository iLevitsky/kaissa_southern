# training/dataset_loader.py

import os
import json
import torch
import numpy as np
from torch.utils.data import Dataset

from kaissa_encoding.board_encoder import encode_board
from kaissa_encoding.move_encoder import ACTION_SPACE_SIZE

class KaissaDataset(Dataset):
    """
    Loads all .json games from a folder and extracts (state, policy, value).
    """

    def __init__(self, data_dir):
        super().__init__()
        self.samples = []  # each sample is (state_tensor, policy_vec, value)

        # load all JSON files
        files = [f for f in os.listdir(data_dir) if f.endswith(".json")]
        for fname in files:
            fullpath = os.path.join(data_dir, fname)
            with open(fullpath, "r") as f:
                game_data = json.load(f)
            winner = game_data["winner"]
            if winner == "Yellow":
                winner_value = 1.0
            elif winner == "Red":
                winner_value = -1.0
            else:
                winner_value = 0.0  # draw or unknown

            moves = game_data["moves"]
            # The game_value from perspective of last mover
            # We'll store +1 for the side that actually got winner
            # But in typical AlphaZero, we store "value" from the perspective
            # of the current player at each step => meaning we must flip sign if color changes
            # We'll do a simpler approach: each move from the winner's perspective => if that move's player is the winner's color => value=+1, else -1

            for move_item in moves:
                st = move_item["state"]
                policy = np.array(move_item["policy"], dtype=np.float32)
                player_color = move_item["player"]  # "Yellow"/"Red"

                # encode state to tensor
                state_tensor = encode_board(st)

                # compute the value from current player's perspective
                if winner is None:
                    value = 0.0
                else:
                    if player_color == winner:
                        value = 1.0
                    else:
                        value = -1.0

                self.samples.append((state_tensor, policy, value))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        (state_tensor, policy_vec, value) = self.samples[idx]
        return {
            "state_tensor": state_tensor,  # shape [3,10,10]
            "policy": torch.from_numpy(policy_vec),  # shape [30000]
            "value": torch.tensor(value, dtype=torch.float32)  # scalar
        }
