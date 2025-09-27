# kaissa_ai/mcts.py

import math
import copy
import torch
import numpy as np
from collections import defaultdict

from kaissa_encoding.move_encoder import ACTION_SPACE_SIZE, encode_move, decode_move
from kaissa_engine.kaissa_engine import KaissaGameEngine
from kaissa_encoding.board_encoder import encode_board

class MCTSNode:
    def __init__(self, engine_state, prior):
        """
        engine_state: a clone() of the KaissaGameEngine
        prior: prior probability from neural net
        """
        self.engine_state = engine_state
        self.prior = prior
        self.visit_count = 0
        self.value_sum = 0.0
        self.children = {}  # action_index -> MCTSNode
        self.is_expanded = False

    @property
    def q_value(self):
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count

    def expand(self, action_probs):
        """Initialize child nodes for each legal action."""
        self.is_expanded = True
        legal_moves = self.engine_state.get_legal_moves()

        # Mask out illegal moves
        # action_probs is shape [ACTION_SPACE_SIZE]
        # We'll only use the legal ones
        for mv in legal_moves:
            # convert ( (sr,sc),(er,ec) ) or ( (sr,sc),(er,ec),promo ) -> action_index
            child_index = to_action_index(mv)
            # prior for this action
            move_prior = action_probs[child_index].item()

            # clone engine
            next_engine = self.engine_state.clone()
            next_engine.apply_move(mv)
            child_node = MCTSNode(next_engine, move_prior)
            self.children[child_index] = child_node

def to_action_index(move):
    """Convert an engine move to an action_index via move_encoder."""
    if len(move) == 2:
        # ((sr, sc), (er, ec))
        (sr, sc), (er, ec) = move
        promo_id = 0
    elif len(move) == 3:
        (sr, sc), (er, ec), promo_type = move
        if promo_type is None:
            promo_id = 0
        elif promo_type == 8:  # Rider
            promo_id = 1
        elif promo_type == 2:  # Tarnsman
            # Actually, Tarnsman=2 in your piece code, but let's assume:
            # If we can't easily check, you might store piece_type in your piece_constants or parse
            # We'll do a small mapping:
            # Rider=7, Tarnsman=2 => see how you assigned them. For now:
            # We'll do a manual check:
            if promo_type == 7:
                promo_id = 1
            elif promo_type == 2:
                promo_id = 2
        else:
            promo_id = 2  # fallback
    else:
        raise ValueError("Invalid move format, cannot encode")

    return encode_move(sr, sc, er, ec, promo_id)

def from_action_index(action_index):
    """Decode back to an engine move: ((sr, sc),(er, ec), optional_promo)."""
    sr, sc, er, ec, promo_id = decode_move(action_index)
    if promo_id == 0:
        return ((sr, sc), (er, ec))
    elif promo_id == 1:
        # Rider => piece code is 7 in your engine
        return ((sr, sc), (er, ec), 7)
    elif promo_id == 2:
        # Tarnsman => piece code is 2
        return ((sr, sc), (er, ec), 2)

class MCTS:
    def __init__(self, model, simulations=800, c_puct=1.5, device="cpu"):
        self.model = model
        self.simulations = simulations
        self.c_puct = c_puct
        self.device = device

    def run(self, root_engine):
        """
        root_engine: KaissaGameEngine at the start of the search
        Return: A vector of visit counts (or probabilities) over ACTION_SPACE_SIZE
        """
        # Create root node
        root = MCTSNode(root_engine.clone(), prior=1.0)
        # Expand root
        root_action_probs, root_value = self._inference(root.engine_state)
        root.expand(root_action_probs)

        # If the game is already over at root, just return uniform
        if root.engine_state.is_game_over():
            return np.zeros(ACTION_SPACE_SIZE, dtype=np.float32), 0.0

        # run simulations
        for _ in range(self.simulations):
            node = root
            search_path = [node]

            # 1) Traverse
            while node.is_expanded and not node.engine_state.is_game_over():
                action_index, node = self._select_child(node)
                search_path.append(node)

            # 2) Expand + Evaluate
            value = 0.0
            if not node.engine_state.is_game_over():
                action_probs, value = self._inference(node.engine_state)
                node.expand(action_probs)
            else:
                # game over => value = 1 for winner, -1 for loser
                winner = node.engine_state.get_winner()
                if winner is not None:
                    # current move was by the *previous* player, so let's see
                    # if node.engine_state.is_yellow_turn is True => last move was red
                    # If last mover was the winner => value=1 else -1
                    # This logic can get tricky. We'll do a simpler approach:
                    # If winner == "Yellow" => +1 if last move was Yellow, else -1
                    last_mover = "Red" if node.engine_state.is_yellow_turn else "Yellow"
                    if winner == last_mover:
                        value = 1.0
                    else:
                        value = -1.0
                else:
                    # Possibly draw or no winner
                    value = 0.0

            # 3) Backprop
            self._backpropagate(search_path, value)

        # Compute final policy from root visits
        visits = np.zeros(ACTION_SPACE_SIZE, dtype=np.float32)
        for action_idx, child in root.children.items():
            visits[action_idx] = child.visit_count

        return visits, root.q_value

    def _select_child(self, node):
        """Select the child that maximizes UCB."""
        best_score = -999999
        best_action = None
        best_child = None

        # sum of visits for parent
        parent_visits = max(1, node.visit_count)

        for action_index, child in node.children.items():
            # UCB: Q + U
            q = child.q_value
            u = self.c_puct * child.prior * math.sqrt(parent_visits) / (1 + child.visit_count)
            score = q + u
            if score > best_score:
                best_score = score
                best_action = action_index
                best_child = child

        return best_action, best_child

    def _inference(self, engine_state):
        """Run the neural net on the board -> (policy, value)."""
        self.model.eval()
        board_tensor = encode_board(engine_state.get_state()).unsqueeze(0)  # shape [1,3,10,10]
        board_tensor = board_tensor.to(self.device)
        with torch.no_grad():
            policy_logits, value = self.model(board_tensor)
            policy = torch.softmax(policy_logits, dim=1)[0]  # shape [ACTION_SPACE_SIZE]
        return policy, value.item()

    def _backpropagate(self, search_path, value):
        """
        search_path: list of MCTSNodes from root to leaf
        value: final evaluation from leaf perspective
        We need to alternate signs if needed, because each node is for a different player
        """
        # In a two-player setting, the value flips each move
        # But in Kaissa, is_yellow_turn flips after each move
        # We'll track a "player sign" as we go up
        for i, node in enumerate(search_path):
            node.visit_count += 1
            node.value_sum += value
            # Flip value for the next parent
            value = -value

