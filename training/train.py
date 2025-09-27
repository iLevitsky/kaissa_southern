# training/train.py

import os
import json
import torch
import torch.optim as optim
import torch.nn.functional as F

from kaissa_net.model import KaissaNet
from training.dataset_loader import KaissaDataset
from torch.utils.data import DataLoader

def train_model(
    data_dir,
    output_model_path,
    epochs=5,
    batch_size=16,
    lr=1e-3,
    device="cpu"
):
    """
    data_dir: e.g. "data/self_play_games/v0.0.2"
    """
    # 1) Build dataset from JSON
    dataset = KaissaDataset(data_dir)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = KaissaNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_samples = 0

        for batch in dataloader:
            # batch keys: state_tensor, policy, value
            states = batch["state_tensor"].to(device)          # shape [B,3,10,10]
            target_policy = batch["policy"].to(device)         # shape [B, ACTION_SPACE_SIZE]
            target_value = batch["value"].to(device)           # shape [B]

            optimizer.zero_grad()
            pred_policy, pred_value = model(states)
            # pred_policy shape [B, ACTION_SPACE_SIZE], pred_value shape [B,1]

            # policy loss = cross entropy
            # but target_policy is typically a distribution => can use cross-entropy or MSE
            # We'll do simple cross-entropy:
            # we need to ensure no zero => let's do log-softmax and negative sum
            log_policy = F.log_softmax(pred_policy, dim=1)
            policy_loss = -torch.sum(target_policy * log_policy, dim=1).mean()

            # value loss = MSE
            pred_value = pred_value.squeeze(1)  # shape [B]
            value_loss = F.mse_loss(pred_value, target_value)

            loss = policy_loss + value_loss
            loss.backward()
            optimizer.step()

            total_policy_loss += policy_loss.item() * states.size(0)
            total_value_loss += value_loss.item() * states.size(0)
            total_samples += states.size(0)

        print(f"Epoch {epoch+1}/{epochs}, policy_loss={(total_policy_loss/total_samples):.4f}, value_loss={(total_value_loss/total_samples):.4f}")

    # save
    torch.save(model.state_dict(), output_model_path)
    print(f"Model saved to {output_model_path}")
