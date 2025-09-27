# kaissa_net/model.py

import torch
import torch.nn as nn
import torch.nn.functional as F

from kaissa_encoding.move_encoder import ACTION_SPACE_SIZE

class KaissaNet(nn.Module):
    """
    Simple 5-layer CNN for a 3x10x10 Kaissa board.
    Outputs:
      - policy: shape [ACTION_SPACE_SIZE]
      - value: single scalar
    """

    def __init__(self):
        super().__init__()

        # Input shape: (3, 10, 10)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(64, 64, kernel_size=3, padding=1)

        # Flatten => for policy
        self.policy_head = nn.Linear(64 * 10 * 10, ACTION_SPACE_SIZE)

        # Flatten => for value
        self.value_head1 = nn.Linear(64 * 10 * 10, 128)
        self.value_head2 = nn.Linear(128, 1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # x shape = (batch, 3, 10, 10)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))

        # Flatten for dense heads
        x = x.view(x.size(0), -1)  # shape [batch, 6400]

        # Policy output
        policy = self.policy_head(x)  # shape [batch, ACTION_SPACE_SIZE]

        # Value output
        v = F.relu(self.value_head1(x))
        v = torch.tanh(self.value_head2(v))  # output in range [-1, 1]

        return policy, v

# ✅ Ensure import works cleanly
__all__ = ["KaissaNet"]
