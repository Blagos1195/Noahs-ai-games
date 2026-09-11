import numpy as np
import torch
import torch.nn as nn


class DynamicDQN(nn.Module):

    def __init__(
        self,
        num_conv_layers=2,
        num_fc_layers=1,
        conv_channels=32,
        fc_dim=128,
        rows=6,
        cols=7,
    ):
        super(DynamicDQN, self).__init__()
        self.rows = rows
        self.cols = cols

        # 2 channels: Channel 0 = 'O' tokens, Channel 1 = 'X' tokens
        layers = []
        in_channels = 2
        for _ in range(num_conv_layers):
            layers.append(
                nn.Conv2d(
                    in_channels, conv_channels, kernel_size=3, padding=1
                )
            )
            layers.append(nn.ReLU())
            in_channels = conv_channels

        self.conv = nn.Sequential(*layers)

        conv_out_size = conv_channels * rows * cols

        fc_layers = []
        in_dim = conv_out_size
        for _ in range(num_fc_layers):
            fc_layers.append(nn.Linear(in_dim, fc_dim))
            fc_layers.append(nn.ReLU())
            in_dim = fc_dim

        fc_layers.append(
            nn.Linear(in_dim, cols)
        )  # Output Q-value per column action
        self.fc = nn.Sequential(*fc_layers)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def board_to_tensor(board):
    rows = len(board)
    cols = len(board[0])
    tensor = np.zeros((2, rows, cols), dtype=np.float32)

    for r in range(rows):
        for c in range(cols):
            cell = board[r][c].strip()
            if cell == "O":
                tensor[0, r, c] = 1.0
            elif cell == "X":
                tensor[1, r, c] = 1.0

    return torch.tensor(tensor, dtype=torch.float32).unsqueeze(0)