import os
import random as r
import numpy as np
import torch
import torch.nn as nn

VS = 1  #1=Player (X) vs. AI (O) 2=AI (X) vs. Random Computer (O) 3=AI (X) vs. AI (O) 4=Player (X) vs. Player (O) (PvP)
ROWS = 6
COLUMNS = 7
VERBOSE = True

MODEL_FILE_X = "connect4_dqn_o.pth"
MODEL_FILE_O = "connect4_dqn_x.pth"

stats = {"X": 0, "O": 0, "ties": 0, "total_games": 0}

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


class DynamicDQN(nn.Module):
    def __init__(self, num_conv=2, num_fc=1, conv_channels=32, fc_dim=128, rows=6, cols=7):
        super().__init__()
        layers = []
        in_ch = 2
        for _ in range(num_conv):
            layers.append(nn.Conv2d(in_ch, conv_channels, kernel_size=3, padding=1))
            layers.append(nn.ReLU())
            in_ch = conv_channels
        self.conv = nn.Sequential(*layers)

        conv_out_size = conv_channels * rows * cols
        fc_layers = []
        in_dim = conv_out_size
        for _ in range(num_fc):
            fc_layers.append(nn.Linear(in_dim, fc_dim))
            fc_layers.append(nn.ReLU())
            in_dim = fc_dim
        fc_layers.append(nn.Linear(in_dim, cols))
        self.fc = nn.Sequential(*fc_layers)

    def forward(self, x):
        feat = self.conv(x)
        feat = feat.view(feat.size(0), -1)
        return self.fc(feat)


ai_models = {}
for symbol, filepath in [("X", MODEL_FILE_X), ("O", MODEL_FILE_O)]:
    model = DynamicDQN().to(device)
    if os.path.exists(filepath):
        model.load_state_dict(torch.load(filepath, map_location=device, weights_only=True))
        model.eval()
        ai_models[symbol] = model
    else:
        ai_models[symbol] = None


def board_to_tensor(board):
    grid = np.zeros((ROWS, COLUMNS), dtype=np.float32)
    for r_idx in range(ROWS):
        for c_idx in range(COLUMNS):
            if board[r_idx][c_idx] == " X ":
                grid[r_idx, c_idx] = 1.0
            elif board[r_idx][c_idx] == " O ":
                grid[r_idx, c_idx] = -1.0

    x_plane = (grid == 1.0).astype(np.float32)
    o_plane = (grid == -1.0).astype(np.float32)
    stacked = np.stack([x_plane, o_plane], axis=0)
    return torch.from_numpy(stacked).unsqueeze(0)


def displayStats(stats):
    total = stats["total_games"]
    if total == 0:
        return
    x_pct = (stats["X"] / total) * 100
    o_pct = (stats["O"] / total) * 100
    tie_pct = (stats["ties"] / total) * 100

    print("\n" + "=" * 32)
    print("      LIFETIME STATISTICS      ")
    print("=" * 32)
    print(f"Total Games Played: {total}")
    print(f"Player X Wins:      {stats['X']} ({x_pct:.1f}%)")
    print(f"Computer O Wins:    {stats['O']} ({o_pct:.1f}%)")
    print(f"Ties/Draws:         {stats['ties']} ({tie_pct:.1f}%)")
    print("=" * 32 + "\n")


def printBoard(board):
    print("  " + ("   ".join(str(i + 1) for i in range(COLUMNS))))
    for row in board:
        print("|" + "|".join(row) + "|")
    print("_" * 29)


def playerInput(board, COLUMNS, player):
    while True:
        try:
            column = int(input(f"Player {player} enter a column (1-7): "))
            if 1 <= column <= COLUMNS and board[0][column - 1] == "   ":
                if VERBOSE:
                    print(f"[VERBOSE] Player selected Column {column} (Index {column-1})")
                return column - 1
            else:
                print("invalid input. Please try again.")
        except ValueError:
            print("invalid input. Please try again.")


def computerInput(board, COLUMNS, player):
    if VS != 4:
        valid_cols = [c for c in range(COLUMNS) if board[0][c] == "   "]
        model = ai_models.get(player)

        if model is not None:
            board_tensor = board_to_tensor(board).to(device)
            with torch.inference_mode():
                q_values = model(board_tensor).squeeze(0).cpu().numpy()

            masked_q = np.full(COLUMNS, -np.inf)
            masked_q[valid_cols] = q_values[valid_cols]
            choice = int(np.argmax(masked_q))

            if VERBOSE:
                print(f"[VERBOSE] AI ({player}) selected Column {choice + 1}")
            return choice
        else:
            choice = r.choice(valid_cols)
            if VERBOSE:
                print(f"[VERBOSE] Random Computer ({player}) picked Column {choice + 1}")
            return choice
    else:
        while True:
            try:
                column = int(input(f"Player {player} enter a column (1-7): "))
                if 1 <= column <= COLUMNS and board[0][column - 1] == "   ":
                    if VERBOSE:
                        print(f"[VERBOSE] Player selected Column {column} (Index {column-1})")
                    return column - 1
                else:
                    print("invalid input. Please try again.")
            except ValueError:
                print("invalid input. Please try again.")


def checkWin(board, player):
    target = f" {player} "

    for r_idx in range(ROWS):
        for c in range(COLUMNS):
            if c <= COLUMNS - 4 and all(board[r_idx][c + i] == target for i in range(4)):
                return "win"

            if r_idx <= ROWS - 4 and all(board[r_idx + i][c] == target for i in range(4)):
                return "win"

            if (
                r_idx <= ROWS - 4
                and c <= COLUMNS - 4
                and all(board[r_idx + i][c + i] == target for i in range(4))
            ):
                return "win"

            if (
                r_idx >= 3
                and c <= COLUMNS - 4
                and all(board[r_idx - i][c + i] == target for i in range(4))
            ):
                return "win"

    if all(board[0][c] != "   " for c in range(COLUMNS)):
        return "tie"

    return "false"


def placeTile(board, column, player):
    for i in range(ROWS):
        if board[ROWS - 1 - i][column] == "   ":
            board[ROWS - 1 - i][column] = " " + player + " "
            if VERBOSE:
                print(f"[VERBOSE] Placed '{player}' at Row {ROWS - 1 - i}, Column {column + 1}")
            break
    return checkWin(board, player)


def playGame():
    board = [["   " for _ in range(COLUMNS)] for _ in range(ROWS)]
    Running = True

    while Running:
        printBoard(board)
        if VS == 1 or VS == 4:
            column = playerInput(board, COLUMNS, "X")
        else:
            column = computerInput(board, COLUMNS, "X")

        result = placeTile(board, column, "X")

        if result == "win":
            printBoard(board)
            print("Player X wins!")
            stats["X"] += 1
            stats["total_games"] += 1
            break
        elif result == "tie":
            printBoard(board)
            print("It's a tie!")
            stats["ties"] += 1
            stats["total_games"] += 1
            break

        printBoard(board)
        column = computerInput(board, COLUMNS, "O")
        result = placeTile(board, column, "O")

        if result == "win":
            printBoard(board)
            print("Player O wins!")
            stats["O"] += 1
            stats["total_games"] += 1
            break
        elif result == "tie":
            printBoard(board)
            print("It's a tie!")
            stats["ties"] += 1
            stats["total_games"] += 1
            break

if __name__ == "__main__":
    while True:
        playGame()
        displayStats(stats)
        again = input("Play another game? (y/n): ").strip().lower()
        if again != "y":
            print("\nThanks for playing!")
            break