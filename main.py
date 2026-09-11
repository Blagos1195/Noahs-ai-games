import os
import random as r
import time
from collections import deque
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Number of rows on the Connect 4 board
ROWS = 6

# Number of columns on the Connect 4 board
COLUMNS = 7

# Default Google Drive directory path for saving trained model weights
GDRIVE_DIR = r"G:\My Drive\CODE"

if os.path.exists(GDRIVE_DIR):
    # Directory path used for saving and loading model checkpoints
    MODEL_DIR = GDRIVE_DIR
else:
    MODEL_DIR = "saved_models"
    os.makedirs(MODEL_DIR, exist_ok=True)

# File path for Player X's model weights
MODEL_FILE_X = os.path.join(MODEL_DIR, "connect4_dqn_x.pth")

# File path for Player O's model weights
MODEL_FILE_O = os.path.join(MODEL_DIR, "connect4_dqn_o.pth")

# Number of convolutional layers in the neural network architecture
NUM_CONV_LAYERS = 2

# Number of fully connected layers in the neural network architecture
NUM_FC_LAYERS = 1

# Number of output channels for each convolutional layer
CONV_CHANNELS = 32

# Dimension (number of nodes) of the hidden fully connected layer
FC_DIM = 128

# Size of mini-batches sampled from replay memory during neural network optimization
BATCH_SIZE = 512

# Number of game instances running simultaneously during vectorized training
PARALLEL_GAMES = 512

# Frequency of optimization steps measured in environment interaction steps
TRAIN_EVERY_STEPS = 8

# Learning rate for the Adam optimizer
LEARNING_RATE = 0.0003

# Discount factor for future rewards in the Q-learning update rule
GAMMA = 0.9999

# Maximum number of transitions stored in the experience replay buffer
MEMORY_CAPACITY = 100000

# Probability of assigning a completely random opponent during self-play
RANDOM_OPPONENT_PROB = 0.20

# Total number of game episodes to complete during the full training session
TOTAL_EPISODES = 2000000

# Frequency of console logging outputs measured in completed game episodes
LOG_INTERVAL = 2000

# Number of CPU cores available on the system
NUM_CORES = os.cpu_count()
if NUM_CORES and NUM_CORES > 0:
    torch.set_num_threads(NUM_CORES)
    torch.set_num_interop_threads(NUM_CORES)

if torch.cuda.is_available():
    # PyTorch compute device selected for tensor operations (CUDA GPU, MPS, or CPU)
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print(f"[INFO] Running on compute hardware: {device.type.upper()}")

# Convolutional filter kernels used to detect 4-in-a-row winning alignments
WIN_KERNELS = torch.zeros((4, 1, 4, 4), dtype=torch.float32)
WIN_KERNELS[0, 0, 0, :] = 1.0
WIN_KERNELS[1, 0, :, 0] = 1.0
WIN_KERNELS[2, 0, torch.arange(4), torch.arange(4)] = 1.0
WIN_KERNELS[3, 0, torch.arange(4), 3 - torch.arange(4)] = 1.0
WIN_KERNELS = WIN_KERNELS.to(device)


def check_wins_batch_torch(boards_tensor, player_val):
    player_mask = (boards_tensor == player_val).float().unsqueeze(1)
    conv_res = F.conv2d(player_mask, WIN_KERNELS)
    wins = (conv_res == 4.0).flatten(1).any(dim=1)
    return wins.cpu().numpy()


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


class ReplayBuffer:
    def __init__(self, capacity=MEMORY_CAPACITY):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = r.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.stack(states),
            torch.tensor(actions, dtype=torch.long),
            torch.tensor(rewards, dtype=torch.float32),
            torch.stack(next_states),
            torch.tensor(dones, dtype=torch.float32),
        )

    def __len__(self):
        return len(self.buffer)


def batch_boards_to_tensor(boards):
    x_planes = (boards == 1).astype(np.float32)
    o_planes = (boards == -1).astype(np.float32)
    stacked = np.stack([x_planes, o_planes], axis=1)
    return torch.from_numpy(stacked)


def check_win_single(board, player_val):
    for r_idx in range(ROWS):
        for c_idx in range(COLUMNS - 3):
            if np.all(board[r_idx, c_idx : c_idx + 4] == player_val):
                return True
    for r_idx in range(ROWS - 3):
        for c_idx in range(COLUMNS):
            if np.all(board[r_idx : r_idx + 4, c_idx] == player_val):
                return True
    for r_idx in range(ROWS - 3):
        for c_idx in range(COLUMNS - 3):
            sub = board[r_idx : r_idx + 4, c_idx : c_idx + 4]
            if np.all(np.diagonal(sub) == player_val):
                return True
            if np.all(np.diagonal(np.fliplr(sub)) == player_val):
                return True
    return False


def get_valid_cols_single(board):
    return np.where(board[0] == 0)[0]


class LearningAgent:
    def __init__(self, model_file, player_symbol):
        # Target path for saving and loading the agent's weight file
        self.model_file = model_file

        # Player identity symbol ('X' or 'O')
        self.symbol = player_symbol

        # Primary neural network used for selecting actions and receiving weight updates
        self.net = DynamicDQN(
            NUM_CONV_LAYERS, NUM_FC_LAYERS, CONV_CHANNELS, FC_DIM, ROWS, COLUMNS
        ).to(device)

        # Target network used to stabilize Q-value target calculations
        self.target_net = DynamicDQN(
            NUM_CONV_LAYERS, NUM_FC_LAYERS, CONV_CHANNELS, FC_DIM, ROWS, COLUMNS
        ).to(device)

        if os.path.exists(model_file):
            self.net.load_state_dict(
                torch.load(model_file, map_location=device, weights_only=True)
            )
            print(f"[INFO] Loaded weights for '{player_symbol}' from {model_file}")

        self.target_net.load_state_dict(self.net.state_dict())
        self.net.train()
        self.target_net.eval()

        # Adam optimizer instance managing gradient descent updates for the policy network
        self.optimizer = optim.Adam(self.net.parameters(), lr=LEARNING_RATE)

        # Experience replay buffer instance holding transition tuples
        self.memory = ReplayBuffer(capacity=MEMORY_CAPACITY)

        # Loss function calculating mean squared error between estimated and target Q-values
        self.loss_fn = nn.MSELoss()

        # Counter tracking the total number of environment interactions performed
        self.step_counter = 0

    def select_single_action(self, board, epsilon=0.0):
        valid_cols = get_valid_cols_single(board)
        if r.random() < epsilon:
            return r.choice(valid_cols)
        board_tensor = batch_boards_to_tensor(board[np.newaxis, ...]).to(device)
        with torch.inference_mode():
            q_values = self.net(board_tensor).squeeze(0).cpu().numpy()
        masked_q = np.full(COLUMNS, -np.inf)
        masked_q[valid_cols] = q_values[valid_cols]
        return np.argmax(masked_q)

    def select_batch_actions(self, boards, epsilon, force_random_mask=None):
        num_games = len(boards)
        actions = np.zeros(num_games, dtype=np.int64)
        batch_tensors = batch_boards_to_tensor(boards).to(device)

        with torch.inference_mode():
            q_values_batch = self.net(batch_tensors).cpu().numpy()

        for i in range(num_games):
            valid_cols = np.where(boards[i, 0] == 0)[0]
            if len(valid_cols) == 0:
                actions[i] = 0
                continue
            use_random = (force_random_mask[i] if force_random_mask is not None else False) or (
                r.random() < epsilon
            )
            if use_random:
                actions[i] = r.choice(valid_cols)
            else:
                masked_q = np.full(COLUMNS, -np.inf)
                masked_q[valid_cols] = q_values_batch[i, valid_cols]
                actions[i] = np.argmax(masked_q)

        return actions

    def train_step(self):
        self.step_counter += 1
        if self.step_counter % TRAIN_EVERY_STEPS != 0:
            return
        if len(self.memory) < BATCH_SIZE:
            return

        states, actions, rewards, next_states, dones = self.memory.sample(BATCH_SIZE)
        states = states.to(device)
        actions = actions.to(device)
        rewards = rewards.to(device)
        next_states = next_states.to(device)
        dones = dones.to(device)

        q_values = self.net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            max_next_q = self.target_net(next_states).max(1)[0]
            target_q_values = rewards + (1 - dones) * GAMMA * max_next_q

        loss = self.loss_fn(q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def save(self):
        self.target_net.load_state_dict(self.net.state_dict())
        torch.save(self.net.state_dict(), self.model_file)
        print(f"[INFO] Saved weights to {self.model_file}")


def run_vectorized_training(agent_x, agent_o):
    # Total count of games finished across all parallel workers
    completed_episodes = 0

    # Total completed episode count at the time of the previous log output
    episodes_at_last_log = 0

    # Running count of Player X wins within the current logging interval
    wins_x = 0

    # Running count of Player O wins within the current logging interval
    wins_o = 0

    # Running count of tied games within the current logging interval
    ties = 0

    # Timestamp recording the start of the current logging interval
    interval_start_time = time.time()

    # Timestamp recording the start of the entire training execution
    total_start_time = time.time()

    print(
        f"\n[INFO] Running Headless Self-Play on device '{device.type.upper()}'"
        f" across {PARALLEL_GAMES} environments..."
    )

    # Array storing current board states for all parallel game instances
    boards = np.zeros((PARALLEL_GAMES, ROWS, COLUMNS), dtype=np.int8)

    # Boolean mask flagging whether Player X acts randomly in each parallel game
    is_random_x = np.random.rand(PARALLEL_GAMES) < RANDOM_OPPONENT_PROB

    # Boolean mask flagging whether Player O acts randomly in each parallel game
    is_random_o = np.random.rand(PARALLEL_GAMES) < RANDOM_OPPONENT_PROB

    while completed_episodes < TOTAL_EPISODES:
        # Current exploration rate governing random move probability
        epsilon = max(0.05, 1.0 - (completed_episodes / (TOTAL_EPISODES * 0.8)))

        # Encoded tensor state representation before Player X takes an action
        states_x_before = batch_boards_to_tensor(boards)

        # Selected column choices for Player X across all parallel environments
        cols_x = agent_x.select_batch_actions(boards, epsilon, force_random_mask=is_random_x)

        for i in range(PARALLEL_GAMES):
            col = cols_x[i]
            valid_cols = np.where(boards[i, 0] == 0)[0]
            if len(valid_cols) > 0 and col in valid_cols:
                for r_idx in range(ROWS - 1, -1, -1):
                    if boards[i, r_idx, col] == 0:
                        boards[i, r_idx, col] = 1
                        break

        # Encoded tensor state representation after Player X takes an action
        states_after_x = batch_boards_to_tensor(boards)

        boards_torch = torch.from_numpy(boards).to(device)

        # Boolean array marking which game instances resulted in a Player X win
        x_wins = check_wins_batch_torch(boards_torch, 1)

        for i in range(PARALLEL_GAMES):
            if x_wins[i]:
                agent_x.memory.push(
                    states_x_before[i], cols_x[i], 1.0, states_after_x[i], 1.0
                )
                wins_x += 1
                completed_episodes += 1
                boards[i] = np.zeros((ROWS, COLUMNS), dtype=np.int8)
                is_random_x[i] = r.random() < RANDOM_OPPONENT_PROB
                is_random_o[i] = r.random() < RANDOM_OPPONENT_PROB
            elif len(np.where(boards[i, 0] == 0)[0]) == 0:
                ties += 1
                completed_episodes += 1
                boards[i] = np.zeros((ROWS, COLUMNS), dtype=np.int8)
                is_random_x[i] = r.random() < RANDOM_OPPONENT_PROB
                is_random_o[i] = r.random() < RANDOM_OPPONENT_PROB

        # Encoded tensor state representation before Player O takes an action
        states_o_before = batch_boards_to_tensor(boards)

        # Selected column choices for Player O across all parallel environments
        cols_o = agent_o.select_batch_actions(boards, epsilon, force_random_mask=is_random_o)

        for i in range(PARALLEL_GAMES):
            col = cols_o[i]
            valid_cols = np.where(boards[i, 0] == 0)[0]
            if len(valid_cols) > 0 and col in valid_cols:
                for r_idx in range(ROWS - 1, -1, -1):
                    if boards[i, r_idx, col] == 0:
                        boards[i, r_idx, col] = -1
                        break

        # Encoded tensor state representation after Player O takes an action
        states_after_o = batch_boards_to_tensor(boards)

        boards_torch = torch.from_numpy(boards).to(device)

        # Boolean array marking which game instances resulted in a Player O win
        o_wins = check_wins_batch_torch(boards_torch, -1)

        for i in range(PARALLEL_GAMES):
            if o_wins[i]:
                agent_o.memory.push(
                    states_o_before[i], cols_o[i], 1.0, states_after_o[i], 1.0
                )
                wins_o += 1
                completed_episodes += 1
                boards[i] = np.zeros((ROWS, COLUMNS), dtype=np.int8)
                is_random_x[i] = r.random() < RANDOM_OPPONENT_PROB
                is_random_o[i] = r.random() < RANDOM_OPPONENT_PROB
            elif len(np.where(boards[i, 0] == 0)[0]) == 0:
                ties += 1
                completed_episodes += 1
                boards[i] = np.zeros((ROWS, COLUMNS), dtype=np.int8)
                is_random_x[i] = r.random() < RANDOM_OPPONENT_PROB
                is_random_o[i] = r.random() < RANDOM_OPPONENT_PROB

        agent_x.train_step()
        agent_o.train_step()

        if completed_episodes % 1000 < PARALLEL_GAMES:
            agent_x.target_net.load_state_dict(agent_x.net.state_dict())
            agent_o.target_net.load_state_dict(agent_o.net.state_dict())

        if completed_episodes - episodes_at_last_log >= LOG_INTERVAL:
            # Number of completed games processed during the current interval
            games_in_interval = completed_episodes - episodes_at_last_log

            # Elapsed time in seconds for the current logging interval
            interval_elapsed = time.time() - interval_start_time

            # Calculated processing speed measured in games completed per second
            actual_gps = (
                games_in_interval / interval_elapsed if interval_elapsed > 0 else 0
            )

            # Sum of all games ended (wins and ties) during the logging interval
            total_interval_games = wins_x + wins_o + ties
            if total_interval_games > 0:
                print(
                    f"Game {completed_episodes:6d}/{TOTAL_EPISODES} | "
                    f"X Win: {(wins_x/total_interval_games)*100:5.1f}% | "
                    f"O Win: {(wins_o/total_interval_games)*100:5.1f}% | "
                    f"Tie: {(ties/total_interval_games)*100:4.1f}% | "
                    f"Epsilon: {epsilon:.3f} | "
                    f"Speed: {actual_gps:6.1f} GPS"
                )

            agent_x.save()
            agent_o.save()

            wins_x, wins_o, ties = 0, 0, 0
            episodes_at_last_log = completed_episodes
            interval_start_time = time.time()

    # Total duration of the training session in seconds
    total_time = time.time() - total_start_time
    print(
        f"\n[COMPLETE] Finished {TOTAL_EPISODES} episodes in"
        f" {total_time:.1f}s ({TOTAL_EPISODES / total_time:.1f} overall GPS)."
    )


def print_ascii_board(board):
    print("\n  " + ("   ".join(str(i + 1) for i in range(COLUMNS))))
    for r_idx in range(ROWS):
        row_str = "|"
        for c_idx in range(COLUMNS):
            val = board[r_idx, c_idx]
            char = " X " if val == 1 else (" O " if val == -1 else "   ")
            row_str += char + "|"
        print(row_str)
    print("_" * (COLUMNS * 4 + 1))


def get_human_input(board, symbol):
    valid_cols = get_valid_cols_single(board)
    while True:
        try:
            col = int(input(f"Player {symbol}, choose column (1-{COLUMNS}): "))
            if (col - 1) in valid_cols:
                return col - 1
            print("Invalid column or column is full. Try again.")
        except ValueError:
            print(f"Enter a valid number between 1 and {COLUMNS}.")


def play_interactive_match(agent_x, agent_o, human_player="X"):
    # NumPy array representing the single interactive game board
    board = np.zeros((ROWS, COLUMNS), dtype=np.int8)
    print(f"\n--- Match Started! Human playing as {human_player} ---")

    while True:
        print_ascii_board(board)

        if human_player == "X":
            # Column index chosen by human input for Player X
            col_x = get_human_input(board, "X")
        else:
            # Column index chosen by agent decision for Player X
            col_x = agent_x.select_single_action(board, epsilon=0.0)
            print(f"AI (X) played column {col_x + 1}")

        for r_idx in range(ROWS - 1, -1, -1):
            if board[r_idx, col_x] == 0:
                board[r_idx, col_x] = 1
                break

        if check_win_single(board, 1):
            print_ascii_board(board)
            print("\n*** PLAYER X WINS! ***")
            break
        if len(get_valid_cols_single(board)) == 0:
            print_ascii_board(board)
            print("\n*** GAME TIED! ***")
            break

        print_ascii_board(board)

        if human_player == "O":
            # Column index chosen by human input for Player O
            col_o = get_human_input(board, "O")
        else:
            # Column index chosen by agent decision for Player O
            col_o = agent_o.select_single_action(board, epsilon=0.0)
            print(f"AI (O) played column {col_o + 1}")

        for r_idx in range(ROWS - 1, -1, -1):
            if board[r_idx, col_o] == 0:
                board[r_idx, col_o] = -1
                break

        if check_win_single(board, -1):
            print_ascii_board(board)
            print("\n*** PLAYER O WINS! ***")
            break
        if len(get_valid_cols_single(board)) == 0:
            print_ascii_board(board)
            print("\n*** GAME TIED! ***")
            break


if __name__ == "__main__":
    # LearningAgent instance representing Player X
    agent_x = LearningAgent(MODEL_FILE_X, "X")

    # LearningAgent instance representing Player O
    agent_o = LearningAgent(MODEL_FILE_O, "O")

    print("\nSelect Operating Mode:")
    print("1. Fast Headless Self-Play Training (Background)")
    print("2. Human (X) vs AI (O)")
    print("3. AI (X) vs Human (O)")

    # User menu selection input string
    choice = input("Enter choice (1-3): ").strip()

    if choice == "1":
        try:
            run_vectorized_training(agent_x, agent_o)
        except KeyboardInterrupt:
            print("\n[INTERRUPTED] Saving model checkpoints before exit...")
            agent_x.save()
            agent_o.save()
            print("[INFO] Safety save complete.")
    elif choice == "2":
        play_interactive_match(agent_x, agent_o, human_player="X")
    elif choice == "3":
        play_interactive_match(agent_x, agent_o, human_player="O")
    else:
        print("Invalid choice. Exiting.")