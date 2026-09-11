import random
import torch
import torch.nn as nn
import torch.optim as optim
from neural_network import DynamicDQN, board_to_tensor
from replay_memory import ReplayBuffer

# --- Model Architecture & Hyperparameters Config ---
ROWS = 6
COLUMNS = 7

# Architecture controls (Change hidden layers here)
NUM_CONV_LAYERS = 2  # Number of 2D Convolutional layers
NUM_FC_LAYERS = 1    # Number of Linear/Fully-Connected hidden layers
CONV_CHANNELS = 32   # Feature maps per conv layer
FC_DIM = 128         # Neurons per linear hidden layer

# Training Hyperparameters
BATCH_SIZE = 32
LEARNING_RATE = 0.0005
GAMMA = 0.99          # Discount factor for future rewards
EPSILON_START = 1.0   # Initial random exploration rate (100%)
EPSILON_END = 0.05    # Minimum exploration rate (5%)
EPSILON_DECAY = 0.995 # Decay rate per game
TARGET_UPDATE_FREQ = 10  # Sync target network every N games
NUM_EPISODES = 500    # Total training games

# --- Setup Network Objects ---
policy_net = DynamicDQN(
    num_conv_layers=NUM_CONV_LAYERS,
    num_fc_layers=NUM_FC_LAYERS,
    conv_channels=CONV_CHANNELS,
    fc_dim=FC_DIM,
)
target_net = DynamicDQN(
    num_conv_layers=NUM_CONV_LAYERS,
    num_fc_layers=NUM_FC_LAYERS,
    conv_channels=CONV_CHANNELS,
    fc_dim=FC_DIM,
)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()  # Target network only predicts, never computes gradients

optimizer = optim.Adam(policy_net.parameters(), lr=LEARNING_RATE)
memory = ReplayBuffer(capacity=10000)
loss_fn = nn.MSELoss()


# --- Game Logic Helpers ---
def get_valid_cols(board):
    return [c for c in range(COLUMNS) if board[0][c] == "   "]


def check_win(board, player):
    target = f" {player} "
    for r in range(ROWS):
        for c in range(COLUMNS):
            if c <= COLUMNS - 4 and all(
                board[r][c + i] == target for i in range(4)
            ):
                return True
            if r <= ROWS - 4 and all(
                board[r + i][c] == target for i in range(4)
            ):
                return True
            if (
                r <= ROWS - 4
                and c <= COLUMNS - 4
                and all(board[r + i][c + i] == target for i in range(4))
            ):
                return True
            if (
                r >= 3
                and c <= COLUMNS - 4
                and all(board[r - i][c + i] == target for i in range(4))
            ):
                return True
    return False


def place_tile(board, col, player):
    for r in range(ROWS - 1, -1, -1):
        if board[r][col] == "   ":
            board[r][col] = f" {player} "
            break


def select_action(board, epsilon):
    valid_cols = get_valid_cols(board)

    # Exploration (Random move)
    if random.random() < epsilon:
        return random.choice(valid_cols)

    # Exploitation (Neural Network prediction)
    state_tensor = board_to_tensor(board)
    with torch.no_grad():
        q_values = policy_net(state_tensor).squeeze(0)

    # Mask full columns with negative infinity
    masked_q = torch.full_like(q_values, float("-inf"))
    for c in valid_cols:
        masked_q[c] = q_values[c]

    return torch.argmax(masked_q).item()


# --- Single Optimization Step ---
def train_step():
    if len(memory) < BATCH_SIZE:
        return  # Wait until enough moves are stored in memory

    # 1. Sample batched tensors directly from ReplayBuffer
    states, actions, rewards, next_states, dones = memory.sample(BATCH_SIZE)

    # 2. Get current Q-values predicted by policy_net for taken actions
    q_values = (
        policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
    )

    # 3. Calculate target Q-values using target_net (Bellman Equation)
    with torch.no_grad():
        max_next_q = target_net(next_states).max(1)[0]
        target_q_values = rewards + (1 - dones) * GAMMA * max_next_q

    # 4. Compute Loss and run backpropagation
    loss = loss_fn(q_values, target_q_values)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


# --- Main Training Loop ---
epsilon = EPSILON_START

for episode in range(1, NUM_EPISODES + 1):
    board = [["   " for _ in range(COLUMNS)] for _ in range(ROWS)]
    done = False

    while not done:
        valid_cols = get_valid_cols(board)
        if not valid_cols:
            break

        # State before move
        state_tensor = board_to_tensor(board)

        # Agent O Turn
        action = select_action(board, epsilon)
        place_tile(board, action, "O")

        # Evaluate outcome
        if check_win(board, "O"):
            reward = 1.0
            done = True
        elif len(get_valid_cols(board)) == 0:
            reward = 0.0
            done = True
        else:
            # Opponent (X) plays a random move
            x_col = random.choice(get_valid_cols(board))
            place_tile(board, x_col, "X")

            if check_win(board, "X"):
                reward = -1.0
                done = True
            elif len(get_valid_cols(board)) == 0:
                reward = 0.0
                done = True
            else:
                reward = 0.0  # Game continues

        # State after move & opponent turn
        next_state_tensor = board_to_tensor(board)

        # Store transition tuple in Replay Memory
        memory.push(
            state_tensor,
            action,
            reward,
            next_state_tensor,
            1.0 if done else 0.0,
        )

        # Run weight optimization step
        train_step()

    # Decay Epsilon exploration rate after each episode
    epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)

    # Periodically update target network weights
    if episode % TARGET_UPDATE_FREQ == 0:
        target_net.load_state_dict(policy_net.state_dict())
        print(
            f"Episode {episode}/{NUM_EPISODES} | Epsilon: {epsilon:.3f} | Memory Size: {len(memory)}"
        )

# Save trained weights
torch.save(policy_net.state_dict(), "connect4_dqn.pth")
print("\nTraining complete! Weights saved to connect4_dqn.pth")