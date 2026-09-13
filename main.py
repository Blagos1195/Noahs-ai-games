import os
import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from neural_network import NeuralNetwork
from replay_memory import TensorReplayBuffer

# =========================================================================
# Hyperparameters & Config
# =========================================================================
ROWS = 6
COLUMNS = 7

PARALLEL_GAMES = None
BATCH_SIZE = 256
LEARNING_RATE = 0.0005
GAMMA = 0.99
MEMORY_CAPACITY = 50000
TOTAL_EPISODES = 200000
LOG_INTERVAL = 2000
OPENING_MOVES_LIMIT = 6

SAVE_DIR = r"G:\My Drive\CODE"
# =========================================================================

if torch.cuda.is_available():
    device = torch.device("cuda")
    if PARALLEL_GAMES is None:
        PARALLEL_GAMES = 2048
        BATCH_SIZE = 1024
    torch.set_float32_matmul_precision("high")
    print(f"[HARDWARE] Running on NVIDIA GPU (CUDA). Environments: {PARALLEL_GAMES}")
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = torch.device("mps")
    if PARALLEL_GAMES is None:
        PARALLEL_GAMES = 1024
        BATCH_SIZE = 512
    print(f"[HARDWARE] Running on Apple Silicon (MPS). Environments: {PARALLEL_GAMES}")
else:
    device = torch.device("cpu")
    if PARALLEL_GAMES is None:
        PARALLEL_GAMES = 256
        BATCH_SIZE = 256
    num_cores = (os.cpu_count() // 2) if os.cpu_count() else 4
    torch.set_num_threads(max(1, num_cores))
    torch.set_num_interop_threads(1)
    print(f"[HARDWARE] Running on CPU ({torch.get_num_threads()} physical threads). Environments: {PARALLEL_GAMES}")

WIN_KERNELS = torch.zeros((4, 1, 4, 4), dtype=torch.float32, device=device)
WIN_KERNELS[0, 0, 0, :] = 1.0
WIN_KERNELS[1, 0, :, 0] = 1.0
WIN_KERNELS[2, 0, torch.arange(4), torch.arange(4)] = 1.0
WIN_KERNELS[3, 0, torch.arange(4), 3 - torch.arange(4)] = 1.0

OPENING_PATTERNS = torch.tensor([
    [3, 3, 2, 4, 3, 3, 3],
    [0, 1, 5, 6, 0, 1, 6],
    [3, 2, 4, 1, 5, 0, 6],
    [0, 1, 2, 3, 4, 5, 6],
], dtype=torch.long, device=device)


def vectorized_check_wins(boards, player_channel):
    mask = boards[:, player_channel:player_channel+1, :, :]
    conv_res = F.conv2d(mask, WIN_KERNELS)
    return (conv_res == 4.0).flatten(1).any(dim=1)


def vectorized_apply_moves(boards, player_channel, actions):
    batch_idx = torch.arange(PARALLEL_GAMES, device=device)
    col_heights = (boards.sum(dim=1) != 0).sum(dim=1)
    target_rows = (ROWS - 1) - col_heights[batch_idx, actions]
    boards[batch_idx, player_channel, target_rows, actions] = 1.0


def resolve_play_styles(num_games, device):
    probs = torch.tensor([0.25, 0.25, 0.50], device=device)
    mode_x = torch.multinomial(probs, num_games, replacement=True)
    mode_o = torch.multinomial(probs, num_games, replacement=True)

    invalid_pairings = (mode_x < 2) & (mode_o < 2)
    mode_x[invalid_pairings] = 2
    mode_o[invalid_pairings] = 2

    opening_strat_x = torch.randint(0, 4, (num_games,), device=device)
    opening_strat_o = torch.randint(0, 4, (num_games,), device=device)

    return mode_x, mode_o, opening_strat_x, opening_strat_o


@torch.inference_mode()
def select_actions_vectorized(net_x, net_o, swap_sides, boards, mode, opening_strat, move_count, epsilon):
    col_full = (boards[:, 0, 0, :] + boards[:, 1, 0, :]) > 0
    valid_mask = (~col_full).float()

    q_values = torch.zeros((PARALLEL_GAMES, COLUMNS), device=device)
    normal_idx = (~swap_sides).nonzero(as_tuple=True)[0]
    swapped_idx = swap_sides.nonzero(as_tuple=True)[0]

    if len(normal_idx) > 0:
        q_values[normal_idx] = net_x(boards[normal_idx])
    if len(swapped_idx) > 0:
        q_values[swapped_idx] = net_o(boards[swapped_idx])

    q_values[col_full] = -float('inf')

    random_acts = torch.multinomial(valid_mask, 1).squeeze(1)
    greedy_acts = torch.argmax(q_values, dim=1)
    select_rand = torch.rand(PARALLEL_GAMES, device=device) < epsilon
    neural_acts = torch.where(select_rand, random_acts, greedy_acts)

    pure_random_acts = torch.multinomial(valid_mask, 1).squeeze(1)

    if move_count < OPENING_MOVES_LIMIT:
        preferred_cols = OPENING_PATTERNS[opening_strat, move_count % COLUMNS]
        is_preferred_valid = ~col_full[torch.arange(PARALLEL_GAMES, device=device), preferred_cols]
        opening_acts = torch.where(is_preferred_valid, preferred_cols, pure_random_acts)
    else:
        opening_acts = neural_acts

    actions = torch.where(mode == 0, pure_random_acts, neural_acts)
    actions = torch.where(mode == 1, opening_acts, actions)

    return actions


def get_clean_state_dict(model):
    if hasattr(model, "_orig_mod"):
        return model._orig_mod.state_dict()
    return model.state_dict()


def save_checkpoint(model_x, model_o):
    os.makedirs(SAVE_DIR, exist_ok=True)
    x_path = os.path.join(SAVE_DIR, "model_x.pth")
    o_path = os.path.join(SAVE_DIR, "model_o.pth")
    
    torch.save(get_clean_state_dict(model_x), x_path)
    torch.save(get_clean_state_dict(model_o), o_path)
    print(f"\n[INFO] Checkpoint saved successfully to {x_path} and {o_path}")


def train():
    os.makedirs(SAVE_DIR, exist_ok=True)
    x_path = os.path.join(SAVE_DIR, "model_x.pth")
    o_path = os.path.join(SAVE_DIR, "model_o.pth")

    model_x = NeuralNetwork(ROWS, COLUMNS).to(device)
    model_o = NeuralNetwork(ROWS, COLUMNS).to(device)

    # Load existing checkpoints if present
    if os.path.exists(x_path) and os.path.exists(o_path):
        model_x.load_state_dict(torch.load(x_path, map_location=device))
        model_o.load_state_dict(torch.load(o_path, map_location=device))
        print(f"[INFO] Resuming training from checkpoints in {SAVE_DIR}")
    else:
        print(f"[INFO] No saved checkpoints found. Starting fresh training run.")

    model_x = NeuralNetwork(ROWS, COLUMNS).to(device)
    model_o = NeuralNetwork(ROWS, COLUMNS).to(device)

    target_x = NeuralNetwork(ROWS, COLUMNS).to(device)
    target_o = NeuralNetwork(ROWS, COLUMNS).to(device)
    target_x.load_state_dict(get_clean_state_dict(model_x))
    target_o.load_state_dict(get_clean_state_dict(model_o))

    optimizer_x = optim.Adam(model_x.parameters(), lr=LEARNING_RATE)
    optimizer_o = optim.Adam(model_o.parameters(), lr=LEARNING_RATE)

    memory_x = TensorReplayBuffer(MEMORY_CAPACITY, device, ROWS, COLUMNS)
    memory_o = TensorReplayBuffer(MEMORY_CAPACITY, device, ROWS, COLUMNS)
    loss_fn = nn.MSELoss()

    boards = torch.zeros((PARALLEL_GAMES, 2, ROWS, COLUMNS), dtype=torch.float32, device=device)
    move_counts = torch.zeros(PARALLEL_GAMES, dtype=torch.long, device=device)

    swap_sides = torch.rand(PARALLEL_GAMES, device=device) < 0.5
    mode_x, mode_o, strat_x, strat_o = resolve_play_styles(PARALLEL_GAMES, device)

    completed_episodes = 0
    wins_model_x_cnt = 0
    wins_model_o_cnt = 0
    draws_cnt = 0

    start_time = time.time()
    print("[INFO] Headless training loop initialized... (Press Ctrl+C at any time to interrupt and save)")

    try:
        while completed_episodes < TOTAL_EPISODES:
            epsilon = max(0.05, 1.0 - (completed_episodes / (TOTAL_EPISODES * 0.7)))

            # --- PLAYER X TURN ---
            states_before_x = boards.clone()
            actions_x = select_actions_vectorized(
                model_x, model_o, swap_sides, boards, mode_x, strat_x, move_counts.min().item(), epsilon
            )
            
            vectorized_apply_moves(boards, player_channel=0, actions=actions_x)
            move_counts += 1
            wins_x = vectorized_check_wins(boards, player_channel=0)
            draws_x = (boards.sum(dim=1) != 0).all(dim=-1).all(dim=-1) & ~wins_x

            if wins_x.any():
                done_idx = wins_x.nonzero(as_tuple=True)[0]
                num_wins = len(done_idx)

                x_wins_mask = ~swap_sides[done_idx]
                o_wins_mask = swap_sides[done_idx]
                wins_model_x_cnt += x_wins_mask.sum().item()
                wins_model_o_cnt += o_wins_mask.sum().item()

                memory_x.push_batch(
                    states_before_x[done_idx], actions_x[done_idx],
                    torch.ones(num_wins, device=device), boards[done_idx],
                    torch.ones(num_wins, device=device)
                )

                if 'states_before_o' in locals():
                    memory_o.push_batch(
                        states_before_o[done_idx], actions_o[done_idx],
                        -torch.ones(num_wins, device=device), boards[done_idx],
                        torch.ones(num_wins, device=device)
                    )

                boards[done_idx] = 0.0
                move_counts[done_idx] = 0
                swap_sides[done_idx] = torch.rand(num_wins, device=device) < 0.5
                
                new_mx, new_mo, new_sx, new_so = resolve_play_styles(num_wins, device)
                mode_x[done_idx], mode_o[done_idx] = new_mx, new_mo
                strat_x[done_idx], strat_o[done_idx] = new_sx, new_so
                
                completed_episodes += num_wins

            if draws_x.any():
                draw_idx = draws_x.nonzero(as_tuple=True)[0]
                num_draws = len(draw_idx)
                draws_cnt += num_draws

                boards[draw_idx] = 0.0
                move_counts[draw_idx] = 0
                swap_sides[draw_idx] = torch.rand(num_draws, device=device) < 0.5
                
                new_mx, new_mo, new_sx, new_so = resolve_play_styles(num_draws, device)
                mode_x[draw_idx], mode_o[draw_idx] = new_mx, new_mo
                strat_x[draw_idx], strat_o[draw_idx] = new_sx, new_so

                completed_episodes += num_draws

            # --- PLAYER O TURN ---
            states_before_o = boards.clone()
            actions_o = select_actions_vectorized(
                model_o, model_x, swap_sides, boards, mode_o, strat_o, move_counts.min().item(), epsilon
            )
            
            vectorized_apply_moves(boards, player_channel=1, actions=actions_o)
            move_counts += 1
            wins_o = vectorized_check_wins(boards, player_channel=1)
            draws_o = (boards.sum(dim=1) != 0).all(dim=-1).all(dim=-1) & ~wins_o

            if wins_o.any():
                done_idx = wins_o.nonzero(as_tuple=True)[0]
                num_wins = len(done_idx)

                o_wins_mask = ~swap_sides[done_idx]
                x_wins_mask = swap_sides[done_idx]
                wins_model_o_cnt += o_wins_mask.sum().item()
                wins_model_x_cnt += x_wins_mask.sum().item()

                memory_o.push_batch(
                    states_before_o[done_idx], actions_o[done_idx],
                    torch.ones(num_wins, device=device), boards[done_idx],
                    torch.ones(num_wins, device=device)
                )

                memory_x.push_batch(
                    states_before_x[done_idx], actions_x[done_idx],
                    -torch.ones(num_wins, device=device), boards[done_idx],
                    torch.ones(num_wins, device=device)
                )

                boards[done_idx] = 0.0
                move_counts[done_idx] = 0
                swap_sides[done_idx] = torch.rand(num_wins, device=device) < 0.5
                
                new_mx, new_mo, new_sx, new_so = resolve_play_styles(num_wins, device)
                mode_x[done_idx], mode_o[done_idx] = new_mx, new_mo
                strat_x[done_idx], strat_o[done_idx] = new_sx, new_so

                completed_episodes += num_wins

            if draws_o.any():
                draw_idx = draws_o.nonzero(as_tuple=True)[0]
                num_draws = len(draw_idx)
                draws_cnt += num_draws

                boards[draw_idx] = 0.0
                move_counts[draw_idx] = 0
                swap_sides[draw_idx] = torch.rand(num_draws, device=device) < 0.5
                
                new_mx, new_mo, new_sx, new_so = resolve_play_styles(num_draws, device)
                mode_x[draw_idx], mode_o[draw_idx] = new_mx, new_mo
                strat_x[draw_idx], strat_o[draw_idx] = new_sx, new_so

                completed_episodes += num_draws

            # --- OPTIMIZATION STEP ---
            if memory_x.size >= BATCH_SIZE:
                st, act, rew, nxt_st, dns = memory_x.sample(BATCH_SIZE)
                q_vals = model_x(st).gather(1, act.unsqueeze(1)).squeeze(1)
                with torch.no_grad():
                    max_next_q = target_x(nxt_st).max(1)[0]
                    targets = rew + (1 - dns) * GAMMA * max_next_q
                loss_x = loss_fn(q_vals, targets)
                optimizer_x.zero_grad(set_to_none=True)
                loss_x.backward()
                optimizer_x.step()

                st, act, rew, nxt_st, dns = memory_o.sample(BATCH_SIZE)
                q_vals = model_o(st).gather(1, act.unsqueeze(1)).squeeze(1)
                with torch.no_grad():
                    max_next_q = target_o(nxt_st).max(1)[0]
                    targets = rew + (1 - dns) * GAMMA * max_next_q
                loss_o = loss_fn(q_vals, targets)
                optimizer_o.zero_grad(set_to_none=True)
                loss_o.backward()
                optimizer_o.step()

            if completed_episodes % 250 == 0:
                target_x.load_state_dict(get_clean_state_dict(model_x))
                target_o.load_state_dict(get_clean_state_dict(model_o))

            # --- LOGGING WITH ACCURATE PERCENTAGES ---
            if completed_episodes % LOG_INTERVAL == 0 and completed_episodes > 0:
                elapsed = time.time() - start_time
                gps = completed_episodes / elapsed
                
                total_logged_games = wins_model_x_cnt + wins_model_o_cnt + draws_cnt
                
                if total_logged_games > 0:
                    x_win_pct = (wins_model_x_cnt / total_logged_games) * 100
                    o_win_pct = (wins_model_o_cnt / total_logged_games) * 100
                    draw_pct = (draws_cnt / total_logged_games) * 100
                else:
                    x_win_pct = o_win_pct = draw_pct = 0.0
                
                print(f"Eps: {completed_episodes:6d}/{TOTAL_EPISODES} | Speed: {gps:5.1f} GPS | "
                      f"Win X: {x_win_pct:5.1f}% | Win O: {o_win_pct:5.1f}% | Tie: {draw_pct:5.1f}% | Eps: {epsilon:.2f}")
                
                wins_model_x_cnt = 0
                wins_model_o_cnt = 0
                draws_cnt = 0

    except KeyboardInterrupt:
        print("\n[NOTICE] Interrupted by user (Ctrl+C / Stop pressed). Saving progress...")
    finally:
        save_checkpoint(model_x, model_o)


if __name__ == "__main__":
    train()