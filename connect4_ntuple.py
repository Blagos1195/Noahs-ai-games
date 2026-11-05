from __future__ import annotations

import argparse
import csv
import random
import time
from pathlib import Path

import numpy as np

try:
    from numba import njit
except ImportError:
    def njit(*args, **kwargs):
        def wrap(fn):
            return fn
        return wrap

ROWS = 6
COLS = 7
STRIDE = 7
SHIFTS = (1, 7, 6, 8)
TABLE_SIZE = 3 ** 8
PATTERN_COUNT = 70


def build_patterns(seed=17):
    patterns = []
    seen = set()

    def add(cells):
        cells = tuple(cells)
        if len(cells) != 8 or len(set(cells)) != 8 or any(
            cell < 0 or cell >= ROWS * COLS for cell in cells
        ):
            return
        mirror = tuple((cell // COLS) * COLS + COLS - 1 - cell % COLS
                       for cell in cells)
        for candidate in (cells, mirror):
            key = tuple(sorted(candidate))
            if key not in seen:
                seen.add(key)
                patterns.append(candidate)

    for r in range(ROWS):
        for c in range(COLS - 3):
            add([r * COLS + c + i for i in range(4)] +
                [(r + 1) * COLS + c + i for i in range(4)] if r + 1 < ROWS else [])
    for r in range(ROWS - 3):
        for c in range(COLS):
            add([(r + i) * COLS + c for i in range(4)] +
                [(r + i) * COLS + c + 1 for i in range(4)] if c + 1 < COLS else [])
    rng = random.Random(seed)
    while len(patterns) < PATTERN_COUNT:
        start = rng.randrange(ROWS * COLS)
        cells = [start]
        while len(cells) < 8:
            r, c = divmod(cells[-1], COLS)
            neighbours = []
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1),
                           (-1, -1), (-1, 1), (1, -1), (1, 1)):
                nr, nc = r + dr, c + dc
                idx = nr * COLS + nc
                if 0 <= nr < ROWS and 0 <= nc < COLS and idx not in cells:
                    neighbours.append(idx)
            if not neighbours:
                break
            cells.append(rng.choice(neighbours))
        if len(cells) == 8:
            add(cells)
    return np.asarray(patterns[:PATTERN_COUNT], dtype=np.int16)

PATTERNS = build_patterns()
POWERS = np.asarray([3 ** i for i in range(7, -1, -1)], dtype=np.int32)


@njit(cache=True)
def has_won(bits):
    for shift in SHIFTS:
        pair = bits & (bits >> shift)
        if pair & (pair >> (2 * shift)):
            return True
    return False


@njit(cache=True)
def legal_mask(heights):
    mask = 0
    for c in range(COLS):
        if heights[c] < ROWS:
            mask |= 1 << c
    return mask


@njit(cache=True)
def _pattern_indices(current, opponent, heights, patterns):
    result = np.empty(patterns.shape[0], dtype=np.int32)
    for p in range(patterns.shape[0]):
        idx = 0
        for j in range(8):
            cell = patterns[p, j]
            r = cell // COLS
            c = cell % COLS
            bit = 1 << (c * STRIDE + (ROWS - 1 - r))
            value = 0
            if current & bit:
                value = 1
            elif opponent & bit:
                value = 2
            idx += value * POWERS[j]
        result[p] = idx
    return result


@njit(cache=True)
def _value(current, opponent, heights, patterns, weights):
    indices = _pattern_indices(current, opponent, heights, patterns)
    total = 0.0
    for i in range(indices.size):
        total += weights[i, indices[i]]
    return total


class ConnectFour:
    def __init__(self):
        self.current = 0
        self.opponent = 0
        self.heights = np.zeros(COLS, dtype=np.int8)
        self.moves = 0

    def play(self, column):
        if not 0 <= column < COLS or self.heights[column] >= ROWS:
            raise ValueError("Illegal column")
        bit = 1 << (column * STRIDE + int(self.heights[column]))
        self.current |= bit
        self.heights[column] += 1
        self.moves += 1
        won = has_won(self.current)
        self.current, self.opponent = self.opponent, self.current
        return won, self.moves == ROWS * COLS


def board_cells(game):
    cells = [["" for _ in range(COLS)] for _ in range(ROWS)]
    for col in range(COLS):
        for level in range(int(game.heights[col])):
            bit = 1 << (col * STRIDE + level)
            row = ROWS - 1 - level
            if game.current & bit:
                cells[row][col] = "current"
            elif game.opponent & bit:
                cells[row][col] = "opponent"
    return cells


class NTupleAgent:
    def __init__(self, alpha=0.02, gamma=1.0, seed=1):
        self.weights = np.zeros((len(PATTERNS), TABLE_SIZE), dtype=np.float32)
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.rng = random.Random(seed)
        self.updates = 0

    def value(self, game):
        return float(_value(game.current, game.opponent, game.heights,
                             PATTERNS, self.weights))

    def choose_action(self, game, epsilon=0.0):
        mask = int(legal_mask(game.heights))
        actions = [c for c in range(COLS) if mask & (1 << c)]
        if not actions:
            raise ValueError("No legal moves")
        if self.rng.random() < epsilon:
            return self.rng.choice(actions)
        best_value = -float("inf")
        best = []
        safe = []
        for col in actions:
            bit = 1 << (col * STRIDE + int(game.heights[col]))
            next_current = game.opponent
            next_opponent = game.current | bit
            if has_won(next_opponent):
                return col
            next_heights = game.heights.copy()
            next_heights[col] += 1
            allows_win = False
            for reply in range(COLS):
                if next_heights[reply] >= ROWS:
                    continue
                reply_bit = 1 << (reply * STRIDE + int(next_heights[reply]))
                if has_won(next_current | reply_bit):
                    allows_win = True
                    break
            score = -_value(next_current, next_opponent, next_heights,
                            PATTERNS, self.weights)
            if not allows_win:
                safe.append((score, col))
            if score > best_value + 1e-9:
                best_value, best = score, [col]
            elif abs(score - best_value) <= 1e-9:
                best.append(col)
        if safe:
            best_value = max(score for score, _ in safe)
            best = [col for score, col in safe if abs(score - best_value) <= 1e-9]
        return self.rng.choice(best)

    def update(self, game, target):
        indices = _pattern_indices(game.current, game.opponent, game.heights, PATTERNS)
        current_value = float(self.weights[np.arange(len(PATTERNS)), indices].sum())
        delta = float(target) - current_value
        self.weights[np.arange(len(PATTERNS)), indices] += (
            self.alpha * delta / len(PATTERNS)
        )
        self.updates += 1
        return delta

    def save(self, path, episodes=0):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        with temporary.open("wb") as file:
            np.savez_compressed(file, weights=self.weights, alpha=self.alpha,
                                gamma=self.gamma, updates=self.updates, episodes=episodes,
                                patterns=PATTERNS)
        temporary.replace(path)

    def load(self, path):
        with np.load(path, allow_pickle=False) as data:
            if data["weights"].shape != self.weights.shape:
                raise ValueError("Checkpoint has incompatible weight dimensions")
            self.weights[:] = data["weights"]
            self.alpha = float(data["alpha"])
            self.gamma = float(data["gamma"])
            self.updates = int(data["updates"])


def play_episode(agent, epsilon, train=True, seed=None):
    game = ConnectFour()
    trajectory = []
    turn = 0
    winner = None
    while game.moves < ROWS * COLS:
        state = (game.current, game.opponent, game.heights.copy())
        action = agent.choose_action(game, epsilon)
        won, draw = game.play(action)
        if train:
            trajectory.append((state, turn))
        if won:
            winner = turn
            break
        if draw:
            break
        turn = 1 - turn
    if train:
        for i in range(len(trajectory) - 1, -1, -1):
            (current, opponent, heights), player = trajectory[i]
            state = ConnectFour()
            state.current, state.opponent, state.heights = current, opponent, heights
            if i == len(trajectory) - 1:
                target = 0.0 if winner is None else (1.0 if player == winner else -1.0)
            else:
                (next_current, next_opponent, next_heights), _ = trajectory[i + 1]
                next_state = ConnectFour()
                next_state.current = next_current
                next_state.opponent = next_opponent
                next_state.heights = next_heights
                target = -agent.gamma * agent.value(next_state)
            agent.update(state, target)
    return winner, game.moves

def evaluate(agent, games=200, seed=100000):
    wins = losses = draws = 0
    started = time.perf_counter()
    for i in range(games):
        game = ConnectFour()
        rng = random.Random(seed + i)
        turn = 0
        while game.moves < ROWS * COLS:
            if turn == 0:
                action = agent.choose_action(game, 0.0)
            else:
                legal = [c for c in range(COLS) if game.heights[c] < ROWS]
                action = rng.choice(legal)
            won, draw = game.play(action)
            if won:
                if turn == 0:
                    wins += 1
                else:
                    losses += 1
                break
            if draw:
                draws += 1
                break
            turn = 1 - turn
    elapsed = time.perf_counter() - started
    return {"games": games, "wins": wins, "losses": losses, "draws": draws,
            "win_rate": wins / games, "seconds": elapsed,
            "games_per_second": games / max(elapsed, 1e-9)}


def train(args):
    agent = NTupleAgent(args.alpha, args.gamma, args.seed)
    if args.resume and Path(args.resume).exists():
        agent.load(args.resume)
    start = time.perf_counter()
    log_path = Path(args.log)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    new_log = not log_path.exists()
    with log_path.open("a", newline="") as file:
        writer = csv.writer(file)
        if new_log:
            writer.writerow(["episode", "epsilon", "winner", "moves", "updates",
                             "eval_win_rate", "eval_games_per_second", "seconds"])
        for episode in range(1, args.games + 1):
            progress = (episode - 1) / max(args.games - 1, 1)
            epsilon = args.epsilon_end + (args.epsilon_start - args.epsilon_end) * (
                1.0 - progress)
            winner, moves = play_episode(agent, epsilon, True, args.seed + episode)
            eval_rate = ""
            eval_speed = ""
            if episode % args.eval_every == 0 or episode == args.games:
                result = evaluate(agent, args.eval_games, args.seed + 9000000)
                eval_rate = result["win_rate"]
                eval_speed = result["games_per_second"]
                print(f"Episode {episode}/{args.games} | random-opponent "
                      f"{result['wins']}/{result['games']} wins | "
                      f"{result['games_per_second']:.0f} eval games/s", flush=True)
            writer.writerow([episode, epsilon, "draw" if winner is None else
                             ("first" if winner == 0 else "second"), moves,
                             agent.updates, eval_rate, eval_speed,
                             time.perf_counter() - start])
            if episode % args.save_every == 0 or episode == args.games:
                agent.save(args.output, episode)
                file.flush()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--alpha", type=float, default=0.02)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--epsilon-start", type=float, default=0.2)
    parser.add_argument("--epsilon-end", type=float, default=0.02)
    parser.add_argument("--eval-every", type=int, default=5000)
    parser.add_argument("--eval-games", type=int, default=200)
    parser.add_argument("--save-every", type=int, default=5000)
    parser.add_argument("--resume", default="")
    parser.add_argument("--output", default="models/connect4_ntuple.npz")
    parser.add_argument("--log", default="training_connect4.csv")
    args = parser.parse_args()
    if args.games < 1 or args.eval_every < 1 or args.eval_games < 1:
        parser.error("games, eval-every and eval-games must be positive")
    train(args)


if __name__ == "__main__":
    main()
