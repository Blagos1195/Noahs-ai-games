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
    current = int(current)
    opponent = int(opponent)
    result = np.empty(patterns.shape[0], dtype=np.int32)
    for p in range(patterns.shape[0]):
        idx = 0
        for j in range(8):
            cell = int(patterns[p, j])
            r, c = divmod(cell, COLS)
            bit = int(1 << (c * STRIDE + (ROWS - 1 - r)))
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


@njit(cache=True)
def _negamax(current, opponent, heights, depth, alpha, beta, patterns, weights):
    if depth <= 0:
        return _value(current, opponent, heights, patterns, weights)
    best = -1e30
    found = False
    for offset in range(COLS):
        col = 3 if offset == 0 else (3 - (offset + 1) // 2 if offset % 2 else 3 + offset // 2)
        if heights[col] >= ROWS:
            continue
        found = True
        bit = 1 << (col * STRIDE + int(heights[col]))
        if has_won(current | bit):
            return 100000.0 + depth
        next_heights = heights.copy()
        next_heights[col] += 1
        score = -_negamax(opponent, current | bit, next_heights, depth - 1,
                          -beta, -alpha, patterns, weights)
        if score > best:
            best = score
        if score > alpha:
            alpha = score
        if alpha >= beta:
            break
    return best if found else 0.0


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

    def choose_action(self, game, epsilon=0.0, weights=None):
        weights = self.weights if weights is None else weights
        mask = int(legal_mask(game.heights))
        actions = [c for c in range(COLS) if mask & (1 << c)]
        if not actions:
            raise ValueError("No legal moves")
        if self.rng.random() < epsilon:
            return self.rng.choice(actions)
        scores = []
        for col in actions:
            bit = 1 << (col * STRIDE + int(game.heights[col]))
            next_current = game.opponent
            next_opponent = game.current | bit
            next_heights = game.heights.copy()
            next_heights[col] += 1
            score = -_value(next_current, next_opponent, next_heights,
                            PATTERNS, weights)
            scores.append((score, col))
        best_value = max(score for score, _ in scores)
        best = [col for score, col in scores if abs(score - best_value) <= 1e-9]
        return self.rng.choice(best)

    def choose_search_action(self, game, epsilon=0.0, depth=4, weights=None):
        weights = self.weights if weights is None else weights
        mask = int(legal_mask(game.heights))
        actions = [c for c in range(COLS) if mask & (1 << c)]
        if not actions:
            raise ValueError("No legal moves")
        if self.rng.random() < epsilon:
            return self.rng.choice(actions)
        best_score = -1e30
        best_actions = []
        alpha = -1e30
        for offset in range(COLS):
            col = 3 if offset == 0 else (3 - (offset + 1) // 2 if offset % 2 else 3 + offset // 2)
            if col not in actions:
                continue
            bit = 1 << (col * STRIDE + int(game.heights[col]))
            if has_won(game.current | bit):
                return col
            next_heights = game.heights.copy()
            next_heights[col] += 1
            score = -_negamax(game.opponent, game.current | bit, next_heights,
                              max(0, depth - 1), -1e30, -alpha, PATTERNS, weights)
            if score > best_score + 1e-9:
                best_score = score
                best_actions = [col]
            elif abs(score - best_score) <= 1e-9:
                best_actions.append(col)
            if score > alpha:
                alpha = score
        return self.rng.choice(best_actions)

    def value_from_state(self, current, opponent, heights):
        return float(_value(current, opponent, heights, PATTERNS, self.weights))

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


def play_episode(agent, epsilon, train=True, seed=None, search_depth=1):
    game = ConnectFour()
    rng = random.Random(seed)
    trajectory = []
    turn = rng.randrange(2)
    winner = None
    while game.moves < ROWS * COLS:
        state = (game.current, game.opponent, game.heights.copy())
        action = (agent.choose_action(game, epsilon) if search_depth <= 1 else
                  agent.choose_search_action(game, epsilon, search_depth))
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


def tactical_action(game, rng):
    legal = [c for c in range(COLS) if game.heights[c] < ROWS]
    winning = []
    blocking = []
    for col in legal:
        bit = 1 << (col * STRIDE + int(game.heights[col]))
        if has_won(game.current | bit):
            winning.append(col)
        if has_won(game.opponent | bit):
            blocking.append(col)
    if winning:
        return rng.choice(winning)
    if blocking:
        return rng.choice(blocking)
    best = [c for c in legal if c == 3]
    if best:
        return best[0]
    return rng.choice(legal)


def play_tactical_episode(agent, epsilon, seed, search_depth=1):
    game = ConnectFour()
    rng = random.Random(seed)
    learner_player = rng.randrange(2)
    turn = 0
    trajectory = []
    winner = None
    while game.moves < ROWS * COLS:
        if turn == learner_player:
            state = (game.current, game.opponent, game.heights.copy())
            action = (agent.choose_action(game, epsilon) if search_depth <= 1 else
                      agent.choose_search_action(game, epsilon, search_depth))
            trajectory.append(state)
        else:
            action = tactical_action(game, rng)
        won, draw = game.play(action)
        if won:
            winner = turn
            break
        if draw:
            break
        turn = 1 - turn
    reward = 0.0 if winner is None else (1.0 if winner == learner_player else -1.0)
    for i in range(len(trajectory) - 1, -1, -1):
        current, opponent, heights = trajectory[i]
        state = ConnectFour()
        state.current, state.opponent, state.heights = current, opponent, heights
        target = reward if i == len(trajectory) - 1 else (
            agent.gamma * agent.value_from_state(*trajectory[i + 1])
        )
        agent.update(state, target)
    return winner == learner_player if winner is not None else None, game.moves



def play_pool_episode(agent, opponent_weights, epsilon, seed, search_depth=2):
    game = ConnectFour()
    rng = random.Random(seed)
    learner_player = rng.randrange(2)
    turn = 0
    trajectory = []
    winner = None
    while game.moves < ROWS * COLS:
        if turn == learner_player:
            state = (game.current, game.opponent, game.heights.copy())
            action = (agent.choose_action(game, epsilon) if search_depth <= 1 else
                      agent.choose_search_action(game, epsilon, search_depth))
            trajectory.append(state)
        else:
            action = (agent.choose_action(game, weights=opponent_weights) if search_depth <= 1 else
                      agent.choose_search_action(game, depth=search_depth, weights=opponent_weights))
        won, draw = game.play(action)
        if won:
            winner = turn
            break
        if draw:
            break
        turn = 1 - turn
    reward = 0.0 if winner is None else (1.0 if winner == learner_player else -1.0)
    for i in range(len(trajectory) - 1, -1, -1):
        current, opponent, heights = trajectory[i]
        state = ConnectFour()
        state.current, state.opponent, state.heights = current, opponent, heights
        target = reward if i == len(trajectory) - 1 else (
            agent.gamma * agent.value_from_state(*trajectory[i + 1])
        )
        agent.update(state, target)
    return winner == learner_player if winner is not None else None, game.moves


def evaluate(agent, games=100, seed=100000, search_depth=2):
    evaluator = NTupleAgent(agent.alpha, agent.gamma, seed)
    evaluator.weights[:] = agent.weights
    wins = losses = draws = 0
    started = time.perf_counter()
    for i in range(games):
        game = ConnectFour()
        rng = random.Random(seed + i)
        learner_player = rng.randrange(2)
        turn = 0
        while game.moves < ROWS * COLS:
            if turn == learner_player:
                action = (evaluator.choose_action(game, 0.0) if search_depth <= 1 else
                          evaluator.choose_search_action(game, 0.0, search_depth))
            else:
                action = tactical_action(game, rng)
            won, draw = game.play(action)
            if won:
                if turn == learner_player:
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
            "win_rate": wins / games, "points": wins + 0.5 * draws,
            "seconds": elapsed, "games_per_second": games / max(elapsed, 1e-9)}

def epsilon_for_elapsed(elapsed, duration, epsilon_start, epsilon_end):
    progress = min(max(elapsed / max(duration, 1e-9), 0.0), 1.0)
    return epsilon_end + (epsilon_start - epsilon_end) * (
        0.5 * (1.0 + np.cos(np.pi * progress))
    )


def train(args):
    agent = NTupleAgent(args.alpha, args.gamma, args.seed)
    if args.resume and Path(args.resume).exists():
        agent.load(args.resume)

    start = time.perf_counter()
    duration = args.time_limit_minutes * 60 if args.time_limit_minutes else 0
    deadline = start + duration if duration else None
    eval_interval = args.eval_every_minutes * 60 if args.eval_every_minutes else 0
    save_interval = args.save_every_minutes * 60 if args.save_every_minutes else 0
    next_eval = start + eval_interval if eval_interval else None
    next_save = start + save_interval if save_interval else None

    log_path = Path(args.log)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    new_log = not log_path.exists()
    with log_path.open("a", newline="") as file:
        writer = csv.writer(file)
        if new_log:
            writer.writerow(["episode", "epsilon", "training_type", "result", "moves",
                             "updates", "eval_win_rate", "eval_games_per_second", "seconds"])
            file.flush()

        episode = 0
        last_eval_episode = 0
        while episode < args.games and (deadline is None or time.perf_counter() < deadline):
            episode += 1
            elapsed = time.perf_counter() - start
            if duration:
                epsilon = epsilon_for_elapsed(
                    elapsed, duration, args.epsilon_start, args.epsilon_end
                )
            else:
                progress = (episode - 1) / max(args.games - 1, 1)
                epsilon = args.epsilon_end + (args.epsilon_start - args.epsilon_end) * (
                    0.5 * (1.0 + np.cos(np.pi * progress))
                )

            if (episode + args.seed) % 5 < 3:
                winner, moves = play_episode(agent, epsilon, True, args.seed + episode,
                                             args.search_depth)
                training_type = "search-self-play" if args.search_depth > 1 else "self-play"
                result = "draw" if winner is None else ("first" if winner == 0 else "second")
            else:
                won, moves = play_tactical_episode(agent, epsilon, args.seed + episode,
                                                   args.search_depth)
                training_type = "search-tactical" if args.search_depth > 1 else "tactical"
                result = "draw" if won is None else ("win" if won else "loss")

            now = time.perf_counter()
            eval_due = (
                (eval_interval and now >= next_eval) or
                (not duration and episode % args.eval_every == 0)
            )
            eval_rate = ""
            eval_speed = ""
            if eval_due:
                result_eval = evaluate(agent, args.eval_games, args.seed + 9000000 + episode)
                eval_rate = result_eval["win_rate"]
                eval_speed = result_eval["games_per_second"]
                last_eval_episode = episode
                print(f"Episode {episode} | tactical opponent "
                      f"{result_eval['wins']}W/{result_eval['draws']}D/"
                      f"{result_eval['losses']}L | "
                      f"{result_eval['games_per_second']:.0f} eval games/s", flush=True)
                if eval_interval:
                    next_eval = time.perf_counter() + eval_interval

            writer.writerow([episode, epsilon, training_type, result, moves,
                             agent.updates, eval_rate, eval_speed,
                             time.perf_counter() - start])

            now = time.perf_counter()
            save_due = (
                (save_interval and now >= next_save) or
                (not duration and episode % args.save_every == 0)
            )
            if save_due:
                agent.save(args.output, episode)
                file.flush()
                print(f"Checkpoint saved at episode {episode}", flush=True)
                if save_interval:
                    next_save = time.perf_counter() + save_interval

        if episode and last_eval_episode != episode:
            result_eval = evaluate(agent, args.eval_games, args.seed + 9000000 + episode)
            print(f"Final evaluation | tactical opponent "
                  f"{result_eval['wins']}W/{result_eval['draws']}D/"
                  f"{result_eval['losses']}L | "
                  f"{result_eval['games_per_second']:.0f} eval games/s", flush=True)
            writer.writerow([episode, args.epsilon_end, "evaluation", "", 0,
                             agent.updates, result_eval["win_rate"],
                             result_eval["games_per_second"],
                             time.perf_counter() - start])

        agent.save(args.output, episode)
        file.flush()
    print(f"Training finished after {episode} episodes.", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--alpha", type=float, default=0.02)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--search-depth", type=int, default=1)
    parser.add_argument("--epsilon-start", type=float, default=0.2)
    parser.add_argument("--epsilon-end", type=float, default=0.02)
    parser.add_argument("--eval-every", type=int, default=5000)
    parser.add_argument("--eval-games", type=int, default=200)
    parser.add_argument("--save-every", type=int, default=5000)
    parser.add_argument("--time-limit-minutes", type=float, default=0)
    parser.add_argument("--eval-every-minutes", type=float, default=0)
    parser.add_argument("--save-every-minutes", type=float, default=0)
    parser.add_argument("--resume", default="")
    parser.add_argument("--output", default="models/connect4_ntuple.npz")
    parser.add_argument("--log", default="training_connect4.csv")
    args = parser.parse_args()
    if (args.games < 1 or args.eval_every < 1 or args.eval_games < 1 or
            args.time_limit_minutes < 0 or args.eval_every_minutes < 0 or
            args.save_every_minutes < 0 or args.search_depth < 1 or args.search_depth > 8):
        parser.error("game counts and intervals must be positive or zero for disabled time limits")
    train(args)


if __name__ == "__main__":
    main()
