from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

import numpy as np
from numba import njit

from connect4_ntuple import (
    COLS, ROWS, STRIDE, PATTERNS, POWERS, NTupleAgent,
    _pattern_indices, has_won, legal_mask
)


@njit(cache=True)
def train_epoch(currents, opponents, heights_all, labels, weights, learning_rate, temperature, train):
    correct = 0
    total = len(labels)
    for sample in range(total):
        current = currents[sample]
        opponent = opponents[sample]
        heights = heights_all[sample]
        label = labels[sample]
        mask = legal_mask(heights)
        actions = np.empty(COLS, dtype=np.int64)
        action_count = 0
        scores = np.empty(COLS, dtype=np.float64)
        indices = np.empty((COLS, PATTERNS.shape[0]), dtype=np.int32)
        for col in range(COLS):
            if mask & (1 << col):
                bit = 1 << (col * STRIDE + int(heights[col]))
                next_current = opponent
                next_opponent = current | bit
                next_heights = heights.copy()
                next_heights[col] += 1
                idx = _pattern_indices(next_current, next_opponent, next_heights, PATTERNS)
                value = 0.0
                for p in range(PATTERNS.shape[0]):
                    value += weights[p, idx[p]]
                score = 100000.0 if has_won(current | bit) else -value
                actions[action_count] = col
                scores[action_count] = score
                indices[action_count] = idx
                action_count += 1
        best = 0
        max_score = scores[0]
        for a in range(1, action_count):
            if scores[a] > max_score:
                max_score = scores[a]
                best = a
        if actions[best] == label:
            correct += 1
        if not train:
            continue
        exp_sum = 0.0
        probabilities = np.empty(action_count, dtype=np.float64)
        for a in range(action_count):
            value = np.exp((scores[a] - max_score) / temperature)
            probabilities[a] = value
            exp_sum += value
        for a in range(action_count):
            probability = probabilities[a] / exp_sum
            target = 1.0 if actions[a] == label else 0.0
            gradient = probability - target
            if not has_won(current | (1 << (int(actions[a]) * STRIDE + int(heights[int(actions[a])])))):
                for p in range(PATTERNS.shape[0]):
                    weights[p, indices[a, p]] += learning_rate * gradient
    return correct / max(total, 1)


def load_dataset(path):
    currents = []
    opponents = []
    all_heights = []
    labels = []
    skipped = 0
    with Path(path).open("r", newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        if not reader.fieldnames:
            raise ValueError("Dataset has no CSV header")
        columns = {name.strip().lower().replace("_", ""): name for name in reader.fieldnames}
        board_column = columns.get("boardflat")
        player_column = columns.get("playertomove")
        if board_column is None or player_column is None:
            raise ValueError(f"Missing board_flat/player_to_move columns: {reader.fieldnames}")
        label_columns = [columns[name] for name in ("9ply", "7ply", "5ply", "3ply", "velenastrong")
                         if name in columns]
        if not label_columns:
            raise ValueError("No supported move-label column found")

        for row in reader:
            try:
                flat = "".join(row[board_column].strip().split(",")).replace(" ", "")
                if len(flat) != ROWS * COLS or any(ch not in "012" for ch in flat):
                    raise ValueError
                player = int(row[player_column])
                if player not in (1, 2):
                    raise ValueError
                label = -1
                for column in label_columns:
                    try:
                        candidate = int(row[column])
                    except (ValueError, TypeError):
                        continue
                    if 0 <= candidate < COLS:
                        label = candidate
                        break
                if label < 0:
                    raise ValueError

                current = 0
                opponent = 0
                heights = np.zeros(COLS, dtype=np.int8)
                occupied_seen = [False] * COLS
                for row_index in range(ROWS):
                    for col in range(COLS):
                        cell = int(flat[row_index * COLS + col])
                        bit = 1 << (col * STRIDE + (ROWS - 1 - row_index))
                        if cell:
                            occupied_seen[col] = True
                            heights[col] += 1
                            if cell == player:
                                current |= bit
                            else:
                                opponent |= bit
                        elif occupied_seen[col]:
                            raise ValueError
                if heights[label] >= ROWS or has_won(current) or has_won(opponent):
                    raise ValueError
                if not (int(legal_mask(heights)) & (1 << label)):
                    raise ValueError
                currents.append(current)
                opponents.append(opponent)
                all_heights.append(heights)
                labels.append(label)
            except (ValueError, TypeError, KeyError):
                skipped += 1

    if len(labels) < 10000:
        raise ValueError(f"Only {len(labels)} usable positions; refusing to train")
    print(f"Loaded {len(labels):,} positions; skipped {skipped:,}", flush=True)
    return (
        np.asarray(currents, dtype=np.int64),
        np.asarray(opponents, dtype=np.int64),
        np.asarray(all_heights, dtype=np.int8),
        np.asarray(labels, dtype=np.int8),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model", default="models/connect4_ntuple.npz")
    parser.add_argument("--output", default="models/candidates/dataset-pretrained.npz")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=0.00008)
    parser.add_argument("--temperature", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()

    if args.epochs < 1 or args.learning_rate <= 0 or args.temperature <= 0:
        parser.error("epochs, learning rate and temperature must be positive")

    currents, opponents, heights, labels = load_dataset(args.dataset)
    rng = np.random.default_rng(args.seed)
    order = rng.permutation(len(labels))
    validation = order[::10]
    training = order[1::10] if len(order) < 2 else order[np.setdiff1d(order, validation, assume_unique=True)]
    agent = NTupleAgent(seed=args.seed)
    agent.load(args.model)

    best_accuracy = -1.0
    best_weights = agent.weights.copy()
    for epoch in range(1, args.epochs + 1):
        rng.shuffle(training)
        train_accuracy = train_epoch(
            currents[training], opponents[training], heights[training], labels[training],
            agent.weights, args.learning_rate, args.temperature, True
        )
        validation_accuracy = train_epoch(
            currents[validation], opponents[validation], heights[validation], labels[validation],
            agent.weights, args.learning_rate, args.temperature, False
        )
        print(
            f"Epoch {epoch}/{args.epochs} | training move accuracy {train_accuracy:.4f} | "
            f"validation move accuracy {validation_accuracy:.4f}",
            flush=True
        )
        if validation_accuracy > best_accuracy:
            best_accuracy = validation_accuracy
            best_weights = agent.weights.copy()
            agent.save(args.output, epoch * len(training))
        else:
            print("Validation accuracy did not improve; retaining the best epoch.", flush=True)
            break

    agent.weights[:] = best_weights
    agent.save(args.output, int(args.epochs * len(training)))
    print(f"Dataset pretraining saved to {args.output}", flush=True)


if __name__ == "__main__":
    main()
