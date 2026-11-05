# Noah's AI Games

## Connect Four

The browser version is served from the repository root through GitHub Pages. It runs the same Python game engine and n-tuple agent in Pyodide. If `models/connect4_ntuple.npz` is present, the browser loads it; otherwise the AI uses the tactical baseline until a trained checkpoint is added.

The Connect Four trainer now uses a learned n-tuple value function with temporal-difference self-play. It does not use Expectimax or a full game-tree search.

Install:

```bash
python -m pip install -r requirements.txt
```

Train from scratch:

```bash
python main.py --games 100000 --eval-every 5000
```

Resume a checkpoint:

```bash
python main.py --games 100000 --resume models/connect4_ntuple.npz
```

The default checkpoint is `models/connect4_ntuple.npz`. Training progress is appended to `training_connect4.csv`. Periodic evaluations play against a random opponent. Checkpoints are written atomically by NumPy's archive writer.

Play against the model:

```bash
python play_connect4.py
```

Play AI against itself:

```bash
python play_connect4.py --mode self-play
```

The policy includes a one-ply tactical safety check: it takes an immediate win and avoids moves that allow an immediate opponent win when a safe alternative exists. Strategic evaluation is learned from self-play.

The old DQN implementation was removed from the main training entrypoints because its replay-buffer perspective handling and batched episode resets made its training data unreliable. The previous neural-network and replay-memory modules are no longer used by the Connect Four trainer.
