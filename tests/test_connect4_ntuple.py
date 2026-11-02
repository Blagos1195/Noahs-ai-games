import numpy as np

from connect4_ntuple import ConnectFour, NTupleAgent, PATTERNS, TABLE_SIZE, has_won


def test_patterns_are_valid_and_sized():
    assert PATTERNS.shape == (70, 8)
    assert np.all(PATTERNS >= 0)
    assert np.all(PATTERNS < 42)
    assert all(len(set(row.tolist())) == 8 for row in PATTERNS)


def test_vertical_win():
    game = ConnectFour()
    for _ in range(3):
        game.play(0)
        game.play(1)
    won, _ = game.play(0)
    assert won


def test_horizontal_win():
    game = ConnectFour()
    for col in (0, 0, 1, 1, 2, 2):
        game.play(col)
    won, _ = game.play(3)
    assert won


def test_diagonal_win():
    game = ConnectFour()
    for col in (0, 1, 1, 2, 3, 2, 2, 3, 4, 3):
        game.play(col)
    won, _ = game.play(3)
    assert won


def test_full_column_rejected():
    game = ConnectFour()
    for _ in range(6):
        game.play(0)
    try:
        game.play(0)
    except ValueError:
        pass
    else:
        raise AssertionError("Full column accepted")


def test_agent_only_chooses_legal_moves():
    agent = NTupleAgent(seed=4)
    game = ConnectFour()
    for _ in range(6):
        game.play(0)
    for _ in range(20):
        action = agent.choose_action(game, epsilon=1.0)
        assert 0 <= action < 7
        assert game.heights[action] < 6


def test_checkpoint_round_trip(tmp_path):
    agent = NTupleAgent(seed=3)
    agent.weights[0, 10] = 0.75
    path = tmp_path / "model.npz"
    agent.save(path, 123)
    restored = NTupleAgent()
    restored.load(path)
    assert restored.weights[0, 10] == 0.75
    assert restored.updates == agent.updates


def test_agent_takes_immediate_win():
    agent = NTupleAgent(seed=1)
    game = ConnectFour()
    game.current = sum(1 << (c * 7) for c in (0, 1, 2))
    game.heights[:] = [1, 1, 1, 0, 0, 0, 0]
    assert agent.choose_action(game) == 3


def test_agent_blocks_immediate_opponent_win():
    agent = NTupleAgent(seed=1)
    game = ConnectFour()
    game.current = 1 << (4 * 7)
    game.opponent = sum(1 << (c * 7) for c in (0, 1, 2))
    game.heights[:] = [1, 1, 1, 0, 1, 0, 0]
    assert agent.choose_action(game) == 3
