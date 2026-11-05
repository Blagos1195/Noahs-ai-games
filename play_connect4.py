import argparse
from pathlib import Path

from connect4_ntuple import ConnectFour, NTupleAgent, ROWS, COLS


def print_board(game, current_player):
    cells = [["." for _ in range(COLS)] for _ in range(ROWS)]
    for c in range(COLS):
        for level in range(int(game.heights[c])):
            bit = 1 << (c * 7 + level)
            r = ROWS - 1 - level
            if game.current & bit:
                cells[r][c] = "X" if current_player == 0 else "O"
            elif game.opponent & bit:
                cells[r][c] = "O" if current_player == 0 else "X"
    print("\n  " + " ".join(str(c + 1) for c in range(COLS)))
    for row in cells:
        print("  " + " ".join(row))
    print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/connect4_ntuple.npz")
    parser.add_argument("--mode", choices=("human", "self-play"), default="human")
    args = parser.parse_args()

    agent = NTupleAgent()
    if Path(args.model).exists():
        agent.load(args.model)
        print(f"Loaded model: {args.model}")
    else:
        print(f"No trained model found at {args.model}. The AI will use its initial policy.")

    while True:
        game = ConnectFour()
        turn = 0
        while game.moves < ROWS * COLS:
            print_board(game, turn)
            if args.mode == "self-play" or turn == 1:
                action = agent.choose_action(game, 0.0)
                print(f"AI chooses column {action + 1}")
            else:
                while True:
                    try:
                        action = int(input("Choose a column (1-7): ")) - 1
                        if 0 <= action < COLS and game.heights[action] < ROWS:
                            break
                    except ValueError:
                        pass
                    print("Invalid or full column.")
            mover = turn
            won, draw = game.play(action)
            turn = 1 - turn
            if won:
                print_board(game, turn)
                print("AI wins." if args.mode == "self-play" or mover == 1 else "You win.")
                break
            if draw:
                print_board(game, turn)
                print("Draw.")
                break
        if input("Play again? (y/n): ").strip().lower() != "y":
            break


if __name__ == "__main__":
    main()
