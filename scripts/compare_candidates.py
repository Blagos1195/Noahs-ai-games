import argparse
import json
import random
import time

from connect4_ntuple import ConnectFour, NTupleAgent, evaluate


def head_to_head(candidate, champion, games, seed, depth):
    candidate_wins = champion_wins = draws = 0
    candidate_agent = NTupleAgent(seed=seed + 1)
    champion_agent = NTupleAgent(seed=seed + 2)
    candidate_agent.weights[:] = candidate.weights
    champion_agent.weights[:] = champion.weights

    for i in range(games):
        game = ConnectFour()
        candidate_player = i % 2
        turn = 0

        while game.moves < 42:
            agent = candidate_agent if turn == candidate_player else champion_agent
            action = agent.choose_search_action(game, depth=depth)
            won, draw = game.play(action)

            if won:
                if turn == candidate_player:
                    candidate_wins += 1
                else:
                    champion_wins += 1
                break
            if draw:
                draws += 1
                break
            turn = 1 - turn

    return {
        "games": games,
        "candidate_wins": candidate_wins,
        "champion_wins": champion_wins,
        "draws": draws,
        "candidate_points": candidate_wins + 0.5 * draws,
        "champion_points": champion_wins + 0.5 * draws,
        "candidate_score_rate": (candidate_wins + 0.5 * draws) / games,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--champion", required=True)
    parser.add_argument("--games", type=int, default=1000)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--seed", type=int, default=700000)
    parser.add_argument("--output", default="comparison.json")
    args = parser.parse_args()

    candidate = NTupleAgent(seed=args.seed)
    candidate.load(args.candidate)
    champion = NTupleAgent(seed=args.seed + 1)
    champion.load(args.champion)

    started = time.perf_counter()
    candidate_tactical = evaluate(candidate, args.games, args.seed, args.depth)
    print("Candidate vs tactical:", json.dumps(candidate_tactical), flush=True)

    champion_tactical = evaluate(champion, args.games, args.seed, args.depth)
    print("Published vs tactical:", json.dumps(champion_tactical), flush=True)

    direct = head_to_head(candidate, champion, args.games, args.seed + 100000,
                          args.depth)
    print("Head-to-head:", json.dumps(direct), flush=True)

    report = {
        "depth": args.depth,
        "seed": args.seed,
        "games_per_test": args.games,
        "candidate_tactical": candidate_tactical,
        "published_tactical": champion_tactical,
        "head_to_head": direct,
        "elapsed_seconds": time.perf_counter() - started,
    }

    with open(args.output, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)


if __name__ == "__main__":
    main()
