"""Benchmark an agent over many seeded episodes.

Usage:  python bench.py [n_episodes] [opponent]
"""
import statistics
import sys

from kaggle_environments import make


def run(n=20, opponent="starter", agent="main.py"):
    scores, wins = [], 0
    for seed in range(n):
        env = make("kaggriculture", configuration={"episodeSteps": 720, "seed": seed})
        env.run([agent, opponent])
        me, opp = (s.reward for s in env.steps[-1])
        scores.append(me)
        wins += me > opp
    return scores, wins


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    opponent = sys.argv[2] if len(sys.argv) > 2 else "starter"

    scores, wins = run(n, opponent)
    print(
        f"vs {opponent}  n={n}\n"
        f"  mean   ${statistics.mean(scores):>10,.0f}\n"
        f"  median ${statistics.median(scores):>10,.0f}\n"
        f"  stdev  ${statistics.stdev(scores):>10,.0f}\n"
        f"  range  ${min(scores):>10,.0f} .. ${max(scores):,.0f}\n"
        f"  winrate {wins}/{n}",
        file=sys.stderr,
    )
