"""Head-to-head agent comparison with paired seeds and a significance test.

    python bench.py main.py champion.py [n_seeds] [seed_offset]

The leaderboard is Elo over wins, not money - a $1 win scores the same as a
$50,000 win - so this measures win rate, not dollars.

Two variance-reduction tricks make small differences detectable despite a
per-episode stdev of over $1,000:

  common random numbers - both agents play the *same* seeds, so weather, weed
  spawns and shop unlocks are identical for each and largely cancel out of the
  comparison.

  seat swapping - every seed is played twice with the agents' positions
  exchanged, cancelling any first-player advantage.

Tune on one seed range and validate on a different one (pass a seed_offset), or
a long enough search will fit the quirks of the tuning seeds rather than the
game.
"""
import math
import statistics
import sys

from kaggle_environments import make


def two_sided_binomial_p(wins, decisive):
    """P(result at least this lopsided | both agents equally strong)."""
    if decisive == 0:
        return 1.0
    tail = min(wins, decisive - wins)
    cumulative = sum(math.comb(decisive, i) for i in range(tail + 1)) / 2**decisive
    return min(1.0, 2 * cumulative)


def duel(challenger, champion, n_seeds=20, seed_offset=0, steps=720):
    wins = losses = ties = errors = 0
    margins = []

    for seed in range(seed_offset, seed_offset + n_seeds):
        for swapped in (False, True):
            lineup = [champion, challenger] if swapped else [challenger, champion]
            env = make("kaggriculture", configuration={"episodeSteps": steps, "seed": seed})
            env.run(lineup)

            rewards = [s.reward for s in env.steps[-1]]
            if any(r is None for r in rewards):
                errors += 1
                continue

            mine, theirs = (rewards[1], rewards[0]) if swapped else (rewards[0], rewards[1])
            margins.append(mine - theirs)
            if mine > theirs:
                wins += 1
            elif mine < theirs:
                losses += 1
            else:
                ties += 1

    return {
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "errors": errors,
        "games": wins + losses + ties,
        "mean_margin": statistics.mean(margins) if margins else 0.0,
        "p": two_sided_binomial_p(wins, wins + losses),
    }


def report(result, challenger, champion):
    games = result["games"]
    decisive = result["wins"] + result["losses"]
    rate = 100 * result["wins"] / decisive if decisive else 0.0
    p = result["p"]

    if p >= 0.05:
        verdict = "NOT SIGNIFICANT - indistinguishable, do not ship"
    elif result["wins"] > result["losses"]:
        verdict = "SIGNIFICANT - challenger is better"
    else:
        verdict = "SIGNIFICANT - challenger is WORSE"

    print(
        f"{challenger}  vs  {champion}\n"
        f"  {games} games ({result['wins']}W {result['losses']}L {result['ties']}T)\n"
        f"  win rate     {rate:.1f}%\n"
        f"  mean margin  ${result['mean_margin']:+,.0f}\n"
        f"  p-value      {p:.4f}\n"
        f"  {verdict}"
        + (f"\n  WARNING: {result['errors']} episodes errored" if result["errors"] else ""),
        file=sys.stderr,
    )


if __name__ == "__main__":
    challenger = sys.argv[1] if len(sys.argv) > 1 else "main.py"
    champion = sys.argv[2] if len(sys.argv) > 2 else "champion.py"
    n_seeds = int(sys.argv[3]) if len(sys.argv) > 3 else 20
    seed_offset = int(sys.argv[4]) if len(sys.argv) > 4 else 0

    report(duel(challenger, champion, n_seeds, seed_offset), challenger, champion)
