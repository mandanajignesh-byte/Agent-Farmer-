"""Score a candidate against a league of opponents, robustly.

The old fitness was the mean money margin against one opponent - champion.py,
which is our own previous version. Three things are wrong with that, and all
three are visible in this project's history.

  Dollars are not the metric. The leaderboard is Elo over wins: a $1 win and a
  $50,000 win score the same. Optimising mean dollars lets one blowout outrank
  five narrow losses, and the five losses are what the ladder counts.

  One opponent is not the field. Every gain this project has recorded is a gain
  against its own lineage. v18 beats v16 locally by $2,600 and scored WORSE on
  the real leaderboard (445.8 against 486.1). A single champion cannot see that.

  Averaging hides the bad matchup. An agent that crushes weak opponents and
  loses to strong ones has a fine mean and a poor rating.

So: bound each game, score against several opponents of known strength, and
combine with a soft minimum so a weak matchup drags the total down instead of
being averaged away.

  per game      u = tanh(margin / SCALE), in (-1, 1)
  per opponent  mean of u over seeds, both seats
  overall       soft-min over opponents

tanh rather than a plain win/loss flag because 12-40 games is a coarse signal:
sign alone barely moves, while tanh still rewards turning a $4,000 loss into a
$500 loss. It just stops a single blowout from dominating.

    python fitness.py                    score main.py against the league
    python fitness.py --noise            measure the noise floor instead
    python fitness.py --seeds 12 --offset 3000
"""
import math
import multiprocessing as mp
import statistics
import sys

from kaggle_environments import make

# A margin this size counts as a decisive win. Set from observed head-to-head
# margins, which run from a few hundred dollars to about $12,000; at $5,000
# tanh returns 0.76, so a clear win scores most of what a blowout would.
SCALE = 5000.0

# Soft-min sharpness. 0 is a plain mean, large values approach the true minimum.
# A hard minimum is too noisy at this many games - one unlucky seed against one
# opponent would decide the whole score - so this leans toward the worst
# matchup without being ruled by it.
SHARPNESS = 4.0

LEAGUE = [
    "league/v10.py",   # 25 tiles, wheat only
    "league/v14.py",   # melon plot
    "league/v16.py",   # marginal crop mix, leaderboard 486.1
    "league/v18.py",   # audit fixes, leaderboard 445.8
    "league/v22.py",   # CARE + parallel feeding, leaderboard 533.5
]

# A saturated opponent carries no gradient. The current agent scores +1.000
# against v10 and v14 - tanh is pinned, so no change it could make would move
# that number, and those games are pure cost during a search. v16 is nearly
# saturated at +0.999. Searching therefore uses only the opponents still close
# enough to discriminate; the full LEAGUE is for validation, where beating a
# weak opponent 100% of the time is still worth confirming rather than
# assuming.
SEARCH_LEAGUE = ["league/v18.py", "league/v22.py"]

SATURATION = 0.99


def _play(job):
    """One episode. Returns our money minus theirs, or None if either crashed."""
    challenger, opponent, seed, swapped, steps = job
    lineup = [opponent, challenger] if swapped else [challenger, opponent]
    env = make("kaggriculture", configuration={"episodeSteps": steps, "seed": seed})
    env.run(lineup)
    rewards = [s.reward for s in env.steps[-1]]
    if any(r is None for r in rewards):
        return None
    mine, theirs = (rewards[1], rewards[0]) if swapped else (rewards[0], rewards[1])
    return mine - theirs


def soft_min(values, sharpness=SHARPNESS):
    """Smooth approximation of min(). Equals the mean at sharpness 0 and
    approaches the true minimum as sharpness grows."""
    if not values:
        return 0.0
    if sharpness <= 0:
        return statistics.mean(values)
    # subtract the min first so exp() cannot overflow
    lo = min(values)
    total = sum(math.exp(-sharpness * (v - lo)) for v in values)
    return lo - math.log(total / len(values)) / sharpness


def score(challenger, seeds, pool, league=None, steps=720):
    """Bounded, league-wide, soft-min score. Higher is better; range (-1, 1)."""
    league = league or LEAGUE
    jobs, index = [], []
    for opponent in league:
        for seed in seeds:
            for swapped in (False, True):
                jobs.append((challenger, opponent, seed, swapped, steps))
                index.append(opponent)

    results = pool.map(_play, jobs)

    per_opponent = {o: [] for o in league}
    raw = {o: [] for o in league}
    crashes = 0
    for opponent, margin in zip(index, results):
        if margin is None:
            crashes += 1
            continue
        per_opponent[opponent].append(math.tanh(margin / SCALE))
        raw[opponent].append(margin)

    means = {o: statistics.mean(v) if v else 0.0 for o, v in per_opponent.items()}
    saturated = [o for o, m in means.items() if abs(m) >= SATURATION]
    return {
        "saturated": saturated,
        "overall": soft_min(list(means.values())),
        "mean": statistics.mean(means.values()) if means else 0.0,
        "worst": min(means, key=means.get) if means else None,
        "per_opponent": means,
        "win_rate": {o: 100 * sum(m > 0 for m in v) / len(v) if v else 0.0
                     for o, v in raw.items()},
        "margin": {o: statistics.mean(v) if v else 0.0 for o, v in raw.items()},
        "crashes": crashes,
        "games": sum(len(v) for v in raw.values()),
    }


def noise_floor(agent_path, seeds, pool, steps=720):
    """An agent against a byte-identical copy of itself.

    NOTE: this measurement is degenerate, and the degeneracy is the point.
    With identical agents, common random numbers and seat swapping, a seed's
    two games are exact mirrors - margin M and -M - so the mean is always
    exactly $0 and the spread is symmetric by construction. It therefore does
    NOT measure the noise floor for comparing two different agents. What it
    measures is seat asymmetry: how much of a game's outcome is decided by
    which chair you sit in. That came out at a stdev of $2,447, ranging to
    $6,689, which is why every comparison here plays both seats.

    For the real noise floor, see EMPIRICAL_SE below."""
    import os
    import shutil
    copy = "_noise_probe.py"
    shutil.copyfile(agent_path, copy)
    try:
        jobs = [(copy, agent_path, s, sw, steps) for s in seeds for sw in (False, True)]
        margins = [m for m in pool.map(_play, jobs) if m is not None]
    finally:
        if os.path.exists(copy):
            os.remove(copy)
    sd = statistics.stdev(margins) if len(margins) > 1 else 0.0
    return {
        "games": len(margins),
        "mean": statistics.mean(margins) if margins else 0.0,
        "stdev": sd,
        "range": (min(margins), max(margins)) if margins else (0, 0),
        "wins": sum(m > 0 for m in margins),
    }


# The real noise floor, measured the only way it can be: run the SAME
# comparison on several independent seed ranges and look at how far the answers
# disagree. The crop fixes against v22 gave +$453, +$2,636, +$1,026 and +$1,010
# over four 40-game ranges - a spread of $942.
#
#   40 games   standard error ~$942    detectable effect ~$1,900
#   160 games  standard error ~$471    detectable effect ~$940
#
# This explains a result that looked contradictory. Those crop fixes measured
# +$1,281 over 160 games: significant by margin (t = 2.7), not significant by
# win count (p = 0.069). Both are right. Counting wins discards magnitude and
# is far less powerful - but the leaderboard counts wins, so it is the metric
# that matters even though it is the blunter one.
#
# That tension is what tanh is for here: bounded like a win, continuous like a
# margin. It also means a single 40-game range can never settle anything worth
# less than about $1,900 a game, which is most of what this project has found.
EMPIRICAL_SE_40_GAMES = 942.0


def _report(result):
    print(f"  overall (soft-min)  {result['overall']:+.4f}", file=sys.stderr)
    print(f"  plain mean          {result['mean']:+.4f}"
          f"   worst matchup: {result['worst']}", file=sys.stderr)
    if result.get("saturated"):
        print(f"  saturated (no gradient, drop when searching): "
              f"{', '.join(result['saturated'])}", file=sys.stderr)
    print(f"  {result['games']} games"
          + (f"   WARNING {result['crashes']} crashed" if result["crashes"] else ""),
          file=sys.stderr)
    print(f"  {'opponent':<18}{'score':>8}{'win rate':>10}{'margin':>12}", file=sys.stderr)
    for o in sorted(result["per_opponent"], key=result["per_opponent"].get):
        print(f"  {o:<18}{result['per_opponent'][o]:>+8.3f}"
              f"{result['win_rate'][o]:>9.0f}%{result['margin'][o]:>+12,.0f}",
              file=sys.stderr)


if __name__ == "__main__":
    args = sys.argv[1:]
    n_seeds, offset, noise = 8, 0, False
    if "--noise" in args:
        noise = True
        args.remove("--noise")
    for flag in ("--seeds", "--offset"):
        if flag in args:
            i = args.index(flag)
            val = int(args[i + 1])
            del args[i:i + 2]
            if flag == "--seeds":
                n_seeds = val
            else:
                offset = val
    challenger = args[0] if args else "main.py"
    seeds = range(offset, offset + n_seeds)

    with mp.Pool(min(8, mp.cpu_count())) as pool:
        if noise:
            n = noise_floor(challenger, seeds, pool)
            print(f"  NOISE FLOOR - {challenger} against an identical copy",
                  file=sys.stderr)
            print(f"  {n['games']} games, true difference is $0", file=sys.stderr)
            print(f"  mean  ${n['mean']:+,.0f}   stdev ${n['stdev']:,.0f}"
                  f"   wins {n['wins']}/{n['games']}", file=sys.stderr)
            print(f"  range ${n['range'][0]:+,.0f} .. ${n['range'][1]:+,.0f}",
                  file=sys.stderr)
            for g in (12, 40, 80, 160):
                print(f"  averaging {g:>3} games -> noise on the mean is about "
                      f"${n['stdev'] / g ** 0.5:,.0f}", file=sys.stderr)
        else:
            print(f"  {challenger} vs the league, seeds {offset}..{offset + n_seeds - 1}",
                  file=sys.stderr)
            _report(score(challenger, seeds, pool))
