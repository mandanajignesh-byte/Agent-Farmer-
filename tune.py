"""Hill-climb main.PARAMS against the league, using fitness.py's own score.

    python tune.py [iterations] [seeds] [seed_offset]

Used to fitness against one champion snapshot in dollars. That is the exact
thing fitness.py was written to replace - one opponent is not the field, and
dollars are not the metric the ladder counts - so the search now optimises
the same bounded, soft-min, multi-opponent score fitness.py reports, against
the same SEARCH_LEAGUE fitness.py already picked as the discriminating
opponents. A search and its own scoreboard should never disagree about what
"better" means.

Two things keep the signal above the noise:

  common random numbers - every candidate plays the identical seed set, so weed
  spawns and shop unlocks cancel out of the comparison.

  seat swapping - each seed is played from both sides.

w_dist is frozen. Multiplying every weight by a constant leaves the argmin
unchanged, so the space has a redundant dimension; pinning one weight removes it
and stops the search wandering along it.
"""
import json
import multiprocessing as mp
import random
import statistics
import sys

import fitness

FROZEN = {"w_dist"}

# How far a single mutation may move each weight.
STEP = {
    "w_water_urgent": 1.0,
    "w_harvest_decay": 1.0,
    "w_harvest_ripe": 1.0,
    "w_water_bonus": 1.0,
    "w_water_idle": 1.0,
    "w_plant": 1.0,
    "w_dig": 1.0,
    "w_shed": 0.4,
    "w_dist_sq": 0.05,
    "w_action_cost": 3.0,
    "w_feed": 1.0,
    "w_harvest_animal": 1.0,
    "w_collect": 1.0,
    "w_care": 1.0,
    "w_drop": 1.0,
    "w_place": 1.0,
    "w_build": 1.5,
    "w_pickup": 1.5,
    "w_pen_shed": 0.5,
    # Days of running costs held back before buying land. burn is ~$1,000, so
    # a step of 0.5 moves the reserve by about half a day of spending.
    "w_land_reserve": 0.5,
    # A dollar threshold on a crop price, so it steps in dollars.
    "w_final_water": 20.0,
    # A count of pens, so a step near 1 is a meaningful move.
    "w_pen_ahead": 1.0,
    "w_animal_reserve": 0.5,
    # Dimensionless (a load ratio scales it), so it can range widely - start
    # with a coarse step and let sigma shrink it in CMA-ES.
    "w_escape_risk": 1.0,
    # 0..1 in principle (a share of the town's drain), but nothing stops the
    # search finding a better fit outside that range, so it is not clamped.
    "w_town_drift": 0.2,
}


def _play(job):
    """One episode against one league opponent. Returns tanh(margin/SCALE),
    the same bounded per-game unit fitness.py scores with - or None if either
    side crashed, so a crash can be told apart from a genuine narrow loss."""
    params, opponent, seed, swapped = job

    import main
    from kaggle_environments import make

    main.PARAMS.update(params)
    lineup = [opponent, main.agent] if swapped else [main.agent, opponent]

    env = make("kaggriculture", configuration={"episodeSteps": 720, "seed": seed})
    env.run(lineup)
    rewards = [s.reward for s in env.steps[-1]]
    if any(r is None for r in rewards):
        return None  # a crashed agent is never an improvement

    mine, theirs = (rewards[1], rewards[0]) if swapped else (rewards[0], rewards[1])
    import math
    return math.tanh((mine - theirs) / fitness.SCALE)


def evaluate(params, seeds, pool, league=None):
    """fitness.py's own score() for one candidate - soft-min over the league,
    each opponent's own mean tanh(margin/SCALE). Searching against exactly the
    metric that will judge the result later is the whole point of wiring the
    two files together; SEARCH_LEAGUE (the opponents still close enough to
    move) is the default so the search does not spend its budget on games
    that are already saturated at +1.000."""
    league = league or fitness.SEARCH_LEAGUE
    jobs = [(params, opponent, s, sw)
            for opponent in league for s in seeds for sw in (False, True)]
    results = pool.map(_play, jobs)

    per_opponent = {o: [] for o in league}
    for (_, opponent, _, _), value in zip(jobs, results):
        if value is not None:
            per_opponent[opponent].append(value)
    means = [statistics.mean(v) for v in per_opponent.values() if v]
    return fitness.soft_min(means)


def mutate(params, rng):
    child = dict(params)
    for key in rng.sample(sorted(STEP), rng.choice([1, 2])):
        child[key] = round(child[key] + rng.gauss(0, STEP[key]), 3)
    return child


def hill_climb(iterations, seeds, pool, rng):
    import main

    best = {k: v for k, v in main.PARAMS.items() if k not in FROZEN}
    best_score = evaluate(best, seeds, pool)
    print(f"baseline score {best_score:+.4f}", file=sys.stderr)

    accepted = 0
    for i in range(1, iterations + 1):
        candidate = mutate(best, rng)
        score = evaluate(candidate, seeds, pool)
        better = score > best_score
        if better:
            best, best_score, accepted = candidate, score, accepted + 1
        print(
            f"  {i:3d}/{iterations}  {score:+.4f}"
            f"  {'ACCEPT' if better else '      '}  best {best_score:+.4f}",
            file=sys.stderr,
        )

    print(f"\naccepted {accepted}/{iterations}", file=sys.stderr)
    return best, best_score


if __name__ == "__main__":
    iterations = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    n_seeds = int(sys.argv[2]) if len(sys.argv) > 2 else 6
    offset = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    seeds = list(range(offset, offset + n_seeds))

    with mp.Pool(min(8, mp.cpu_count())) as pool:
        best, score = hill_climb(iterations, seeds, pool, random.Random(0))

    print(f"\nbest score {score:+.4f} on seeds {seeds}", file=sys.stderr)
    print(json.dumps(best, indent=4), file=sys.stderr)
    with open("tuned_params.json", "w") as f:
        json.dump(best, f, indent=4)
