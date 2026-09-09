"""Hill-climb main.PARAMS against a champion snapshot.

    python tune.py [iterations] [seeds] [seed_offset]

Fitness is the mean money margin against champion.py on a fixed seed set, not
win rate: win rate over a dozen games is too coarse a signal to climb, while the
margin is smooth. Win rate is the real metric, so validate the result with
bench.py afterwards - and on seeds the search never saw.

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

FROZEN = {"w_dist"}
CHAMPION = "champion.py"

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
    "w_action_cost": 3.0,
    "w_feed": 1.0,
    "w_harvest_animal": 1.0,
    "w_collect": 1.0,
    "w_place": 1.0,
    "w_build": 1.5,
    "w_pickup": 1.5,
    "w_pen_shed": 0.5,
}


def _play(job):
    """One episode. Returns our money minus theirs."""
    params, seed, swapped = job

    import main
    from kaggle_environments import make

    main.PARAMS.update(params)
    lineup = [CHAMPION, main.agent] if swapped else [main.agent, CHAMPION]

    env = make("kaggriculture", configuration={"episodeSteps": 720, "seed": seed})
    env.run(lineup)
    rewards = [s.reward for s in env.steps[-1]]
    if any(r is None for r in rewards):
        return -1e6  # a crashed agent is never an improvement

    mine, theirs = (rewards[1], rewards[0]) if swapped else (rewards[0], rewards[1])
    return mine - theirs


def evaluate(params, seeds, pool):
    jobs = [(params, s, sw) for s in seeds for sw in (False, True)]
    return statistics.mean(pool.map(_play, jobs))


def mutate(params, rng):
    child = dict(params)
    for key in rng.sample(sorted(STEP), rng.choice([1, 2])):
        child[key] = round(child[key] + rng.gauss(0, STEP[key]), 3)
    return child


def hill_climb(iterations, seeds, pool, rng):
    import main

    best = {k: v for k, v in main.PARAMS.items() if k not in FROZEN}
    best_score = evaluate(best, seeds, pool)
    print(f"baseline margin ${best_score:+,.0f}", file=sys.stderr)

    accepted = 0
    for i in range(1, iterations + 1):
        candidate = mutate(best, rng)
        score = evaluate(candidate, seeds, pool)
        better = score > best_score
        if better:
            best, best_score, accepted = candidate, score, accepted + 1
        print(
            f"  {i:3d}/{iterations}  ${score:+9,.0f}"
            f"  {'ACCEPT' if better else '      '}  best ${best_score:+,.0f}",
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

    print(f"\nbest margin ${score:+,.0f} on seeds {seeds}", file=sys.stderr)
    print(json.dumps(best, indent=4), file=sys.stderr)
    with open("tuned_params.json", "w") as f:
        json.dump(best, f, indent=4)
