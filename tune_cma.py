"""Tune main.PARAMS with CMA-ES instead of hill climbing.

    python tune_cma.py [generations] [seeds] [seed_offset]

Hill climbing perturbs one or two weights at a time and keeps the result if it
scores better. That stalls in a local optimum - acceptance fell to 2 in 150 once
the weights were roughly right - because escaping often needs several weights to
move together, and any single one of those moves looks worse on its own.

CMA-ES samples a whole population of candidate vectors each generation, then
adapts both the step size and the covariance between weights from what scored
well. It can therefore learn that two weights want to move together, which is
exactly the move hill climbing cannot make.

Weights are normalised by their step scale before being handed to CMA, since the
raw values span 0.05 to 30 and a single sigma cannot serve both ends.

Fitness is the money margin against champion.py on a fixed seed set, for the
same reasons as tune.py: win rate over a dozen games is too coarse to optimise,
and identical seeds cancel most of the per-episode noise out of the comparison.
Validate the result with bench.py on seeds the search never saw.
"""
import json
import multiprocessing as mp
import statistics
import sys

import cma

from tune import CHAMPION, FROZEN, STEP, _play


def evaluate(params, seeds, pool):
    jobs = [(params, s, sw) for s in seeds for sw in (False, True)]
    return statistics.mean(pool.map(_play, jobs))


def main(generations, n_seeds, offset):
    import main as agent

    names = [k for k in STEP if k not in FROZEN]
    scales = [STEP[k] for k in names]
    start = [agent.PARAMS[k] / s for k, s in zip(names, scales)]
    seeds = list(range(offset, offset + n_seeds))

    def to_params(vector):
        return {n: round(v * s, 3) for n, v, s in zip(names, vector, scales)}

    es = cma.CMAEvolutionStrategy(start, 1.0, {"verbose": -9, "popsize": 8})
    best, best_score = to_params(start), None

    with mp.Pool(min(8, mp.cpu_count())) as pool:
        base = evaluate(best, seeds, pool)
        best_score = base
        print(f"baseline margin ${base:+,.0f}  over {len(names)} weights", file=sys.stderr)

        for gen in range(1, generations + 1):
            population = es.ask()
            scores = [evaluate(to_params(v), seeds, pool) for v in population]
            es.tell(population, [-s for s in scores])  # CMA minimises

            top = max(scores)
            if top > best_score:
                best_score = top
                best = to_params(population[scores.index(top)])
            print(
                f"  gen {gen:3d}/{generations}  best this gen ${top:+9,.0f}"
                f"   overall ${best_score:+9,.0f}   sigma {es.sigma:.3f}",
                file=sys.stderr,
            )

    print(f"\nbest margin ${best_score:+,.0f} on seeds {seeds}", file=sys.stderr)
    print(json.dumps(best, indent=4), file=sys.stderr)
    with open("tuned_params.json", "w") as f:
        json.dump(best, f, indent=4)


if __name__ == "__main__":
    main(
        int(sys.argv[1]) if len(sys.argv) > 1 else 15,
        int(sys.argv[2]) if len(sys.argv) > 2 else 6,
        int(sys.argv[3]) if len(sys.argv) > 3 else 0,
    )
