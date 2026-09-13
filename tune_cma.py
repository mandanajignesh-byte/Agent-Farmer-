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

Fitness is fitness.py's own league soft-min score, the same objective tune.py
climbs and the same one the ladder is judged on - not dollars against one
champion. evaluate() is imported from tune.py rather than redefined, so the
two searches and the scoreboard can never quietly drift apart.
"""
import json
import multiprocessing as mp
import sys

import cma

from tune import FROZEN, STEP, evaluate


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
        print(f"baseline score {base:+.4f}  over {len(names)} weights", file=sys.stderr)

        for gen in range(1, generations + 1):
            population = es.ask()
            scores = [evaluate(to_params(v), seeds, pool) for v in population]
            es.tell(population, [-s for s in scores])  # CMA minimises

            top = max(scores)
            if top > best_score:
                best_score = top
                best = to_params(population[scores.index(top)])
            print(
                f"  gen {gen:3d}/{generations}  best this gen {top:+.4f}"
                f"   overall {best_score:+.4f}   sigma {es.sigma:.3f}",
                file=sys.stderr,
            )

    print(f"\nbest score {best_score:+.4f} on seeds {seeds}", file=sys.stderr)
    print(json.dumps(best, indent=4), file=sys.stderr)
    with open("tuned_params.json", "w") as f:
        json.dump(best, f, indent=4)


if __name__ == "__main__":
    main(
        int(sys.argv[1]) if len(sys.argv) > 1 else 15,
        int(sys.argv[2]) if len(sys.argv) > 2 else 6,
        int(sys.argv[3]) if len(sys.argv) > 3 else 0,
    )
