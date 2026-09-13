"""Sweep one weight on its own, before committing hours to a full retune.

    python sweep.py w_final_water 0 30 50 100 [--seeds 8] [--offset 0] [--vs PATH]

A search over twenty-odd weights cannot say whether one idea works - it mixes
the new term with everything else, and a null result is unreadable. One
parameter, one axis, the same seeds throughout, against one real opponent
(default: league_public/public_2900.py - the one this weight is meant to
matter against; pass --vs to test a different one, e.g. league/v22.py).

Plays raw dollar margin directly, not tune._play's bounded per-opponent-scale
unit - that unit exists so several different opponents can be combined in one
search objective without one dominating; a sweep only ever looks at one
opponent at a time, so there is nothing to combine and the compression only
throws away precision. (tune._play used to return raw margin too, until the
search itself needed the bounded unit - this file was left calling it and
silently printing a mangled result, "$-1" every time, until that was caught.)
"""
import multiprocessing as mp
import statistics
import sys

from kaggle_environments import make
import main as agent

DEFAULT_OPPONENT = "league_public/public_2900.py"


def _play_raw(job):
    """One episode. Returns our money minus theirs, or None if either side
    crashed."""
    params, opponent, seed, swapped = job
    import main as m
    m.PARAMS.update(params)
    lineup = [opponent, m.agent] if swapped else [m.agent, opponent]
    env = make("kaggriculture", configuration={"episodeSteps": 720, "seed": seed})
    env.run(lineup)
    rewards = [s.reward for s in env.steps[-1]]
    if any(r is None for r in rewards):
        return None
    mine, theirs = (rewards[1], rewards[0]) if swapped else (rewards[0], rewards[1])
    return mine - theirs


def main(name, values, n_seeds, offset, opponent):
    if name not in agent.PARAMS:
        sys.exit(f"{name} is not in main.PARAMS")
    seeds = range(offset, offset + n_seeds)
    print(f"{name}  (currently {agent.PARAMS[name]})  "
          f"on seeds {offset}..{offset + n_seeds - 1} vs {opponent}",
          file=sys.stderr)
    with mp.Pool(min(8, mp.cpu_count())) as pool:
        for v in values:
            params = dict(agent.PARAMS)
            params[name] = v
            jobs = [(params, opponent, s, sw) for s in seeds for sw in (False, True)]
            r = [x for x in pool.map(_play_raw, jobs) if x is not None]
            wins = sum(x > 0 for x in r)
            print(f"  {v:>8.2f}   margin ${statistics.mean(r):>+9,.0f}"
                  f"   {wins}/{len(r)} wins", flush=True)


if __name__ == "__main__":
    args = [a for a in sys.argv[1:]]
    n_seeds, offset, opponent = 8, 0, DEFAULT_OPPONENT
    for flag, kind in (("--seeds", int), ("--offset", int), ("--vs", str)):
        if flag in args:
            i = args.index(flag)
            val = kind(args[i + 1])
            del args[i:i + 2]
            if flag == "--seeds":
                n_seeds = val
            elif flag == "--offset":
                offset = val
            else:
                opponent = val
    if len(args) < 2:
        sys.exit(__doc__)
    main(args[0], [float(v) for v in args[1:]], n_seeds, offset, opponent)
