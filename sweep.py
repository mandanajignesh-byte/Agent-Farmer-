"""Sweep one weight on its own, before committing hours to a full retune.

    python sweep.py w_final_water 0 30 50 100 [--seeds 8] [--offset 0]

A search over twenty weights cannot say whether one idea works - it mixes the
new term with nineteen others, and a null result is unreadable. One parameter,
one axis, the same seeds throughout, against champion.py.
"""
import multiprocessing as mp
import statistics
import sys

from tune import _play
import main as agent


def main(name, values, n_seeds, offset):
    if name not in agent.PARAMS:
        sys.exit(f"{name} is not in main.PARAMS")
    seeds = range(offset, offset + n_seeds)
    print(f"{name}  (currently {agent.PARAMS[name]})  "
          f"on seeds {offset}..{offset + n_seeds - 1} vs champion.py",
          file=sys.stderr)
    with mp.Pool(min(8, mp.cpu_count())) as pool:
        for v in values:
            params = dict(agent.PARAMS)
            params[name] = v
            jobs = [(params, s, sw) for s in seeds for sw in (False, True)]
            r = pool.map(_play, jobs)
            wins = sum(x > 0 for x in r)
            print(f"  {v:>8.2f}   margin ${statistics.mean(r):>+9,.0f}"
                  f"   {wins}/{len(r)} wins", flush=True)


if __name__ == "__main__":
    args = [a for a in sys.argv[1:]]
    n_seeds, offset = 8, 0
    for flag, default in (("--seeds", 8), ("--offset", 0)):
        if flag in args:
            i = args.index(flag)
            val = int(args[i + 1])
            del args[i:i + 2]
            if flag == "--seeds":
                n_seeds = val
            else:
                offset = val
    if len(args) < 2:
        sys.exit(__doc__)
    main(args[0], [float(v) for v in args[1:]], n_seeds, offset)
