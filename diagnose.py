"""Compare what the agent does in games it wins against games it loses.

    python diagnose.py [n_seeds] [offset]

Aggregate margins say whether a change helped. They say nothing about why some
games are lost, and averaging wins and losses together hides exactly the
difference we want to see. This plays a seed set, splits the games by outcome,
and prints the same measurements for each group so the divergence is visible.
"""
import collections
import multiprocessing as mp
import statistics
import sys

from kaggle_environments import make

import main as agent

TRACK_DAYS = (4, 8, 12, 16, 20, 24, 28)


def _census(farm):
    crops = animals = pens = weeds = empty = 0
    for row in farm["tiles"]:
        for tile in row:
            if tile is None:
                empty += 1
            elif isinstance(tile, dict):
                kind = tile.get("kind")
                if kind == "PLANT":
                    crops += 1
                elif kind == "WEED":
                    weeds += 1
                elif kind in ("COOP", "PASTURE"):
                    pens += 1
                    animals += bool(tile.get("animal"))
    return crops, animals, pens, weeds, empty


def play(seed):
    """One episode, instrumented. Returns everything needed to compare games."""
    actions = collections.Counter()
    money, farmstate = {}, {}
    sold = collections.Counter()
    last_shed = {}

    def spy(obs):
        act = agent.agent(obs)
        for a in [act["farmer"]] + act["hands"]:
            actions[a[0]] += 1
        for order in act["market"]:
            if order[0] == "SELL":
                sold[order[1]] += order[2]
        farm = obs["farms"][obs["player"]]
        if obs["hour"] == 12 and obs["day"] in TRACK_DAYS:
            money[obs["day"]] = farm["money"]
            farmstate[obs["day"]] = _census(farm)
        if obs["step"] > 700:
            last_shed.update(obs["private"]["shed"])
        return act

    env = make("kaggriculture", configuration={"episodeSteps": 720, "seed": seed})
    env.run([spy, "champion.py"])
    mine, theirs = (s.reward for s in env.steps[-1])

    total = sum(actions.values()) or 1
    move = sum(actions[d] for d in ("NORTH", "SOUTH", "EAST", "WEST"))
    return {
        "seed": seed,
        "won": mine > theirs,
        "mine": mine,
        "theirs": theirs,
        "money": money,
        "farm": farmstate,
        "move_pct": 100 * move / total,
        "idle_pct": 100 * actions["PASS"] / total,
        "actions": {k: 100 * v / total for k, v in actions.items()},
        "sold": dict(sold),
    }


def summarise(group, label):
    if not group:
        print(f"\n{label}: none")
        return
    print(f"\n{label}  ({len(group)} games)")
    print(f"  our score      ${statistics.mean(g['mine'] for g in group):>9,.0f}"
          f"   opponent ${statistics.mean(g['theirs'] for g in group):>9,.0f}")
    print(f"  movement {statistics.mean(g['move_pct'] for g in group):>5.1f}%"
          f"   idle {statistics.mean(g['idle_pct'] for g in group):>5.1f}%")

    print("  money by day: " + "  ".join(
        f"d{d}=${statistics.mean(g['money'].get(d, 0) for g in group):>7,.0f}"
        for d in TRACK_DAYS))
    for i, name in enumerate(("crops", "animals", "pens", "weeds", "empty")):
        line = "  ".join(
            f"d{d}={statistics.mean(g['farm'].get(d, (0,) * 5)[i] for g in group):>5.1f}"
            for d in TRACK_DAYS)
        print(f"  {name:<8}{line}")

    keys = sorted({k for g in group for k in g["actions"]},
                  key=lambda k: -statistics.mean(g["actions"].get(k, 0) for g in group))
    print("  actions: " + "  ".join(
        f"{k}={statistics.mean(g['actions'].get(k, 0) for g in group):.1f}%"
        for k in keys[:9]))
    print("  sold: " + "  ".join(
        f"{k}={statistics.mean(g['sold'].get(k, 0) for g in group):.0f}"
        for k in sorted({k for g in group for k in g["sold"]})))


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    offset = int(sys.argv[2]) if len(sys.argv) > 2 else 500
    with mp.Pool(min(8, mp.cpu_count())) as pool:
        games = pool.map(play, range(offset, offset + n))

    summarise([g for g in games if g["won"]], "WON")
    summarise([g for g in games if not g["won"]], "LOST")
