"""Assert the agent's DECISIONS are sane, not just its constants.

test_model.py checks what the agent believes about the rules. Nothing checked
what it does with those beliefs, and that is where this project's most expensive
bugs have lived:

  a string patch silently failed to apply, so w_pickup candidates were never
  generated, so one worker carried the whole herd's feed on foot and six animals
  starved to death every season. Nothing failed. The agent ran, scored, and
  benchmarked as merely slightly worse.

  CARE - the highest-return action in the game - was never emitted once, for
  weeks, because no candidate produced it.

Both are invisible to a rules suite and to a benchmark. Both are trivially
visible to an invariant checked over a real game: no animal should ever starve
while there is wheat in the shed, and a farm with animals should emit CARE.

These run the real agent over a full 720-turn season and assert properties that
must hold regardless of tuning.

    python test_behaviour.py [n_seeds]
"""
import collections
import multiprocessing as mp
import sys

from kaggle_environments import make

import main as agent

FAILURES = []
MOVES = ("NORTH", "SOUTH", "EAST", "WEST")


def check(name, ok, detail=""):
    print(f"  [{'ok  ' if ok else 'FAIL'}] {name}  {detail}")
    if not ok:
        FAILURES.append(f"{name}: {detail}")


def observe(seed):
    """Play one full season and record everything an invariant might need."""
    rec = {
        "actions": collections.Counter(),
        "orders": collections.Counter(),
        "max_orders": 0,
        "escapes": 0,
        "starved_with_feed": 0,
        "idle_with_work": 0,
        "plant_without_seed": 0,
        "duplicate_targets": 0,
        "animals_seen": 0,
        "fertilizer_missed": 0,
        "shed_peak": 0,
        "bad_shape": 0,
    }
    prev_animals = {}

    def spy(obs):
        farm = obs["farms"][obs["player"]]
        priv = obs["private"]
        act = agent.agent(obs)

        if not isinstance(act, dict) or set(act) != {"farmer", "hands", "market"}:
            rec["bad_shape"] += 1
            return act
        if len(act["hands"]) != len(farm["hands"]):
            rec["bad_shape"] += 1

        all_actions = [act["farmer"]] + act["hands"]
        for a in all_actions:
            rec["actions"][a[0]] += 1

        rec["max_orders"] = max(rec["max_orders"], len(act["market"]))
        for o in act["market"]:
            rec["orders"][o[0]] += 1

        # an animal that vanished from a pen it occupied last turn starved
        animals_now = {}
        unfed = 0
        fert_available = 0
        for y, row in enumerate(farm["tiles"]):
            for x, t in enumerate(row):
                if isinstance(t, dict) and t.get("kind") in ("COOP", "PASTURE"):
                    if t.get("animal"):
                        animals_now[(x, y)] = t["animal"]
                        unfed += not t["fed_today"]
                        fert_available += bool(t.get("fertilizer_available"))
        for pos, name in prev_animals.items():
            if pos not in animals_now:
                rec["escapes"] += 1
                if priv["shed"].get("WHEAT", 0) > 0:
                    rec["starved_with_feed"] += 1
        prev_animals.clear()
        prev_animals.update(animals_now)
        rec["animals_seen"] = max(rec["animals_seen"], len(animals_now))

        # PLANT is only legal against a seed we hold
        planted = collections.Counter(a[1] for a in all_actions
                                      if a[0] == "PLANT" and len(a) > 1)
        for crop, n in planted.items():
            if n > priv["seeds"].get(crop, 0):
                rec["plant_without_seed"] += 1

        # a worker standing idle while a fed-and-uncared animal waits is the
        # CARE bug; a worker idle while an animal is unfed and wheat sits in
        # the shed is the starvation bug
        idle = sum(1 for a in all_actions if a[0] == "PASS")
        if idle and unfed and priv["shed"].get("WHEAT", 0) > 0:
            rec["idle_with_work"] += 1
        if fert_available and idle:
            rec["fertilizer_missed"] += 1

        rec["shed_peak"] = max(rec["shed_peak"], sum(priv["shed"].values()))
        return act

    env = make("kaggriculture", configuration={"episodeSteps": 720, "seed": seed})
    env.run([spy, "champion.py"])
    rec["reward"] = [s.reward for s in env.steps[-1]][0]
    return rec


def main(n_seeds):
    with mp.Pool(min(8, mp.cpu_count())) as pool:
        games = pool.map(observe, range(200, 200 + n_seeds))

    total = {k: sum(g[k] for g in games) for k in games[0]
             if isinstance(games[0][k], (int, float)) and k != "reward"}
    acts = collections.Counter()
    orders = collections.Counter()
    for g in games:
        acts.update(g["actions"])
        orders.update(g["orders"])
    worker_actions = sum(acts.values())

    print(f"{n_seeds} full seasons, {worker_actions:,} worker-actions\n")

    check("the agent never crashes or returns a malformed action",
          total["bad_shape"] == 0, f"({total['bad_shape']} malformed turns)")
    check("market orders never exceed the 10-per-turn cap",
          max(g["max_orders"] for g in games) <= 10,
          f"(peak {max(g['max_orders'] for g in games)})")
    check("never plants a crop it holds no seed for",
          total["plant_without_seed"] == 0,
          f"({total['plant_without_seed']} turns)")

    # the two bugs that cost the most and were invisible to every benchmark
    check("no animal starves while wheat sits in the shed",
          total["starved_with_feed"] == 0,
          f"({total['starved_with_feed']} of {total['escapes']} escapes had feed available)")
    # Reported, not asserted. A worker holding no wheat cannot feed, so PASS can
    # be the correct action even with a hungry animal on the farm - the invariant
    # that matters is that nothing starves. Kept visible because a rise here is
    # a leading indicator of the feed pipeline backing up.
    print(f"  [info] turns with an idle worker and an unfed animal: "
          f"{total['idle_with_work']}")

    animals = max(g["animals_seen"] for g in games)
    check("the farm actually keeps animals", animals > 0, f"(peak herd {animals})")
    if animals:
        check("CARE is emitted", acts["CARE"] > 0, f"({acts['CARE']} times)")
        check("FEED is emitted", acts["FEED"] > 0, f"({acts['FEED']} times)")
        check("COLLECT_FERTILIZER is emitted", acts["COLLECT_FERTILIZER"] > 0,
              f"({acts['COLLECT_FERTILIZER']} times)")

    check("produce is actually sold", orders["SELL"] > 0, f"({orders['SELL']} orders)")
    check("hands are hired", orders["HIRE"] > 0, f"({orders['HIRE']} orders)")
    check("weeds are cleared", acts["DIG"] > 0, f"({acts['DIG']} times)")
    check("the shed never overflows",
          max(g["shed_peak"] for g in games) <= 100,
          f"(peak {max(g['shed_peak'] for g in games)} of 100)")

    move = sum(acts[m] for m in MOVES)
    print(f"\n  action mix: move {100*move/worker_actions:.1f}%"
          f"   idle {100*acts['PASS']/worker_actions:.1f}%"
          f"   productive {100*(worker_actions-move-acts['PASS'])/worker_actions:.1f}%")
    print("  " + "  ".join(f"{k}={v}" for k, v in acts.most_common(8)))

    print()
    if FAILURES:
        print(f"{len(FAILURES)} BEHAVIOURAL INVARIANT(S) BROKEN:")
        for f in FAILURES:
            print(f"  - {f}")
        sys.exit(1)
    print("every behavioural invariant holds")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 6)
