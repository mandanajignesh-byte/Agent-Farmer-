"""Summarise a downloaded episode replay.

    python analyze.py replays/episode-*.json

Reports the outcome, what each side built, and - most usefully - where the two
money curves separated. Early divergence points at build order; steady
divergence points at per-turn efficiency. They need different fixes.
"""
import glob
import json
import sys

ME = "jignesh13"
ANIMALS = ("GOOSE", "COW", "SHEEP")


def _census(farm):
    """Count what is standing on a farm right now."""
    crops, animals, structures, weeds = {}, {}, 0, 0
    for row in farm["tiles"]:
        for tile in row:
            if not isinstance(tile, dict):
                continue
            kind = tile.get("kind")
            if kind == "PLANT":
                crops[tile["crop"]] = crops.get(tile["crop"], 0) + 1
            elif kind == "WEED":
                weeds += 1
            elif kind in ("COOP", "PASTURE"):
                structures += 1
                if tile.get("animal"):
                    animals[tile["animal"]] = animals.get(tile["animal"], 0) + 1
    return crops, animals, structures, weeds


def analyse(path):
    replay = json.load(open(path))
    names = replay["info"]["TeamNames"]
    steps = replay["steps"]
    mine = names.index(ME) if ME in names else 1
    theirs = 1 - mine
    opponent = names[theirs]

    final = [e.get("reward") for e in steps[-1]]
    verdict = "WIN " if final[mine] > final[theirs] else "LOSS"

    print(f"\n{'=' * 66}")
    print(f"{verdict}  vs {opponent:<22} {ME} ${final[mine]:>9,.0f}   "
          f"{opponent} ${final[theirs]:>9,.0f}")
    print("=" * 66)

    # Money every 3 days, plus what was standing at that moment.
    print(f"{'day':>4} {'me':>9} {'them':>9} {'gap':>10}   their farm")
    biggest_swing, swing_day = 0, 0
    previous_gap = 0

    for day in range(0, 30, 3):
        step = min(day * 24, len(steps) - 1)
        farms = steps[step][0]["observation"]["farms"]
        my_money = farms[mine]["money"]
        their_money = farms[theirs]["money"]
        gap = my_money - their_money

        if abs(gap - previous_gap) > abs(biggest_swing):
            biggest_swing, swing_day = gap - previous_gap, day
        previous_gap = gap

        crops, animals, structures, _weeds = _census(farms[theirs])
        crop_text = " ".join(f"{c[:4].lower()}:{n}" for c, n in sorted(crops.items()))
        animal_text = " ".join(f"{a[:4].lower()}:{n}" for a, n in sorted(animals.items()))
        quads = len(farms[theirs]["unlocked_quadrants"])
        print(f"{day:>4} {my_money:>9,.0f} {their_money:>9,.0f} {gap:>10,.0f}   "
              f"q{quads} {crop_text} {animal_text}"
              + (f" [{structures} pens]" if structures else ""))

    my_crops, my_animals, my_structures, _ = _census(steps[-1][0]["observation"]["farms"][mine])
    their_crops, their_animals, their_structures, _ = _census(
        steps[-1][0]["observation"]["farms"][theirs])
    print(f"\n  end   me: {my_crops} animals={my_animals or '{}'} pens={my_structures}")
    print(f"        them: {their_crops} animals={their_animals or '{}'} pens={their_structures}")
    print(f"  largest 3-day swing: ${biggest_swing:,.0f} around day {swing_day}")


def summarise(paths):
    """One line per game, then the aggregate that matters."""
    rows = []
    for path in paths:
        replay = json.load(open(path))
        names = replay["info"]["TeamNames"]
        mine = names.index(ME) if ME in names else 1
        theirs = 1 - mine
        final = [e.get("reward") or 0 for e in replay["steps"][-1]]
        crops, animals, pens, _ = _census(
            replay["steps"][-1][0]["observation"]["farms"][theirs])
        rows.append({
            "opponent": names[theirs],
            "me": final[mine],
            "them": final[theirs],
            "won": final[mine] > final[theirs],
            "animals": sum(animals.values()),
            "pens": pens,
            "crops": sum(crops.values()),
        })

    rows.sort(key=lambda r: -r["them"])
    print(f"\n{'result':<7}{'opponent':<24}{'me':>9}{'them':>9}{'animals':>9}{'pens':>6}{'crops':>7}")
    print("-" * 71)
    for r in rows:
        print(f"{'WIN' if r['won'] else 'LOSS':<7}{r['opponent'][:23]:<24}"
              f"{r['me']:>9,.0f}{r['them']:>9,.0f}{r['animals']:>9}{r['pens']:>6}{r['crops']:>7}")

    with_animals = [r for r in rows if r["animals"] > 0]
    without = [r for r in rows if r["animals"] == 0]
    my_scores = sorted(r["me"] for r in rows)

    def record(group):
        wins = sum(r["won"] for r in group)
        return f"{wins}W-{len(group) - wins}L" if group else "none"

    print(f"\n  overall            {record(rows)}   ({len(rows)} games)")
    print(f"  vs agents WITH animals    {record(with_animals)}")
    print(f"  vs agents WITHOUT animals {record(without)}")
    if with_animals:
        print(f"\n  their score, with animals    "
              f"${sum(r['them'] for r in with_animals) / len(with_animals):>9,.0f} avg")
    if without:
        print(f"  their score, without animals ${sum(r['them'] for r in without) / len(without):>9,.0f} avg")
    print(f"\n  our score: min ${my_scores[0]:,.0f}  median ${my_scores[len(my_scores) // 2]:,.0f}"
          f"  max ${my_scores[-1]:,.0f}")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    targets = args or sorted(glob.glob("replays/*.json"))
    if "--each" in sys.argv:
        for path in targets:
            analyse(path)
    summarise(targets)
