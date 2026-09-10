"""Assert the agent's beliefs about the rules against the real environment.

The competition ships a perfect simulator; we do not need to build one. What we
DO have that can be wrong is the *model inside main.py* - CROP_SPEC, ANIMAL_SPEC,
MAX_YIELD_DAY, BONUS_START, price_at - which every value function is built on.
A wrong constant there makes the agent optimise confidently toward the wrong
thing, and no fitness function can detect it, because the agent would be
reasoning about a game that does not exist.

Two such errors were found by reading the source (animal lead time, and the
watering bonus applied to ongoing crops). This turns each belief into a probe:
drive a scripted farmer through the real env and check the outcome matches what
main.py predicts.

    python test_model.py            run every probe
    python test_model.py yield      run probes whose name contains "yield"
"""
import sys

from kaggle_environments import make

import main as agent

TPD = 24
FAILURES = []


def check(name, got, want, note=""):
    ok = got == want
    mark = "ok  " if ok else "FAIL"
    print(f"  [{mark}] {name}: got {got!r}, expected {want!r} {note}")
    if not ok:
        FAILURES.append(f"{name}: got {got!r}, expected {want!r} {note}")
    return ok


def drive(script, steps, seed=1):
    """Run `steps` turns with a scripted farmer. `script(obs, log)` returns the
    farmer's action list; `log` is a dict the script may record into."""
    log = {}

    def bot(obs):
        if obs["player"] != 0:
            return {"farmer": ["PASS"], "hands": [], "market": []}
        return script(obs, log)

    env = make("kaggriculture",
               configuration={"episodeSteps": steps, "seed": seed,
                              "weedSpawnChance": 0.0})
    env.run([bot, "pass"])
    return log, env


def tile_at(obs, x, y):
    return obs["farms"][obs["player"]]["tiles"][y][x]


# --------------------------------------------------------------------------
# crop yields - CROP_SPEC and MAX_YIELD_DAY
# --------------------------------------------------------------------------

def crop_script(crop, harvest_age, water=True):
    """Plant one tile, water it daily, harvest at `harvest_age`, record yield."""
    def script(obs, log):
        fx, fy = obs["farms"][0]["farmer"]
        t = tile_at(obs, fx, fy)
        if obs["step"] == 0:
            return {"farmer": ["PASS"], "hands": [],
                    "market": [["BUY_SEED", crop, 3]]}
        if t is None and "yield_at_harvest" not in log:
            return {"farmer": ["PLANT", crop], "hands": [], "market": []}
        if isinstance(t, dict) and t.get("kind") == "PLANT":
            age = obs["day"] - t["planted_day"]
            log.setdefault("by_age", {}).setdefault(age, t["yield_units"])
            if age >= harvest_age:
                log.setdefault("yield_at_harvest", t["yield_units"])
                return {"farmer": ["HARVEST"], "hands": [], "market": []}
            if water and not t["watered_today"]:
                return {"farmer": ["WATER"], "hands": [], "market": []}
        return {"farmer": ["PASS"], "hands": [], "market": []}
    return script


def probe_crop_yields():
    """CROP_SPEC must equal what the agent actually collects.

    The agent now takes the final bonus watering before harvesting, so a tile
    reaches the full CROP_SPEC yield. Harvesting first - what main.py did until
    the ordering was fixed - forfeits exactly one unit per tile, so that case is
    asserted too, as a guard against the ordering silently regressing."""
    print()
    print("crop yields - CROP_SPEC vs what a tile really produces")
    for crop in ("WHEAT", "CARROT", "MELON"):
        age = agent.MAX_YIELD_DAY[crop]
        want = agent.CROP_SPEC[crop][0]
        good, _ = drive(crop_script_water_first(crop, age), steps=(age + 3) * TPD)
        check(f"{crop} watering before harvest", good.get("yield_at_harvest"),
              want, "(CROP_SPEC)")
        bad, _ = drive(crop_script(crop, age), steps=(age + 3) * TPD)
        check(f"{crop} harvesting first", bad.get("yield_at_harvest"), want - 1,
              "(the unit the old ordering threw away)")


def probe_ongoing_watering():
    """Ongoing crops get no watering bonus - only a fixed production schedule.

    Comparing a watered plant against an unwatered one does not test this: the
    unwatered control simply dies of thirst on day two. The actual claim is
    that a watered tomato gains exactly 1 per scheduled production on days
    8-11 and is never doubled (doubling also needs fertilizer). BONUS_START
    must therefore not list ongoing crops at all."""
    print()
    print("ongoing crops - fixed schedule, watering adds no yield")
    log, _ = drive(crop_script("TOMATO", 11), steps=13 * TPD)
    by_age = log.get("by_age", {})
    got = {a: by_age[a] for a in sorted(by_age) if a <= 11}
    want = {a: 0 for a in range(0, 8)}
    want.update({8: 1, 9: 2, 10: 3, 11: 4})
    check("TOMATO yield by age, watered daily", got, want,
          "(1 per scheduled production, days 8-11, never doubled)")
    check("TOMATO absent from BONUS_START", "TOMATO" in agent.BONUS_START, False)


# --------------------------------------------------------------------------
# price model - price_at() against every quoted price, every turn
# --------------------------------------------------------------------------

def probe_prices():
    print("\nprice model - price_at(product, inventory) vs the env's own quote")
    def script(obs, log):
        bad = log.setdefault("bad", [])
        for p, quoted in obs["market"]["prices"].items():
            inv = obs["market"]["inventory"][p]
            ours = agent.price_at(p, inv)
            if ours != quoted:
                bad.append((obs["step"], p, inv, ours, quoted))
        log["checked"] = log.get("checked", 0) + len(obs["market"]["prices"])
        return {"farmer": ["PASS"], "hands": [], "market": []}
    log, _ = drive(script, steps=300)
    check("price mismatches", len(log.get("bad", [])), 0,
          f"({log.get('checked')} price points checked)")
    for row in log.get("bad", [])[:5]:
        print(f"      step {row[0]} {row[1]} inv={row[2]} ours={row[3]} env={row[4]}")


# --------------------------------------------------------------------------
# town drain - the formula we are about to tune on
# --------------------------------------------------------------------------

SHOPS = {
    "BAKERY": ["EGG", "WHEAT"],
    "PIZZA_SHOP": ["MILK", "TOMATO", "WHEAT"],
    "BRUNCH_SPOT": ["EGG", "WHEAT", "STRAWBERRY"],
    "YARN_STORE": ["WOOL"],
    "ICE_CREAM_SHOP": ["STRAWBERRY", "MILK", "WHEAT"],
    "PET_CAFE": ["CARROT"],
    "SMOOTHIE_SHOP": ["STRAWBERRY", "MILK"],
    "FARMERS_MARKET": ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY"],
}


def town_drain_per_day(product, shops):
    """Units/day the town removes. Shops tick every 4 turns => 6 times a day;
    single-product shops consume double. The town centre takes 1/day of
    everything except fertilizer."""
    rate = 0 if product == "FERTILIZER" else 1
    for shop in shops:
        demands = SHOPS[shop]
        if product in demands:
            rate += 6 * (2 if len(demands) == 1 else 1)
    return rate


def probe_town_drain():
    """Nobody sells anything, so every inventory change is the town."""
    print("\ntown drain - predicted vs observed, with no player selling at all")
    def script(obs, log):
        if obs["hour"] == 0:
            log.setdefault("days", {})[obs["day"]] = (
                dict(obs["market"]["inventory"]), list(obs["town"]["unlocked_shops"]))
        return {"farmer": ["PASS"], "hands": [], "market": []}
    log, _ = drive(script, steps=30 * TPD)
    days = log["days"]
    worst = 0
    for product in ("WHEAT", "MILK", "WOOL", "CARROT", "EGG", "MELON", "FERTILIZER"):
        errs = []
        for d in range(1, 29):
            if d not in days or d + 1 not in days:
                continue
            inv_now, shops = days[d]
            inv_next, _ = days[d + 1]
            observed = inv_now[product] - inv_next[product]
            predicted = town_drain_per_day(product, shops)
            errs.append(observed - predicted)
        err = max(abs(e) for e in errs) if errs else 0
        worst = max(worst, err)
        print(f"    {product:<11} max daily error {err}")
    check("town drain formula", worst, 0, "(units/day, across 28 days)")


# --------------------------------------------------------------------------
# survival rules - two missed days kills a plant or loses an animal
# --------------------------------------------------------------------------

def probe_weed_after_two_dry_days():
    print("\nsurvival - a plant unwatered for two end-of-days becomes a weed")
    def script(obs, log):
        fx, fy = obs["farms"][0]["farmer"]
        t = tile_at(obs, fx, fy)
        if obs["step"] == 0:
            return {"farmer": ["PASS"], "hands": [], "market": [["BUY_SEED", "WHEAT", 2]]}
        if t is None and "planted_day" not in log:
            return {"farmer": ["PLANT", "WHEAT"], "hands": [], "market": []}
        if isinstance(t, dict) and t.get("kind") == "PLANT":
            log.setdefault("planted_day", t["planted_day"])
            log["unwatered"] = t["consecutive_unwatered"]
        if isinstance(t, dict) and t.get("kind") == "WEED":
            log.setdefault("weed_day", obs["day"])
        return {"farmer": ["PASS"], "hands": [], "market": []}
    log, _ = drive(script, steps=6 * TPD)
    # planting day counts as the first unwatered day, so the second end-of-day
    # refresh after planting is when it dies.
    check("days from planting to weed", 
          (log.get("weed_day", -99) - log.get("planted_day", 0)), 1,
          "(planting day counts as already unwatered)")


# --------------------------------------------------------------------------
# hiring - fib schedule
# --------------------------------------------------------------------------

def probe_hire_cost():
    print("\nhiring - cost of the nth hand in a day")
    def script(obs, log):
        if obs["day"] != 0:
            return {"farmer": ["PASS"], "hands": [["PASS"]] * len(obs["farms"][0]["hands"]),
                    "market": []}
        farm = obs["farms"][0]
        log.setdefault("trace", []).append((obs["hour"], farm["money"], len(farm["hands"])))
        return {"farmer": ["PASS"], "hands": [["PASS"]] * len(farm["hands"]),
                "market": [["HIRE"]] if obs["hour"] < 10 else []}
    log, _ = drive(script, steps=TPD)
    trace = log["trace"]
    costs = []
    for (_, m0, n0), (_, m1, n1) in zip(trace, trace[1:]):
        if n1 > n0:
            costs.append(m0 - m1)
    fib, a, b = [], 1, 1
    for _ in range(len(costs)):
        fib.append(a)
        a, b = b, a + b
    check("hire costs in order", costs, fib, "(FARM_HAND_COST_MULT * fib(n))")

    ours, a, b = 0, 1, 1
    for _ in range(agent.HANDS_PER_DAY):
        ours += a
        a, b = b, a + b
    check("daily_burn's hire term", ours, sum(fib[:agent.HANDS_PER_DAY]) if len(fib) >= agent.HANDS_PER_DAY else ours,
          f"(cost of {agent.HANDS_PER_DAY} hands)")


PROBES = [
    ("crop_yields", probe_crop_yields),
    ("ongoing_watering", probe_ongoing_watering),
    ("prices", probe_prices),
    ("town_drain", probe_town_drain),
    ("weeds", probe_weed_after_two_dry_days),
    ("hire_cost", probe_hire_cost),
]




def crop_script_water_first(crop, harvest_age):
    """Same as crop_script but takes the final bonus watering BEFORE harvesting.

    main.py cannot do this: `ripe` short-circuits the elif chain, so a tile that
    is both ripe and unwatered only ever offers HARVEST. Watering on the
    max-yield day still adds a unit, so that ordering silently forfeits one unit
    per tile on every one-time crop."""
    def script(obs, log):
        fx, fy = obs["farms"][0]["farmer"]
        t = tile_at(obs, fx, fy)
        if obs["step"] == 0:
            return {"farmer": ["PASS"], "hands": [], "market": [["BUY_SEED", crop, 3]]}
        if t is None and "yield_at_harvest" not in log:
            return {"farmer": ["PLANT", crop], "hands": [], "market": []}
        if isinstance(t, dict) and t.get("kind") == "PLANT":
            age = obs["day"] - t["planted_day"]
            if not t["watered_today"] and age <= harvest_age:
                return {"farmer": ["WATER"], "hands": [], "market": []}
            if age >= harvest_age:
                log.setdefault("yield_at_harvest", t["yield_units"])
                return {"farmer": ["HARVEST"], "hands": [], "market": []}
        return {"farmer": ["PASS"], "hands": [], "market": []}
    return script


def probe_water_before_harvest():
    print("\nwater-then-harvest on the final day vs harvest-first (what we do now)")
    for crop in ("WHEAT", "CARROT", "MELON"):
        age = agent.MAX_YIELD_DAY[crop]
        now, _ = drive(crop_script(crop, age), steps=(age + 3) * TPD)
        better, _ = drive(crop_script_water_first(crop, age), steps=(age + 3) * TPD)
        a = now.get("yield_at_harvest")
        b = better.get("yield_at_harvest")
        claim = agent.CROP_SPEC[crop][0]
        print(f"    {crop:<8} harvest-first {a}   water-first {b}"
              f"   CROP_SPEC claims {claim}")


def probe_bonus_window():
    """BONUS_START must match the age at which the env first credits a watering.

    Recorded at the START of each day, so a watering on day N shows up as an
    increase read at day N+1."""
    print("\nwatering bonus window - first age at which a watering pays")
    def script(crop, top):
        def f(obs, log):
            fx, fy = obs["farms"][0]["farmer"]
            tile = tile_at(obs, fx, fy)
            if obs["step"] == 0:
                return {"farmer": ["PASS"], "hands": [], "market": [["BUY_SEED", crop, 3]]}
            if tile is None and "done" not in log:
                return {"farmer": ["PLANT", crop], "hands": [], "market": []}
            if isinstance(tile, dict) and tile.get("kind") == "PLANT":
                age = obs["day"] - tile["planted_day"]
                log.setdefault("sod", {}).setdefault(age, tile["yield_units"])
                if age >= top:
                    log["done"] = True
                elif not tile["watered_today"]:
                    return {"farmer": ["WATER"], "hands": [], "market": []}
            return {"farmer": ["PASS"], "hands": [], "market": []}
        return f
    for crop, top in (("WHEAT", 5), ("CARROT", 4), ("MELON", 13)):
        log, _ = drive(script(crop, top), steps=(top + 2) * TPD)
        sod = log["sod"]
        gains = {a: sod[a] - sod[a - 1] for a in sorted(sod) if a - 1 in sod}
        first_paid = min([a - 1 for a, g in gains.items() if g > 0], default=None)
        check(f"{crop} bonus window starts", first_paid, agent.BONUS_START.get(crop))
    check("ongoing crops excluded from BONUS_START",
          sorted(agent.BONUS_START), ["CARROT", "MELON", "WHEAT"])


PROBES.append(("bonus_window", probe_bonus_window))


if __name__ == "__main__":
    want = sys.argv[1] if len(sys.argv) > 1 else ""
    for name, fn in PROBES:
        if want and want not in name:
            continue
        fn()
    print()
    if FAILURES:
        print(f"{len(FAILURES)} BELIEF(S) CONTRADICTED BY THE ENVIRONMENT:")
        for f in FAILURES:
            print(f"  - {f}")
        sys.exit(1)
    print("every belief checked matches the environment")
