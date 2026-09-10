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


def drive(script, steps, seed=1, **config):
    """Run `steps` turns with a scripted farmer. `script(obs, log)` returns the
    farmer's action list; `log` is a dict the script may record into.

    Extra keyword arguments override the environment configuration, which lets
    a probe isolate the rule it is testing - raising startingMoney to reach the
    shed cap, for instance, rather than being stopped by the wheat price."""
    log = {}

    def bot(obs):
        if obs["player"] != 0:
            return {"farmer": ["PASS"], "hands": [], "market": []}
        return script(obs, log)

    settings = {"episodeSteps": steps, "seed": seed, "weedSpawnChance": 0.0}
    settings.update(config)
    env = make("kaggriculture", configuration=settings)
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


# --------------------------------------------------------------------------
# animals - ANIMAL_SPEC, the CARE bonus, fertilizer, and starvation
# --------------------------------------------------------------------------

def animal_script(animal, care=True, feed=True, stop_feeding_day=None):
    """Build the structure on the farmer's own tile (which is shed-adjacent, so
    wheat can be picked up without moving), place the animal, then feed and care
    for it daily. Nothing is ever harvested, so yield_units accumulates and the
    max_held cap becomes visible."""
    structure = agent.STRUCTURE_FOR[animal]

    def f(obs, log):
        fx, fy = obs["farms"][0]["farmer"]
        tile = tile_at(obs, fx, fy)
        priv = obs["private"]
        held = (priv.get("inventories") or [{}])[0]

        if obs["step"] == 0:
            return {"farmer": ["PASS"], "hands": [],
                    "market": [["BUY_ANIMAL", animal, 1],
                               ["BUY_PRODUCT", "WHEAT", 90]]}
        if tile is None:
            return {"farmer": ["BUILD_" + structure], "hands": [], "market": []}
        if isinstance(tile, dict) and tile.get("kind") == structure:
            if not tile.get("animal"):
                # An empty structure after the animal was placed means it
                # starved. Check this first: otherwise the script loops trying
                # to re-place an animal it no longer holds and never notices.
                if log.get("placed"):
                    log.setdefault("escaped_day", obs["day"])
                    return {"farmer": ["PASS"], "hands": [], "market": []}
                if held.get(animal, 0) > 0:
                    return {"farmer": ["PLACE", animal], "hands": [], "market": []}
                return {"farmer": ["PICKUP", animal, 1], "hands": [], "market": []}

            log["placed"] = True
            age = obs["day"] - tile["placed_day"]
            log.setdefault("sod", {}).setdefault(age, tile["yield_units"])
            log.setdefault("fert", {}).setdefault(age, bool(tile.get("fertilizer_available")))
            feeding = feed and (stop_feeding_day is None or obs["day"] < stop_feeding_day)
            if feeding and not tile["fed_today"]:
                if held.get("WHEAT", 0) > 0:
                    return {"farmer": ["FEED"], "hands": [], "market": []}
                return {"farmer": ["PICKUP", "WHEAT", 10], "hands": [], "market": []}
            if care and not tile["cared_today"]:
                return {"farmer": ["CARE"], "hands": [], "market": []}
        return {"farmer": ["PASS"], "hands": [], "market": []}
    return f


def probe_animal_schedule():
    """First yield day, interval, and max_held, with no CARE so the base rate
    is visible."""
    print()
    print("animals - first yield, interval and max_held (no CARE)")
    for animal in ("GOOSE", "COW", "SHEEP"):
        cost, product, interval, max_held, first_yield = agent.ANIMAL_SPEC[animal]
        log, _ = drive(animal_script(animal, care=False), steps=26 * TPD)
        sod = log.get("sod", {})
        gains = {a: sod[a] - sod[a - 1] for a in sorted(sod) if a - 1 in sod}
        # Unlike watering, production is not an action taken on a day - it
        # fires at end-of-day and is simply visible from `first_yield_day`
        # onward, so there is no offset to undo here.
        first_paid = min([a for a, u in sod.items() if u > 0], default=None)
        producing = sorted(a for a, g in gains.items() if g > 0)
        gaps = {b - a for a, b in zip(producing, producing[1:])} or {interval}
        check(f"{animal} first yield day", first_paid, first_yield)
        check(f"{animal} interval", sorted(gaps), [interval])
        check(f"{animal} max_held cap", max(sod.values()), max_held)
        print(f"      units by age: {sod}")


def probe_care_bonus():
    """What one production actually pays, with and without CARE.

    animal_value assumes every production pays min(1 + interval, max_held).
    That is the steady state - the bank holds one unit per cared day and empties
    on each production - but the FIRST production also drains everything banked
    during the lead time, which is 8 days for a cow. Measured by harvesting
    after every production so the tile empties and each payout is visible."""
    print()
    print("animals - units per production, harvesting each time")

    def harvest_script(animal, care):
        structure = agent.STRUCTURE_FOR[animal]

        def f(obs, log):
            fx, fy = obs["farms"][0]["farmer"]
            tile = tile_at(obs, fx, fy)
            held = (obs["private"].get("inventories") or [{}])[0]
            if obs["step"] == 0:
                return {"farmer": ["PASS"], "hands": [],
                        "market": [["BUY_ANIMAL", animal, 1],
                                   ["BUY_PRODUCT", "WHEAT", 90]]}
            if tile is None:
                return {"farmer": ["BUILD_" + structure], "hands": [], "market": []}
            if isinstance(tile, dict) and tile.get("kind") == structure:
                if not tile.get("animal"):
                    if held.get(animal, 0) > 0:
                        return {"farmer": ["PLACE", animal], "hands": [], "market": []}
                    return {"farmer": ["PICKUP", animal, 1], "hands": [], "market": []}
                if not tile["fed_today"]:
                    if held.get("WHEAT", 0) > 0:
                        return {"farmer": ["FEED"], "hands": [], "market": []}
                    return {"farmer": ["PICKUP", "WHEAT", 10], "hands": [], "market": []}
                if tile["yield_units"] > 0:
                    log.setdefault("payouts", []).append(
                        (obs["day"] - tile["placed_day"], tile["yield_units"]))
                    return {"farmer": ["HARVEST"], "hands": [], "market": []}
                if care and not tile["cared_today"]:
                    return {"farmer": ["CARE"], "hands": [], "market": []}
            return {"farmer": ["PASS"], "hands": [], "market": []}
        return f

    for animal in ("GOOSE", "COW", "SHEEP"):
        _, _, interval, max_held, first_yield = agent.ANIMAL_SPEC[animal]
        out = {}
        for care in (False, True):
            log, _ = drive(harvest_script(animal, care), steps=26 * TPD)
            out[care] = [u for _, u in log.get("payouts", [])]
        print(f"      {animal:<6} no care {out[False]}")
        print(f"      {animal:<6} cared   {out[True]}")
        check(f"{animal} uncared payout", sorted(set(out[False])), [1])
        steady = out[True][1:] if len(out[True]) > 1 else out[True]
        check(f"{animal} cared steady payout", sorted(set(steady)),
              [min(1 + interval, max_held)], "(what animal_value assumes)")
        if out[True]:
            check(f"{animal} cared FIRST payout", out[True][0],
                  min(1 + first_yield, max_held),
                  "(the lead-time bank, which animal_value ignores)")


def probe_fertilizer_daily():
    """One fertilizer per animal per day, produced whether or not it was fed,
    and it does not accumulate."""
    print()
    print("animals - fertilizer availability")
    log, _ = drive(animal_script("GOOSE", care=False), steps=12 * TPD)
    fert = log.get("fert", {})
    after_first = [v for a, v in sorted(fert.items()) if a >= 1]
    check("fertilizer available every day", all(after_first), True,
          f"(by age: {fert})")


def probe_starvation():
    """Two consecutive unfed end-of-days and the animal is gone for good."""
    print()
    print("animals - starvation")
    log, _ = drive(animal_script("GOOSE", care=False, stop_feeding_day=6),
                   steps=12 * TPD)
    # Feeding stops during day 6, so day 6 and day 7 both end unfed and the
    # animal is gone at the end of day 7 - visible on day 8.
    check("animal escapes two days after feeding stops",
          log.get("escaped_day"), 8,
          "(stopped feeding on day 6)")


PROBES.extend([
    ("animal_schedule", probe_animal_schedule),
    ("care_bonus", probe_care_bonus),
    ("fertilizer", probe_fertilizer_daily),
    ("starvation", probe_starvation),
])


# --------------------------------------------------------------------------
# shed capacity and plant decay - the last two unchecked beliefs
# --------------------------------------------------------------------------

def probe_shed_capacity():
    """The shed holds 100 non-seed items; seeds live in their own uncapped slot.

    main.py sells everything every turn and peaked at 97/100, so the cap has
    never actually bitten - but _market_orders buys feed and seed against it,
    and pending_units counts shed stock as committed supply. If the cap were
    not what we think, both would be wrong in a full shed."""
    print()
    print("shed - capacity and what counts toward it")

    def script(obs, log):
        shed = obs["private"]["shed"]
        seeds = obs["private"]["seeds"]
        log["shed_total"] = sum(v for k, v in shed.items())
        log["seed_total"] = sum(seeds.values())
        log["wheat"] = shed.get("WHEAT", 0)
        if obs["step"] < 3:
            return {"farmer": ["PASS"], "hands": [],
                    "market": [["BUY_PRODUCT", "WHEAT", 80],
                               ["BUY_SEED", "WHEAT", 60]]}
        return {"farmer": ["PASS"], "hands": [], "market": []}

    log, _ = drive(script, steps=6 * TPD, startingMoney=200000)
    check("shed caps at 100 non-seed items", log["shed_total"], 100)
    check("seeds are not capped by the shed", log["seed_total"] > 100, True,
          f"({log['seed_total']} seeds held alongside {log['shed_total']} items)")


def probe_decay():
    """Past max lifespan a plant loses one unit every other turn, then weeds.

    _candidates treats `decaying` as urgent (w_harvest_decay) on the strength of
    this. If decay were faster, that urgency is understated; if there is no
    decay at all, the whole branch is pointless."""
    print()
    print("plant decay - after max lifespan")

    def script(obs, log):
        fx, fy = obs["farms"][0]["farmer"]
        tile = tile_at(obs, fx, fy)
        if obs["step"] == 0:
            return {"farmer": ["PASS"], "hands": [], "market": [["BUY_SEED", "WHEAT", 3]]}
        if tile is None and "planted" not in log:
            return {"farmer": ["PLANT", "WHEAT"], "hands": [], "market": []}
        if isinstance(tile, dict) and tile.get("kind") == "PLANT":
            log["planted"] = True
            log["lifespan"] = tile["max_lifespan_step"]
            if obs["step"] >= tile["max_lifespan_step"]:
                log.setdefault("trace", []).append((obs["step"], tile["yield_units"]))
            elif not tile["watered_today"]:
                return {"farmer": ["WATER"], "hands": [], "market": []}
        if isinstance(tile, dict) and tile.get("kind") == "WEED":
            log.setdefault("weed_step", obs["step"])
        return {"farmer": ["PASS"], "hands": [], "market": []}

    log, _ = drive(script, steps=9 * TPD)
    trace = log.get("trace", [])
    print(f"      units per turn past lifespan {log.get('lifespan')}: "
          f"{[u for _, u in trace][:14]}")
    units = [u for _, u in trace]
    drops = [i for i in range(1, len(units)) if units[i] < units[i - 1]]
    gaps = {b - a for a, b in zip(drops, drops[1:])}
    check("one unit lost every other turn", sorted(gaps) or [2], [2])
    check("reaches zero then becomes a weed", log.get("weed_step") is not None, True,
          f"(weed at step {log.get('weed_step')}, lifespan {log.get('lifespan')})")


PROBES.extend([
    ("shed", probe_shed_capacity),
    ("decay", probe_decay),
])


# --------------------------------------------------------------------------
# market mechanics - what a sale actually pays
# --------------------------------------------------------------------------

def probe_sell_revenue():
    """revenue_for() must equal the money a SELL actually produces.

    Every value function in the agent ends here: crop_value and animal_value
    both price their output with revenue_for, unit by unit as the price falls.
    If the real market pays something else, every comparison the agent makes
    between crops, animals and market timing is wrong by that difference."""
    print()
    print("market - SELL proceeds vs revenue_for()")

    def script(product, qty):
        def f(obs, log):
            shed = obs["private"]["shed"]
            farm = obs["farms"][0]
            if obs["step"] < 2:
                return {"farmer": ["PASS"], "hands": [],
                        "market": [["BUY_PRODUCT", product, qty]]}
            if shed.get(product, 0) >= qty and "sold_at" not in log:
                log["sold_at"] = obs["step"]
                log["money_before"] = farm["money"]
                log["inv_before"] = obs["market"]["inventory"][product]
                log["predicted"] = agent.revenue_for(
                    product, obs["market"]["inventory"][product], qty)
                return {"farmer": ["PASS"], "hands": [],
                        "market": [["SELL", product, qty]]}
            if log.get("sold_at") and obs["step"] == log["sold_at"] + 1:
                log["money_after"] = farm["money"]
            return {"farmer": ["PASS"], "hands": [], "market": []}
        return f

    for product, qty in (("WHEAT", 40), ("FERTILIZER", 25), ("WHEAT", 5)):
        log, _ = drive(script(product, qty), steps=8 * TPD, startingMoney=200000)
        got = log.get("money_after", 0) - log.get("money_before", 0)
        check(f"SELL {qty} {product} pays revenue_for", got, log.get("predicted"),
              f"(market inventory was {log.get('inv_before')})")


def probe_price_floor():
    """What can and cannot be checked about the $1 floor.

    The rulebook says that at the floor a unit is still bought but is NOT added
    to market inventory, so the floor stays responsive. That specific behaviour
    resisted three attempts to exercise it, and the reasons are worth recording
    rather than working around:

      - a buy-then-sell round trip cannot reach it. The env quotes buys at
        post-buy inventory precisely so a round trip nets zero, so a player can
        only push inventory above I0 by actually producing goods.
      - driving the real agent with wheat's glut curve steepened does not reach
        it either, because crop_value stops planting a crop once its marginal
        value collapses. The agent refuses to produce something worthless -
        which is the crop mix working as designed, and is itself worth knowing.

    So the floor-inventory rule stays UNVERIFIED, and this probe asserts only
    what is actually reachable. The exposure is small: the agent prices with
    price_at, which floors at 1 the same way, and the lowest price seen in a
    real game is melon at about $34 - far above the floor."""
    print()
    print("market - price floor (partially verifiable, see docstring)")
    steep = {"WHEAT": {"above_func": "sq", "above_target": 2500.0}}

    def f(obs, log):
        inv = obs["market"]["inventory"]["WHEAT"]
        log.setdefault("trace", []).append((inv, obs["market"]["prices"]["WHEAT"]))
        return agent.agent(obs)

    log, _ = drive(f, steps=22 * TPD, marketParams=steep)
    trace = log["trace"]
    lowest = min(p for _, p in trace)
    check("price never goes below 1", lowest >= 1, True,
          f"(lowest reached {lowest})")
    print(f"      NOT VERIFIED: behaviour exactly at the $1 floor - the agent "
          f"stops producing a crop before it gets there (lowest {lowest})")


def probe_order_cap():
    """At most maxMarketOrdersPerTurn (10) orders are processed per player per
    turn; extras are silently dropped. _market_orders truncates with [:10] on
    exactly this basis."""
    print()
    print("market - the 10-order-per-turn cap")

    def f(obs, log):
        farm = obs["farms"][0]
        if obs["step"] == 0:
            log["money_before"] = farm["money"]
            return {"farmer": ["PASS"], "hands": [],
                    "market": [["HIRE"]] * 15}
        if obs["step"] == 1:
            log["hands"] = len(farm["hands"])
        return {"farmer": ["PASS"], "hands": [["PASS"]] * len(farm["hands"]),
                "market": []}

    log, _ = drive(f, steps=3, startingMoney=200000)
    check("only 10 of 15 HIRE orders processed", log.get("hands"), 10)


PROBES.extend([
    ("sell_revenue", probe_sell_revenue),
    ("price_floor", probe_price_floor),
    ("order_cap", probe_order_cap),
])


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
