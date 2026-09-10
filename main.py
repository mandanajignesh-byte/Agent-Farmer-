"""Kaggriculture agent.

Every turn each worker (the farmer plus any hired hands) is given one action.
Candidate tiles are scored by a weighted sum of features and the lowest score
wins; workers claim tiles one at a time so no two walk to the same place.

All tuneable behaviour lives in PARAMS so the weights can be searched rather
than hand-picked - see tune.py. The defaults below reproduce v6 exactly.
"""
import math


# ---------------------------------------------------------------------------
# Market price model, validated against 270 real price points with zero error.
# Inlined rather than imported: a submission is a single main.py with no
# sibling modules on the path, so `import market` fails there while working
# perfectly in local testing.
# ---------------------------------------------------------------------------


MARKET_I0 = 10_000

# base, T, below_func, below_target, above_func, above_target
MARKET_PARAMS = {
    "WHEAT": (25, 400, "sqrt", 0.80, "log", 0.20),
    "CARROT": (35, 450, "hinge", 1.00, "sqrt", 0.70),
    "TOMATO": (60, 200, "hinge", 0.40, "sqrt", 0.60),
    "STRAWBERRY": (120, 100, "sqrt", 0.70, "linear", 1.60),
    "MELON": (250, 300, "log", 0.20, "sq", 3.60),
    "EGG": (50, 332, "hinge", 0.40, "log", 0.20),
    "MILK": (160, 122, "sqrt", 0.60, "linear", 1.60),
    "WOOL": (200, 105, "log", 0.20, "sq", 3.20),
    "FERTILIZER": (100, 200, "linear", 0.40, "linear", 0.40),
}

SHAPES = {
    "linear": lambda x, t: x,
    "sq": lambda x, t: x * x,
    "sqrt": lambda x, t: math.sqrt(x),
    "log": lambda x, t: math.log(1 + x),
    "log10": lambda x, t: math.log10(1 + x),
    "hinge": lambda x, t: (x / t) + 8 * max(0.0, x / t - 1) ** 2,
}


def price_at(product, inventory):
    """Price when market inventory sits at `inventory`. Scarcity raises it,
    glut lowers it, and the two sides use different curves - wheat barely sags
    on glut but spikes on scarcity, while melon does the opposite."""
    if product not in MARKET_PARAMS:
        return 0
    base, t, below_func, below_target, above_func, above_target = MARKET_PARAMS[product]
    gap = abs(inventory - MARKET_I0)
    if gap == 0:
        return base

    scarce = inventory < MARKET_I0
    func, target = (below_func, below_target) if scarce else (above_func, above_target)
    shape = SHAPES[func]
    amp = target * base / shape(t, t)
    move = amp * shape(gap, t)
    return max(1, round(base + move if scarce else base - move))


def revenue_for(product, inventory, units):
    """Total takings for selling `units`, priced one at a time as the price
    falls. This is what a marginal tile is actually worth, not units * quote."""
    total = 0
    for i in range(int(units)):
        total += price_at(product, inventory + i)
    return total


TURNS_PER_DAY = 24
SEED_COST = {"WHEAT": 10, "CARROT": 20, "TOMATO": 50, "STRAWBERRY": 100, "MELON": 80}
MAX_YIELD_DAY = {"WHEAT": 4, "CARROT": 3, "TOMATO": 11, "STRAWBERRY": 16, "MELON": 10}
# Watering only adds yield from half-way to max yield onward. Before that it is
# pure survival, and survival tolerates every other day - so an early watering
# on a healthy plant buys nothing at all.
#
# The environment derives this window from ITS max_yield_day, which is not the
# day we choose to harvest: melon caps at 6 units on day 10 so we pick then, but
# the rulebook's max_yield_day is 12 and the window therefore opens at 6, not 5.
# Deriving the window from MAX_YIELD_DAY above put melon's start a day early and
# spent one wasted watering on every melon tile. Ongoing crops are absent on
# purpose - watering never adds yield to them at all, it only keeps them alive,
# which the `dying` rescue already covers. Every value here is probed against
# the environment in test_model.py.
BONUS_START = {"WHEAT": 2, "CARROT": 2, "MELON": 6}

HANDS_PER_DAY = 8
SEED_BUFFER = HANDS_PER_DAY + 2
# Quadrants cost 1k, 2k, 4k. Land lost money at v6 and won +$6,590 at v10 -
# nothing about the price changed, the agent got good enough to work the tiles.
# Three quadrants and eight hands were swept together, since the two are
# coupled: optimal staffing rises with area, and four quadrants loses at every
# staffing level because fib cost caps the workforce before 100 tiles can be
# tended. Still hardcoded, and still worth replacing with a value computed from
# spare labour the way crops and animals now are.
MAX_QUADRANTS = 3
LAND_LAST_DAY = 20

WATER, HARVEST, PLANT, DIG, PASS = "WATER", "HARVEST", "PLANT", "DIG", "PASS"
COLLECT = "COLLECT_FERTILIZER"
DROP = "DROP"
WEED = "WEED"

SEASON_DAYS = 30

# (yield without fertilizer, seed cost, days to reach that yield)
CROP_SPEC = {
    "WHEAT": (4, 10, 4),
    "CARROT": (3, 20, 3),
    "TOMATO": (4, 50, 11),
    "STRAWBERRY": (4, 100, 16),
    "MELON": (6, 80, 10),
}

# (purchase price, product sold, days between yields, max units held,
#  days from placement to first yield)
ANIMAL_SPEC = {
    "GOOSE": (300, "EGG", 1, 4, 4),
    "COW": (400, "MILK", 2, 6, 8),
    "SHEEP": (500, "WOOL", 3, 6, 6),
}


def pending_units(farm, private, crop):
    """What we are already committed to selling of this crop - shed stock plus
    the expected yield of everything currently in the ground. A new tile's
    output arrives behind all of it, so that is the inventory level it gets
    priced at."""
    units = private["shed"].get(crop, 0)
    for row in farm["tiles"]:
        for tile in row:
            if isinstance(tile, dict) and tile.get("kind") == PLANT and tile["crop"] == crop:
                units += CROP_SPEC[tile["crop"]][0]
    return units


def crop_value(crop, inventory, pending, days_left=None):
    """Profit per tile per day for ONE MORE tile of this crop, pricing its yield
    unit by unit as it pushes the price down.

    This is what makes tile counts self-balancing: plant more melon and melon's
    marginal value falls until carrot overtakes it, then wheat overtakes carrot.
    No target count has to be chosen, and the mix re-balances on its own if an
    opponent floods a market."""
    yield_units, seed_cost, days = CROP_SPEC[crop]
    if days_left is not None and days_left < days:
        # Cannot mature before the season ends, and we only harvest at
        # age >= MAX_YIELD_DAY, so it would never be picked even partially.
        # The seed is simply spent.
        return -seed_cost
    revenue = revenue_for(crop, inventory.get(crop, MARKET_I0) + pending, yield_units)
    actions = 1 + 1 / days  # a watering a day, plus the harvest at the end
    return (revenue - seed_cost) / days - PARAMS["w_action_cost"] * actions


def animal_value(animal, prices, days_left, inventory=None, herd=0, shed=None):
    """Profit per tile per day, with the purchase amortised over the season that
    is left. Late in the game that term explodes and the value goes negative, so
    the agent stops buying without needing a cutoff date - it stops exactly at
    the payback period.

    Income is priced marginally, the same way crop_value does it: a herd of a
    dozen animals produces hundreds of units over a season and drives its own
    prices down. Fertilizer especially - no town shop consumes it, so the only
    thing draining that market is other players buying."""
    cost, product, interval, max_held, first_yield = ANIMAL_SPEC[animal]
    inventory = inventory or {}
    shed = shed or {}

    # what our existing herd will still add before this animal's output lands
    made_per_animal = max(days_left, 0)
    prod_pending = shed.get(product, 0) + herd * made_per_animal * (1 + interval) / interval
    fert_pending = shed.get("FERTILIZER", 0) + herd * made_per_animal

    # CARE banks one unit a day and pays the whole bank out on the next
    # production, so an animal cared for every day yields 1 + interval per
    # interval instead of 1 - triple for a cow, four times for a sheep. Capped
    # by max_held, which the bank cannot exceed.
    per_event = min(1 + interval, max_held)
    units = revenue_for(product, inventory.get(product, MARKET_I0) + prod_pending, per_event)
    produce = units / interval
    # Every surviving animal yields one fertilizer a day, free, fed or not -
    # and fertilizer's base price of $100 makes that stream comparable to the
    # milk. Valuing an animal on its product alone undercounts it by about
    # half, which is why the herd never grew.
    fertilizer = revenue_for(
        "FERTILIZER", inventory.get("FERTILIZER", MARKET_I0) + fert_pending, 1)
    feed = prices.get("WHEAT", 25)  # bought, not grown - tiles cost actions
    actions = 3 + 1 / interval  # feed, care and collect daily; harvest each interval

    # An animal produces nothing for its first `first_yield` days - 4 for a
    # goose, 6 for a sheep, 8 for a cow - but eats and takes actions from the
    # day it is placed. Ignoring that overvalued every late purchase by its
    # whole lead time: a cow bought on day 24 never yields once, and the old
    # formula happily recommended it. Fertilizer is exempt, since a surviving
    # animal drops one a day from the start whether it is producing or not.
    productive = max(0, days_left - first_yield)
    produce *= productive / max(days_left, 1)

    return (produce + fertilizer - feed - cost / max(days_left, 1)
            - PARAMS["w_action_cost"] * actions)

def daily_burn(farm, private, prices, ranked, livestock):
    """What the farm spends in a day at current prices.

    Land is bought with the same dollars that buy seed, animals and feed, so
    what is left after a purchase has to carry the farm until income arrives.
    A flat $500 reserve did not: the agent bought land on turn one, fell to
    $204 by day 3, and then wanted seed on 510 of 720 turns without being able
    to pay for any of it. Income only arrived on day 12.

    Every term is read from the game rather than assumed - hire cost from the
    fib schedule, seed from what the ranking would actually buy, feed from the
    live wheat price - so the only free parameter is how many days of it to
    hold back, which tuning decides."""
    hires, a, b = 0, 1, 1
    for _ in range(HANDS_PER_DAY):
        hires += a
        a, b = b, a + b
    seed = sum(SEED_COST[c] * SEED_BUFFER for c in ranked[:2])
    feed = livestock * 3 * prices.get("WHEAT", 25)
    return hires + seed + feed


# Lower score wins. The first six were hardcoded priority levels 1-6 scaled by
# the old PRIORITY_WEIGHT of 2; there was never a reason for them to be evenly
# spaced integers, so they are now searchable.
PARAMS = {
    "w_water_urgent": 5.907,
    "w_harvest_decay": 2.157,
    "w_harvest_ripe": 4.517,
    # Watering inside the bonus window earns a unit of yield; outside it, on a
    # plant in no danger, it earns nothing and only costs the walk.
    "w_water_bonus": 7.903,
    "w_water_idle": 30.0,
    "w_plant": 10.623,
    "w_dig": 10.988,
    "w_dist": 1.0,
    "w_dist_sq": 0.0,
    # PLANT only. Where an existing plant sits is already fixed, but choosing
    # where to plant fixes every future trip to that tile.
    "w_shed": -0.499,
    # An unfed animal is gone permanently and cost $300-500, so feeding
    # outranks everything a crop can ask for.
    "w_feed": 0.562,
    "w_harvest_animal": 3.5,
    "w_collect": 3.5,
    "w_care": 3.0,
    "w_drop": 4.0,
    "w_place": -0.02,
    "w_build": 8.0,
    "w_pickup": 8.238,
    # Pens are serviced every day, so where one is built fixes its running cost
    # for the rest of the season - the same argument as w_shed for planting.
    "w_pen_shed": 1.096,
    # Days of running costs to keep in the bank before buying land. Starts at
    # zero so the search has to turn it on, the same way every other feature
    # had to earn its place.
    "w_land_reserve": 0.0,
    # What a worker-action is worth. Charged against every tile use so a crop
    # (about 1 action/day) and an animal (about 2.5) compete on equal terms.
    # Dollar value the last bonus watering must beat before it is taken ahead
    # of the harvest. Zero means always take it.
    "w_final_water": 0.0,
    "w_action_cost": 0.0,
}


def _distance(ax, ay, bx, by):
    return abs(ax - bx) + abs(ay - by)


def _shed_distance(x, y, board_size):
    """Steps to the nearest tile the shed can be reached from."""
    half = board_size // 2
    return min(
        _distance(x, y, cx, cy) for cx in (half - 1, half) for cy in (half - 1, half)
    )


def _step_toward(fx, fy, tx, ty):
    if tx > fx:
        return "EAST"
    if tx < fx:
        return "WEST"
    if ty > fy:
        return "SOUTH"
    return "NORTH"


def _final_water_pays(crop, age, inventory, pending):
    """Is the last bonus watering worth the tile-turnover it costs?

    On the max-yield day a tile is both ripe and still able to gain a unit.
    Watering first collects that unit but delays the harvest, and the delay
    costs a slice of the tile's next cycle. Measured over 10 games: melon gains
    19% of its units for no lost harvests, because a 10-day tile has no cycle
    to slow - while carrot gains 48% per tile and loses 28% of its harvests,
    which is close to a wash.

    So the test is the unit's own worth, not the crop's name. The threshold is
    a tuned dollar figure starting at zero, which reproduces "always water"
    until the search decides otherwise."""
    if age < MAX_YIELD_DAY[crop]:
        return True  # mid-window watering displaces no harvest at all
    return price_at(crop, inventory.get(crop, MARKET_I0) + pending.get(crop, 0))         > PARAMS["w_final_water"]


def _candidates(obs, farm, private):
    """Every actionable tile as (param_key, action, x, y, yield_units)."""
    step, hour = obs["step"], obs["hour"]
    seeds = private["seeds"]
    can_plant = TURNS_PER_DAY - hour >= 2

    inventory = obs["market"]["inventory"]
    days_left = SEASON_DAYS - obs["day"]
    pending = {c: pending_units(farm, private, c) for c in CROP_SPEC}

    found, plantable = [], []
    empty_pens = animals_placed = unfed = 0
    for y, row in enumerate(farm["tiles"]):
        for x, tile in enumerate(row):
            if tile == "LOCKED":
                continue
            if tile is None:
                if can_plant:
                    plantable.append((x, y))
                continue
            if not isinstance(tile, dict):
                continue
            if tile.get("kind") == WEED:
                found.append(("w_dig", [DIG], x, y, 0, None))
                continue
            if tile.get("kind") in ("COOP", "PASTURE"):
                if tile.get("animal"):
                    animals_placed += 1
                    unfed += not tile["fed_today"]
                else:
                    empty_pens += 1
                found += _pen_jobs(obs, tile, x, y)
                continue
            if tile.get("kind") != PLANT:
                continue

            lifespan = tile["max_lifespan_step"]
            decaying = lifespan != -1 and step >= lifespan
            ripe = obs["day"] - tile["planted_day"] >= MAX_YIELD_DAY[tile["crop"]]
            units = tile["yield_units"]

            # Rescue only pays while watering still earns something. Past the
            # bonus window a dying plant's yield no longer grows, so spending an
            # action to keep it alive preserves a shrinking number - banking the
            # harvest now is strictly better. Watering on the max-yield day
            # itself still adds a unit, so the window is inclusive.
            age = obs["day"] - tile["planted_day"]
            water_earns = age <= MAX_YIELD_DAY[tile["crop"]]
            dying = tile["consecutive_unwatered"] >= 1 and not tile["watered_today"]

            crop = tile["crop"]
            earning = (crop in BONUS_START
                       and BONUS_START[crop] <= age <= MAX_YIELD_DAY[crop])

            if dying and (water_earns or units == 0):
                found.append(("w_water_urgent", [WATER], x, y, 0, None))
            elif units > 0 and decaying:
                found.append(("w_harvest_decay", [HARVEST], x, y, units, None))
            elif not tile["watered_today"] and earning and _final_water_pays(
                    tile["crop"], age, inventory, pending):
                # The last day of the bonus window is also the first day the
                # tile counts as ripe, and harvesting used to win that tie - so
                # every one-time crop was picked one watering short of the yield
                # CROP_SPEC promises. Verified against the environment: wheat
                # 3 -> 4, carrot 2 -> 3, melon 5 -> 6. Decay is still a full day
                # away, so the harvest simply happens on a later turn.
                found.append(("w_water_bonus", [WATER], x, y, 0, None))
            elif units > 0 and ripe:
                found.append(("w_harvest_ripe", [HARVEST], x, y, units, None))
            elif not tile["watered_today"]:
                found.append(("w_water_idle", [WATER], x, y, 0, None))

    # Planting more tiles in a turn than we hold seeds for makes every PLANT
    # that turn fail, not just the surplus ones - so each planned planting must
    # be backed by a seed we actually have.
    budget = {c: seeds.get(c, 0) for c in CROP_SPEC}

    prices = obs["market"]["prices"]
    _av = lambda a: animal_value(a, prices, days_left, inventory,
                                 animals_placed, private["shed"])
    best_animal = max(ANIMAL_SPEC, key=_av)
    animal_worth = _av(best_animal)

    # A pen only pays once an animal stands in it, so build them a couple ahead
    # of demand rather than covering the farm in empty structures.
    for x, y in plantable:
        if animal_worth > 0 and empty_pens < 2:
            crop_best = max((crop_value(c, inventory, pending[c], days_left)
                             for c in CROP_SPEC), default=0)
            if animal_worth > crop_best:
                empty_pens += 1
                found.append(("w_build", ["BUILD_" + STRUCTURE_FOR[best_animal]],
                              x, y, 0, None))
                continue
        affordable = [c for c in CROP_SPEC if budget[c] > 0]
        best = max(
            affordable,
            key=lambda c: crop_value(c, inventory, pending[c], days_left),
            default=None,
        )
        # Nothing left that can mature in time - stop planting entirely and
        # leave the workers free to harvest and sell.
        if best is None or crop_value(best, inventory, pending[best], days_left) <= 0:
            break
        budget[best] -= 1
        # this tile's own output crowds the next one
        pending[best] += CROP_SPEC[best][0]
        found.append(("w_plant", [PLANT, best], x, y, 0, None))
    # Restocking is pure overhead - it feeds no animal and grows no crop - so
    # ask for it only when the herd actually needs more wheat than the workers
    # are already carrying, and offer a single trip rather than one per shed
    # tile. Collect a full day's feed at once so one walk covers the herd.
    shed = private["shed"]
    board = len(farm["tiles"])

    # SELL draws from the shed, and a worker's inventory only reaches the shed
    # at end of day - after which, on the final day, there is no turn left to
    # sell it. Anything still being carried then scores nothing, so on the last
    # day it is worth walking it in. One candidate per item type actually held,
    # so only a worker carrying that item takes the job.
    if obs["day"] >= SEASON_DAYS - 1:
        held = set()
        for inv in (private.get("inventories") or []):
            held.update(k for k, v in inv.items() if v > 0 and k in MARKET_PARAMS)
        for item in held:
            sx, sy = _shed_tiles(board)[0]
            found.append(("w_drop", [DROP], sx, sy, 0, item))
    carried = sum(inv.get("WHEAT", 0) for inv in (private.get("inventories") or []))
    # One fetch per turn meant a single worker carried the whole herd's feed and
    # walked it round every pen. As the herd grew, animals starved waiting -
    # six escaped in a season at seven animals, $400 each plus their output.
    # Offer a trip per shed-access tile so several workers can share the round.
    hungry = unfed - carried
    if hungry > 0 and shed.get("WHEAT", 0) > 0:
        for sx, sy in _shed_tiles(board)[:min(4, hungry)]:
            found.append(("w_pickup", ["PICKUP", "WHEAT", max(hungry, 1)], sx, sy, 0, None))
    for animal in ANIMAL_SPEC:
        if shed.get(animal, 0) > 0 and empty_pens:
            sx, sy = _shed_tiles(board)[0]
            found.append(("w_pickup", ["PICKUP", animal, 1], sx, sy, 0, None))
    return found


STRUCTURE_FOR = {"GOOSE": "COOP", "COW": "PASTURE", "SHEEP": "PASTURE"}


def _pen_jobs(obs, tile, x, y):
    """Jobs on a coop or pasture. Feeding needs wheat in hand, so it carries a
    requirement the assignment step checks against that worker's inventory."""
    jobs = []
    animal = tile.get("animal")
    if not animal:
        # empty pen - place an animal we are carrying
        for name, structure in STRUCTURE_FOR.items():
            if structure == tile["kind"]:
                jobs.append(("w_place", ["PLACE", name], x, y, 0, name))
        return jobs
    if not tile["fed_today"]:
        jobs.append(("w_feed", ["FEED"], x, y, 0, "WHEAT"))
    # Caring a fed animal banks a unit that pays out on its next production -
    # roughly $160-200 for one action, the best return of anything on the farm.
    # The bank is only credited if the animal is also fed that day.
    if not tile["cared_today"]:
        jobs.append(("w_care", ["CARE"], x, y, 0, None))
    if tile["yield_units"] > 0:
        jobs.append(("w_harvest_animal", [HARVEST], x, y, tile["yield_units"], None))
    # One fertilizer per animal per day, free, produced whether or not it was
    # fed. It does NOT accumulate - miss a day and that unit is gone - and at a
    # $100 base it is worth more per unit than milk. animal_value has been
    # counting this income all along; until now nothing ever collected it.
    if tile.get("fertilizer_available"):
        jobs.append(("w_collect", [COLLECT], x, y, 0, None))
    return jobs


def _shed_tiles(board_size):
    half = board_size // 2
    return [(cx, cy) for cx in (half - 1, half) for cy in (half - 1, half)]


def _score(candidate, wx, wy, board_size):
    key, _action, x, y, _units, _needs = candidate
    d = _distance(wx, wy, x, y)
    # Linear is well motivated - walking N steps costs exactly N actions, with
    # no economy of scale. The quadratic term tests whether a long trip carries
    # extra risk the linear cost cannot express: over ten turns of walking,
    # plants dry out and another worker may take the target first.
    score = PARAMS[key] + PARAMS["w_dist"] * d + PARAMS["w_dist_sq"] * d * d
    if key == "w_plant":
        score += PARAMS["w_shed"] * _shed_distance(x, y, board_size)
    elif key == "w_build":
        score += PARAMS["w_pen_shed"] * _shed_distance(x, y, board_size)
    return score


def _assign_actions(obs, farm, private):
    """One action per worker. Repeatedly takes the best (worker, tile) pair
    available anywhere, rather than letting workers pick in a fixed order - the
    farmer choosing first could otherwise take a tile a hand was standing on and
    send that hand walking."""
    board_size = len(farm["tiles"])
    workers = [tuple(farm["farmer"])] + [tuple(h) for h in farm["hands"]]
    carrying = private.get("inventories") or [{}] * len(workers)
    pool = _candidates(obs, farm, private)

    actions = [[PASS] for _ in workers]
    waiting = set(range(len(workers)))

    while waiting and pool:
        best = None
        for w in waiting:
            wx, wy = workers[w]
            held = carrying[w] if w < len(carrying) else {}
            for c, candidate in enumerate(pool):
                if candidate[5] and not held.get(candidate[5], 0):
                    continue
                score = _score(candidate, wx, wy, board_size)
                if best is None or score < best[0]:
                    best = (score, w, c)

        if best is None:
            break  # everything left needs an item nobody is carrying
        _score_, w, c = best
        wx, wy = workers[w]
        _key, action, tx, ty, _units, _needs = pool.pop(c)
        waiting.discard(w)

        if (tx, ty) != (wx, wy):
            actions[w] = [_step_toward(wx, wy, tx, ty)]
        else:
            actions[w] = list(action)

    return actions[0], actions[1:]


def _market_orders(obs, farm, private):
    # Hands vanish at end of day and must be rehired each morning. fib(n) makes
    # the first few nearly free: 1, 1, 2, 3, 5, 8 ...
    # Selling first, always. Orders are capped at 10 per turn, so anything
    # placed ahead of SELL can crowd it out - and when cash runs low, failed
    # HIRE orders repeat every turn and do exactly that, starving the agent of
    # the income it needs to recover.
    pens_now = [t for row in farm["tiles"] for t in row
                if isinstance(t, dict) and t.get("animal")]
    keep_wheat = len(pens_now) * 3
    orders = []
    for item, qty in private["shed"].items():
        if qty <= 0 or item in ANIMAL_SPEC:
            continue
        if item == "WHEAT":
            qty -= keep_wheat  # feed stock is not for sale
        if qty > 0:
            orders.append(["SELL", item, qty])
    orders += [["HIRE"]] * max(0, HANDS_PER_DAY - farm["hires_today"])

    prices = obs["market"]["prices"]
    days_left = SEASON_DAYS - obs["day"]
    pens = [t for row in farm["tiles"] for t in row
            if isinstance(t, dict) and t.get("kind") in ("COOP", "PASTURE")]
    empty_pens = sum(1 for t in pens if not t.get("animal"))
    livestock = len(pens) - empty_pens

    inventory = obs["market"]["inventory"]
    ranked = sorted(
        CROP_SPEC,
        key=lambda c: crop_value(c, inventory, pending_units(farm, private, c), days_left),
        reverse=True,
    )

    # Land is worth owning - removing it loses 36 of 40 games - but buying it
    # before the farm can afford to work it starves the seed budget for a third
    # of the season. Hold back a tuned number of days of running costs.
    bought = len(farm["unlocked_quadrants"]) - 1
    if bought < MAX_QUADRANTS - 1 and obs["day"] <= LAND_LAST_DAY:
        reserve = PARAMS["w_land_reserve"] * daily_burn(
            farm, private, prices, ranked, livestock)
        if farm["money"] >= 1000 * 2**bought + reserve:
            orders.append(["BUY_LAND"])
    _av = lambda a: animal_value(a, prices, days_left, inventory,
                                 livestock, private["shed"])
    best_animal = max(ANIMAL_SPEC, key=_av)
    waiting = sum(private["shed"].get(a, 0) for a in ANIMAL_SPEC)
    if (_av(best_animal) > 0 and empty_pens > waiting
            and farm["money"] > ANIMAL_SPEC[best_animal][0] + 500):
        orders.append(["BUY_ANIMAL", best_animal, 1])

    # Feed is bought, never grown - a tile costs actions, which are scarcer than
    # money. Keep a few days of buffer so a price spike never starves the herd.
    if livestock:
        want = livestock * 3 - private["shed"].get("WHEAT", 0)
        if want > 0 and farm["money"] > prices.get("WHEAT", 25) * want * 2:
            orders.append(["BUY_PRODUCT", "WHEAT", want])

    # Only buy seed for a crop still worth planting. crop_value goes negative
    # once a crop cannot mature before the season ends, and the planting loop
    # already refuses those - but the buying did not, so the agent kept
    # stocking seed it could never use. Roughly $1,000 of dead stock by turn
    # 720, when unsold inventory scores nothing.
    for crop in ranked[:2]:
        if crop_value(crop, inventory, pending_units(farm, private, crop), days_left) <= 0:
            continue
        # Restock all-or-nothing. Buying only what we can afford instead was
        # measured and is worse - 26W-54L over 80 games on two independent seed
        # ranges, about -$2,000 a game. The farm ends up permanently at zero
        # cash: dripping every spare dollar into seed leaves nothing banked,
        # and the crop mix it buys turns out identical either way. Holding out
        # for a full restock is what lets money accumulate at all.
        shortfall = SEED_BUFFER - private["seeds"].get(crop, 0)
        if shortfall > 0 and farm["money"] > SEED_COST[crop] * shortfall * 2:
            orders.append(["BUY_SEED", crop, shortfall])

    return orders[:10]


def agent(obs):
    farm = obs["farms"][obs["player"]]
    private = obs["private"]
    farmer, hands = _assign_actions(obs, farm, private)
    return {
        "farmer": farmer,
        "hands": hands,
        "market": _market_orders(obs, farm, private),
    }
