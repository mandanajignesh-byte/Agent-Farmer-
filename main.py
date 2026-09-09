"""Kaggriculture agent.

Every turn each worker (the farmer plus any hired hands) is given one action.
Candidate tiles are scored by a weighted sum of features and the lowest score
wins; workers claim tiles one at a time so no two walk to the same place.

All tuneable behaviour lives in PARAMS so the weights can be searched rather
than hand-picked - see tune.py. The defaults below reproduce v6 exactly.
"""
import market

TURNS_PER_DAY = 24
SEED_COST = {"WHEAT": 10, "CARROT": 20, "TOMATO": 50, "STRAWBERRY": 100, "MELON": 80}
MAX_YIELD_DAY = {"WHEAT": 4, "CARROT": 3, "TOMATO": 11, "STRAWBERRY": 16, "MELON": 10}
# Watering only adds yield from half-way to max yield onward. Before that it is
# pure survival, and survival tolerates every other day - so an early watering
# on a healthy plant buys nothing at all.
BONUS_START = {crop: (day + 1) // 2 for crop, day in MAX_YIELD_DAY.items()}

HANDS_PER_DAY = 8
SEED_BUFFER = HANDS_PER_DAY + 2
# Quadrants cost 1k, 2k, 4k. Buying land is a measured LOSS at current
# movement efficiency - 12 seeds, mean vs starter:
#            hands=6   hands=8  hands=10
#   1 quad   $11,048   $10,128    $7,445
#   2 quads   $9,555   $10,765    $9,237
# Labour is capped by fib cost and ~65% of every turn is already walking,
# so 25 tiles is past the optimum. Raise this only if movement improves.
MAX_QUADRANTS = 3
LAND_RESERVE = 500
LAND_LAST_DAY = 20

WATER, HARVEST, PLANT, DIG, PASS = "WATER", "HARVEST", "PLANT", "DIG", "PASS"
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

# (purchase price, product sold, days between yields)
ANIMAL_SPEC = {
    "GOOSE": (300, "EGG", 1),
    "COW": (400, "MILK", 2),
    "SHEEP": (500, "WOOL", 3),
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
    revenue = market.revenue_for(crop, inventory.get(crop, market.I0) + pending, yield_units)
    return (revenue - seed_cost) / days


def animal_value(animal, prices, days_left):
    """Profit per tile per day, with the purchase amortised over the season that
    is left. Late in the game that term explodes and the value goes negative, so
    the agent stops buying without needing a cutoff date - it stops exactly at
    the payback period."""
    cost, product, interval = ANIMAL_SPEC[animal]
    income = prices.get(product, 0) / interval
    feed = prices.get("WHEAT", 25)  # bought, not grown - tiles cost actions
    return income - feed - cost / max(days_left, 1)

# Lower score wins. The first six were hardcoded priority levels 1-6 scaled by
# the old PRIORITY_WEIGHT of 2; there was never a reason for them to be evenly
# spaced integers, so they are now searchable.
PARAMS = {
    "w_water_urgent": 5.994,
    "w_harvest_decay": 3.188,
    "w_harvest_ripe": 5.514,
    # Watering inside the bonus window earns a unit of yield; outside it, on a
    # plant in no danger, it earns nothing and only costs the walk.
    "w_water_bonus": 8.316,
    "w_water_idle": 30.0,
    "w_plant": 10.498,
    "w_dig": 11.086,
    "w_dist": 1.0,
    # PLANT only. Where an existing plant sits is already fixed, but choosing
    # where to plant fixes every future trip to that tile.
    "w_shed": -1.418,
    # An unfed animal is gone permanently and cost $300-500, so feeding
    # outranks everything a crop can ask for.
    "w_feed": 2.0,
    "w_harvest_animal": 3.5,
    "w_place": 3.0,
    "w_build": 10.0,
    "w_pickup": 6.0,
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


def _candidates(obs, farm, private):
    """Every actionable tile as (param_key, action, x, y, yield_units)."""
    step, hour = obs["step"], obs["hour"]
    seeds = private["seeds"]
    can_plant = TURNS_PER_DAY - hour >= 2

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

            if tile["consecutive_unwatered"] >= 1 and not tile["watered_today"]:
                found.append(("w_water_urgent", [WATER], x, y, 0, None))
            elif units > 0 and decaying:
                found.append(("w_harvest_decay", [HARVEST], x, y, units, None))
            elif units > 0 and ripe:
                found.append(("w_harvest_ripe", [HARVEST], x, y, units, None))
            elif not tile["watered_today"]:
                age = obs["day"] - tile["planted_day"]
                earning = BONUS_START[tile["crop"]] <= age <= MAX_YIELD_DAY[tile["crop"]]
                key = "w_water_bonus" if earning else "w_water_idle"
                found.append((key, [WATER], x, y, 0, None))

    # Planting more tiles in a turn than we hold seeds for makes every PLANT
    # that turn fail, not just the surplus ones - so each planned planting must
    # be backed by a seed we actually have.
    inventory = obs["market"]["inventory"]
    days_left = SEASON_DAYS - obs["day"]
    pending = {c: pending_units(farm, private, c) for c in CROP_SPEC}
    budget = {c: seeds.get(c, 0) for c in CROP_SPEC}

    prices = obs["market"]["prices"]
    best_animal = max(ANIMAL_SPEC, key=lambda a: animal_value(a, prices, days_left))
    animal_worth = animal_value(best_animal, prices, days_left)

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
    carried = sum(inv.get("WHEAT", 0) for inv in (private.get("inventories") or []))
    if unfed > carried and shed.get("WHEAT", 0) > 0:
        sx, sy = _shed_tiles(board)[0]
        found.append(("w_pickup", ["PICKUP", "WHEAT", max(unfed, 1)], sx, sy, 0, None))
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
    if tile["yield_units"] > 0:
        jobs.append(("w_harvest_animal", [HARVEST], x, y, tile["yield_units"], None))
    return jobs


def _shed_tiles(board_size):
    half = board_size // 2
    return [(cx, cy) for cx in (half - 1, half) for cy in (half - 1, half)]


def _score(candidate, wx, wy, board_size):
    key, _action, x, y, _units, _needs = candidate
    score = PARAMS[key] + PARAMS["w_dist"] * _distance(wx, wy, x, y)
    if key == "w_plant":
        score += PARAMS["w_shed"] * _shed_distance(x, y, board_size)
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

    bought = len(farm["unlocked_quadrants"]) - 1
    if bought < MAX_QUADRANTS - 1 and obs["day"] <= LAND_LAST_DAY:
        if farm["money"] >= 1000 * 2**bought + LAND_RESERVE:
            orders.append(["BUY_LAND"])

    prices = obs["market"]["prices"]
    days_left = SEASON_DAYS - obs["day"]
    pens = [t for row in farm["tiles"] for t in row
            if isinstance(t, dict) and t.get("kind") in ("COOP", "PASTURE")]
    empty_pens = sum(1 for t in pens if not t.get("animal"))
    livestock = len(pens) - empty_pens

    best_animal = max(ANIMAL_SPEC, key=lambda a: animal_value(a, prices, days_left))
    waiting = sum(private["shed"].get(a, 0) for a in ANIMAL_SPEC)
    if (animal_value(best_animal, prices, days_left) > 0 and empty_pens > waiting
            and farm["money"] > ANIMAL_SPEC[best_animal][0] + 500):
        orders.append(["BUY_ANIMAL", best_animal, 1])

    # Feed is bought, never grown - a tile costs actions, which are scarcer than
    # money. Keep a few days of buffer so a price spike never starves the herd.
    if livestock:
        want = livestock * 3 - private["shed"].get("WHEAT", 0)
        if want > 0 and farm["money"] > prices.get("WHEAT", 25) * want * 2:
            orders.append(["BUY_PRODUCT", "WHEAT", want])



    inventory = obs["market"]["inventory"]
    days_left = SEASON_DAYS - obs["day"]
    ranked = sorted(
        CROP_SPEC,
        key=lambda c: crop_value(c, inventory, pending_units(farm, private, c), days_left),
        reverse=True,
    )
    for crop in ranked[:2]:
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
