"""Kaggriculture agent.

Every turn each worker (the farmer plus any hired hands) is given one action.
Candidate tiles are scored by a weighted sum of features and the lowest score
wins; workers claim tiles one at a time so no two walk to the same place.

All tuneable behaviour lives in PARAMS so the weights can be searched rather
than hand-picked - see tune.py. The defaults below reproduce v6 exactly.
"""

TURNS_PER_DAY = 24
CROP = "WHEAT"
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
    seeds = private["seeds"].get(CROP, 0)
    can_plant = seeds > 0 and TURNS_PER_DAY - hour >= 2

    found, plantable = [], []
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
                found.append(("w_dig", DIG, x, y, 0))
                continue
            if tile.get("kind") != PLANT:
                continue

            lifespan = tile["max_lifespan_step"]
            decaying = lifespan != -1 and step >= lifespan
            ripe = obs["day"] - tile["planted_day"] >= MAX_YIELD_DAY[tile["crop"]]
            units = tile["yield_units"]

            if tile["consecutive_unwatered"] >= 1 and not tile["watered_today"]:
                found.append(("w_water_urgent", WATER, x, y, 0))
            elif units > 0 and decaying:
                found.append(("w_harvest_decay", HARVEST, x, y, units))
            elif units > 0 and ripe:
                found.append(("w_harvest_ripe", HARVEST, x, y, units))
            elif not tile["watered_today"]:
                age = obs["day"] - tile["planted_day"]
                earning = BONUS_START[tile["crop"]] <= age <= MAX_YIELD_DAY[tile["crop"]]
                key = "w_water_bonus" if earning else "w_water_idle"
                found.append((key, WATER, x, y, 0))

    # Planting more tiles than we hold seeds for makes every PLANT that turn
    # fail, not just the surplus ones.
    found += [("w_plant", PLANT, x, y, 0) for x, y in plantable[:seeds]]
    return found


def _score(candidate, wx, wy, board_size):
    key, _action, x, y, _units = candidate
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
    pool = _candidates(obs, farm, private)

    actions = [[PASS] for _ in workers]
    waiting = set(range(len(workers)))

    while waiting and pool:
        best = None
        for w in waiting:
            wx, wy = workers[w]
            for c, candidate in enumerate(pool):
                score = _score(candidate, wx, wy, board_size)
                if best is None or score < best[0]:
                    best = (score, w, c)

        _score_, w, c = best
        wx, wy = workers[w]
        _key, action, tx, ty, _units = pool.pop(c)
        waiting.discard(w)

        if (tx, ty) != (wx, wy):
            actions[w] = [_step_toward(wx, wy, tx, ty)]
        else:
            actions[w] = [action, CROP] if action == PLANT else [action]

    return actions[0], actions[1:]


def _market_orders(obs, farm, private):
    # Hands vanish at end of day and must be rehired each morning. fib(n) makes
    # the first few nearly free: 1, 1, 2, 3, 5, 8 ...
    orders = [["HIRE"]] * max(0, HANDS_PER_DAY - farm["hires_today"])

    bought = len(farm["unlocked_quadrants"]) - 1
    if bought < MAX_QUADRANTS - 1 and obs["day"] <= LAND_LAST_DAY:
        if farm["money"] >= 1000 * 2**bought + LAND_RESERVE:
            orders.append(["BUY_LAND"])

    orders += [["SELL", item, qty] for item, qty in private["shed"].items() if qty > 0]

    shortfall = SEED_BUFFER - private["seeds"].get(CROP, 0)
    if shortfall > 0 and farm["money"] > SEED_COST[CROP] * shortfall:
        orders.append(["BUY_SEED", CROP, shortfall])

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
