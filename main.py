"""Kaggriculture agent - v1.

Crops only, wheat only. Implements the priority list:
  1. water a plant that dies tonight
  2. harvest a decaying plant
  3. harvest a ready plant
  4. water any unwatered plant
  5. plant a seed on an empty tile
  6. PASS

Target selection is strict priority first, nearest tile as tie-break.
"""

TURNS_PER_DAY = 24
CROP = "WHEAT"
SEED_COST = {"WHEAT": 10, "CARROT": 20, "TOMATO": 50, "STRAWBERRY": 100, "MELON": 80}
SEED_BUFFER = 3

WATER, HARVEST, PLANT = "WATER", "HARVEST", "PLANT"
ACTION_FOR_PRIORITY = {1: WATER, 2: HARVEST, 3: HARVEST, 4: WATER, 5: PLANT}


def _distance(ax, ay, bx, by):
    return abs(ax - bx) + abs(ay - by)


def _step_toward(fx, fy, tx, ty):
    if tx > fx:
        return "EAST"
    if tx < fx:
        return "WEST"
    if ty > fy:
        return "SOUTH"
    return "NORTH"


def _bucket_tiles(obs, farm, private):
    """Group actionable tiles by priority. Lower key = more urgent."""
    step, hour = obs["step"], obs["hour"]
    can_plant = private["seeds"].get(CROP, 0) > 0 and TURNS_PER_DAY - hour >= 2
    buckets = {1: [], 2: [], 3: [], 4: [], 5: []}

    for y, row in enumerate(farm["tiles"]):
        for x, tile in enumerate(row):
            if tile == "LOCKED":
                continue
            if tile is None:
                if can_plant:
                    buckets[5].append((x, y))
                continue
            if not isinstance(tile, dict) or tile.get("kind") != PLANT:
                continue

            if tile["consecutive_unwatered"] >= 1 and not tile["watered_today"]:
                buckets[1].append((x, y))
            elif tile["yield_units"] > 0:
                lifespan = tile["max_lifespan_step"]
                decaying = lifespan != -1 and step >= lifespan
                buckets[2 if decaying else 3].append((x, y))
            elif not tile["watered_today"]:
                buckets[4].append((x, y))

    return buckets


def _farmer_action(obs, farm, private):
    fx, fy = farm["farmer"]
    buckets = _bucket_tiles(obs, farm, private)

    for priority in sorted(buckets):
        candidates = buckets[priority]
        if not candidates:
            continue
        tx, ty = min(candidates, key=lambda p: _distance(fx, fy, p[0], p[1]))
        if (tx, ty) != (fx, fy):
            return [_step_toward(fx, fy, tx, ty)]
        action = ACTION_FOR_PRIORITY[priority]
        return [action, CROP] if action == PLANT else [action]

    return ["PASS"]


def _market_orders(obs, farm, private):
    orders = [["SELL", item, qty] for item, qty in private["shed"].items() if qty > 0]

    shortfall = SEED_BUFFER - private["seeds"].get(CROP, 0)
    if shortfall > 0 and farm["money"] > SEED_COST[CROP] * shortfall:
        orders.append(["BUY_SEED", CROP, shortfall])

    return orders[:10]


def agent(obs):
    farm = obs["farms"][obs["player"]]
    private = obs["private"]
    return {
        "farmer": _farmer_action(obs, farm, private),
        "hands": [],
        "market": _market_orders(obs, farm, private),
    }
