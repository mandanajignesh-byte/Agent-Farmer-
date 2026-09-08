"""Kaggriculture agent - v1.

Crops only, wheat only. Implements the priority list:
  1. water a plant that dies tonight
  2. harvest a decaying plant
  3. harvest a ready plant
  4. water any unwatered plant
  5. plant a seed on an empty tile
  6. PASS

Target tile is whichever minimises priority * PRIORITY_WEIGHT + distance,
so a cheap job underfoot can outrank a marginally better one across the
farm, while a plant about to die still outranks everything.
"""

TURNS_PER_DAY = 24
CROP = "WHEAT"
SEED_COST = {"WHEAT": 10, "CARROT": 20, "TOMATO": 50, "STRAWBERRY": 100, "MELON": 80}
MAX_YIELD_DAY = {"WHEAT": 4, "CARROT": 3, "TOMATO": 11, "STRAWBERRY": 16, "MELON": 10}
HANDS_PER_DAY = 6
SEED_BUFFER = HANDS_PER_DAY + 2
# Scales priority against walking distance when picking a target tile.
# Swept over 15 seeds: W=1 $7,216 / W=2 $7,315 / W=3 $7,132 / strict $7,104.
# The whole spread sits inside one standard error, so this is not a
# measured win - it only rules out pathological cross-farm thrashing.
PRIORITY_WEIGHT = 2

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
    seeds = private["seeds"].get(CROP, 0)
    can_plant = seeds > 0 and TURNS_PER_DAY - hour >= 2
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

            lifespan = tile["max_lifespan_step"]
            decaying = lifespan != -1 and step >= lifespan
            ripe = obs["day"] - tile["planted_day"] >= MAX_YIELD_DAY[tile["crop"]]
            has_yield = tile["yield_units"] > 0

            if tile["consecutive_unwatered"] >= 1 and not tile["watered_today"]:
                buckets[1].append((x, y))
            elif has_yield and decaying:
                buckets[2].append((x, y))
            elif has_yield and ripe:
                buckets[3].append((x, y))
            elif not tile["watered_today"]:
                buckets[4].append((x, y))

    # Planting more tiles than we hold seeds for makes every PLANT that turn
    # fail, not just the surplus ones.
    buckets[5] = buckets[5][:seeds]
    return buckets


def _assign_actions(obs, farm, private):
    """One action per worker, farmer first. Claimed tiles leave the pool so no
    two workers walk to the same tile."""
    workers = [tuple(farm["farmer"])] + [tuple(h) for h in farm["hands"]]
    pool = [
        (priority, x, y)
        for priority, tiles in _bucket_tiles(obs, farm, private).items()
        for x, y in tiles
    ]

    actions = []
    for wx, wy in workers:
        best = None
        for i, (priority, tx, ty) in enumerate(pool):
            score = priority * PRIORITY_WEIGHT + _distance(wx, wy, tx, ty)
            if best is None or score < best[0]:
                best = (score, i, priority, tx, ty)

        if best is None:
            actions.append(["PASS"])
            continue

        _, index, priority, tx, ty = best
        pool.pop(index)
        if (tx, ty) != (wx, wy):
            actions.append([_step_toward(wx, wy, tx, ty)])
        else:
            action = ACTION_FOR_PRIORITY[priority]
            actions.append([action, CROP] if action == PLANT else [action])

    return actions[0], actions[1:]


def _market_orders(obs, farm, private):
    # Hands vanish at end of day and must be rehired each morning. fib(n) makes
    # the first few nearly free: 1, 1, 2, 3, 5, 8 ...
    orders = [["HIRE"]] * max(0, HANDS_PER_DAY - farm["hires_today"])

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
