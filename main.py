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


# Which shops want which product. Shops tick every 4 turns (6 times a day) and
# a single-product shop consumes double; the town centre itself also takes 1
# unit/day of everything except fertilizer. Verified exact against 28 days of
# untouched inventory drift in test_model.py - not a guess.
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
    """Units/day the town removes from the market on its own, whether or not
    we sell anything. This never reverses over a season, so a product's price
    drifts in one direction the whole time it stays unlocked - milk, wool and
    egg get scarcer (pricier) every day; fertilizer, drained by nobody but
    other players, does not drift at all."""
    rate = 0 if product == "FERTILIZER" else 1
    for shop in shops:
        demands = SHOPS[shop]
        if product in demands:
            rate += 6 * (2 if len(demands) == 1 else 1)
    return rate


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
FERTILIZE = "FERTILIZE"
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


def crop_value(crop, inventory, pending, days_left=None, shops=None, fert_pending=0,
                planted_tiles=0):
    """Profit per tile per day for ONE MORE tile of this crop, pricing its yield
    unit by unit as it pushes the price down.

    This is what makes tile counts self-balancing: plant more melon and melon's
    marginal value falls until carrot overtakes it, then wheat overtakes carrot.
    No target count has to be chosen, and the mix re-balances on its own if an
    opponent floods a market.

    Wheat and carrot are two of the town's most heavily-drained products (four
    shops want wheat, one wants carrot) - the same drift animal_value prices
    for milk, wool and egg applies here too, at the same tunable trust level
    (w_town_drift, starts at 0). Melon and tomato are not in any shop's list
    and drift by exactly zero either way.

    Whether fertilizing this crop is worth it is decided once here and reused
    - not recomputed independently by _candidates - so a tile that fertilizing
    would help is valued at its real, higher worth (the extra headroom units,
    priced the same marginal way) at the cost of the extra action fertilizing
    takes, rather than being ranked as if that action and that yield did not
    exist. Melon, tomato and strawberry have zero headroom, so this changes
    nothing for them regardless of price."""
    yield_units, seed_cost, days = CROP_SPEC[crop]
    if days_left is not None and days_left < days:
        # Cannot mature before the season ends, and we only harvest at
        # age >= MAX_YIELD_DAY, so it would never be picked even partially.
        # The seed is simply spent.
        return -seed_cost
    # Tried projecting drift over the crop's own cycle (days) instead of
    # days_left (the whole remaining season), the same fix as animal_value's
    # cost amortization and for the same reason - the price a tile sells at
    # is wherever the market sits when IT matures, not whenever the season
    # ends. Reverted: it measured only $0-2.60 of difference in the one
    # snapshot checked (the town's base drain rate is modest early on), but
    # broke a real safety invariant in test_behaviour.py - 3 of 3 animal
    # escapes had feed available afterward, where 0 did before. Likely
    # cause: WHEAT is both a sellable crop and animal feed, and even a
    # small drop in crop_value(WHEAT) shifts how much gets planted, which
    # shifts how much ends up in the shed to feed animals with - a coupling
    # this fix did not account for. Not worth the size of the measured gain.
    horizon = max(days_left or 0, 0) / 2
    drain = town_drain_per_day(crop, shops or [])
    effective_inventory = (inventory.get(crop, MARKET_I0) + pending
                            - PARAMS["w_town_drift"] * drain * horizon)
    revenue = revenue_for(crop, effective_inventory, yield_units)
    actions = 1 + 1 / days  # a watering a day, plus the harvest at the end
    if fertilize_value(crop, inventory, pending, fert_pending) > 0:
        headroom = TRUE_MAX_YIELD[crop] - yield_units
        revenue += revenue_for(crop, effective_inventory + yield_units, headroom)
        actions += 1 / days  # the fertilize action, amortised like the harvest
    # crop_value's own version of animal_value's service_load/survival - see
    # PARAMS["w_water_risk"] for why this exists (crop planting had no
    # capacity discount at all before it) and what real regression it fixes.
    # One more tile competes with every other planted tile and the whole
    # herd for the same hands; tile_load is this tile's share of that
    # competition, in daily actions, against what the farm can actually
    # staff (w_workforce_capacity, not the animal side's separate w_pen_ahead
    # target - crops and animals are priced against the same hands, but not
    # yet against each other's CURRENT load, only their own kind's - a known
    # simplification, not a full joint labour market).
    tile_load = (planted_tiles + 1) * actions / PARAMS["w_workforce_capacity"]
    survival = 1 / (1 + PARAMS["w_water_risk"] * max(0.0, tile_load - 1))
    revenue *= survival
    return (revenue - seed_cost) / days - PARAMS["w_action_cost"] * actions


# The environment's own per-crop yield cap (CROPS[c]["max_yield"], verified
# against the env source, not a guess) - higher than what CROP_SPEC's own
# "achievable" figure reaches without fertilizer for wheat and carrot.
# Melon, tomato and strawberry already reach this cap unfertilized, so
# fertilize_value naturally prices them at zero headroom without needing to
# special-case them.
TRUE_MAX_YIELD = {"WHEAT": 6, "CARROT": 4, "TOMATO": 4, "STRAWBERRY": 4, "MELON": 6}


def fertilize_value(crop, inventory, pending, fert_pending):
    """Profit from spending one fertilizer on this crop instead of selling
    it, priced the same dynamic way as everything else - not the static
    "$50 vs $100, selling wins" comparison this used to be judged by.

    That comparison was true at a single snapshot price, but fertilizer has
    no town-shop demand at all (verified in test_model.py) - the only thing
    that ever drains its market is us or an opponent selling less of it. A
    farm running a serious herd produces far more fertilizer than a crop
    plot can absorb by fertilizing, and every unit sold pushes fertilizer's
    own price down while a well-protected crop like wheat barely sags on
    glut - so at high volume the marginal fertilizer sale can easily be
    worth less than the marginal wheat gain, which is exactly backwards from
    the snapshot comparison. headroom is the extra yield fertilizer can
    still add before the crop's own true cap, verified against the
    environment; it is already zero for melon, tomato and strawberry, which
    reach that cap without any fertilizer at all."""
    yield_units, _, _ = CROP_SPEC[crop]
    headroom = TRUE_MAX_YIELD[crop] - yield_units
    if headroom <= 0:
        return -1
    extra_revenue = revenue_for(crop, inventory.get(crop, MARKET_I0) + pending, headroom)
    fert_given_up = price_at("FERTILIZER", inventory.get("FERTILIZER", MARKET_I0) + fert_pending)
    return extra_revenue - fert_given_up - PARAMS["w_action_cost"]


def animal_value(animal, prices, days_left, inventory=None, herd=0, shed=None,
                  shops=None):
    """Profit per tile per day, with the purchase amortised over the season that
    is left. Late in the game that term explodes and the value goes negative, so
    the agent stops buying without needing a cutoff date - it stops exactly at
    the payback period.

    Income is priced marginally, the same way crop_value does it: a herd of a
    dozen animals produces hundreds of units over a season and drives its own
    prices down. Fertilizer especially - no town shop consumes it, so the only
    thing draining that market is other players buying.

    Two more forces move these prices on their own, independent of anything we
    do, and both are folded in below rather than left for tuning to guess at:

    - The town's shops eat product every day, and that drain never reverses
      (town_drain_per_day is verified exact in test_model.py). A product's
      price drifts up for the rest of the season purely from that, so it is
      priced here as an average over the animal's remaining life rather than
      a single snapshot - milk, wool and egg are worth more to be holding the
      longer the season has left to run. Wheat drains the same way (five of
      eight shops want it), so feed cost is projected forward too.
    - Feeding needs a worker free to carry wheat and stand at the pen every
      day. A herd bigger than the workforce can service starts missing
      feedings, and two consecutive misses loses the animal for good (rule
      verified in test_model.py). That risk is invisible in today's price, so
      it is modelled from the ratio of daily pen chores to hands available -
      below capacity it costs nothing; above it, w_escape_risk (starts at 0,
      so nothing changes until tuning turns it on) decides how much it bites.
    """
    cost, product, interval, max_held, first_yield = ANIMAL_SPEC[animal]
    inventory = inventory or {}
    shed = shed or {}
    shops = shops or []

    # Town drain is linear and never reverses, so its mean over the horizon we
    # are pricing for is just the midpoint. Applied as a further shortfall on
    # top of whatever we and the existing herd already add to the inventory.
    horizon = max(days_left, 0) / 2

    def drifted(item, extra_pending=0):
        drain = town_drain_per_day(item, shops)
        # The town is not the only other seller: an opponent adding supply to
        # the same market cancels this drift, and how much it cancels is not
        # something the rules state - only measurement can say. w_town_drift
        # (starts at 0) is how much of the town's uncontested drain we trust
        # will still be there once a real opponent is selling too.
        return (inventory.get(item, MARKET_I0) + extra_pending
                - PARAMS["w_town_drift"] * drain * horizon)

    # what our existing herd will still add before this animal's output lands.
    #
    # Was max(days_left, 0) - the WHOLE remaining season, projected onto TODAY's
    # price as if the herd's entire future output already crowds the market
    # before a single unit of it exists. Verified against a real loss (Sean
    # Peppers, replays_latest/episode-109321852): their WOOL price does crash
    # to the $1 floor - but only by day 21-27, after weeks of actual selling,
    # not on day 0. With this bug, a 3rd sheep already priced that entire
    # crash in immediately (produce fell from $230 to $0.82 per tile-day at
    # herd=3), which is what really capped the herd at w_pen_ahead's ~3, not
    # any real lack of profit. crop_value bounds its own equivalent projection
    # to the item's own fixed cycle length, never the season - the same fix
    # here: how much the herd will add before ITS OWN next production event
    # (interval days out), not everything it will ever produce.
    made_per_animal = min(max(days_left, 0), interval)
    prod_pending = shed.get(product, 0) + herd * made_per_animal * (1 + interval) / interval
    fert_pending = shed.get("FERTILIZER", 0) + herd * made_per_animal

    # CARE banks one unit a day and pays the whole bank out on the next
    # production, so an animal cared for every day yields 1 + interval per
    # interval instead of 1 - triple for a cow, four times for a sheep. Capped
    # by max_held, which the bank cannot exceed.
    per_event = min(1 + interval, max_held)
    units = revenue_for(product, drifted(product, prod_pending), per_event)
    produce = units / interval
    # Every surviving animal yields one fertilizer a day, free, fed or not -
    # and fertilizer's base price of $100 makes that stream comparable to the
    # milk. Valuing an animal on its product alone undercounts it by about
    # half, which is why the herd never grew.
    fertilizer = revenue_for("FERTILIZER", drifted("FERTILIZER", fert_pending), 1)
    feed = price_at("WHEAT", drifted("WHEAT"))  # bought, not grown - tiles cost actions
    actions = 3 + 1 / interval  # feed, care and collect daily; harvest each interval

    # An animal produces nothing for its first `first_yield` days - 4 for a
    # goose, 6 for a sheep, 8 for a cow - but eats and takes actions from the
    # day it is placed. Ignoring that overvalued every late purchase by its
    # whole lead time: a cow bought on day 24 never yields once, and the old
    # formula happily recommended it. Fertilizer is exempt, since a surviving
    # animal drops one a day from the start whether it is producing or not.
    productive = max(0, days_left - first_yield)
    produce *= productive / max(days_left, 1)

    # service_load is the daily pen chores the herd this animal would join
    # needs, divided by hands the farm can actually staff. Was HANDS_PER_DAY
    # (8) - the BASE daily hire target before any backlog response, not what
    # real games actually reach. Verified against real Kaggle replays and
    # league_public/master_v3.py: hand counts of 12-17 by mid-game are
    # ordinary once herds/plots justify the fib-scaled hiring cost. Pricing
    # capacity at 8 forever made this discount bite at herd sizes real,
    # working farms handle routinely, which is what actually capped the
    # herd around 3-6 - not any real shortage of hands. w_workforce_capacity
    # is the same real number, shared with crop_value's own version of this
    # discount (see PARAMS["w_water_risk"]). At or under 1 there are enough
    # hands and the risk is zero by construction; over 1, w_escape_risk sets
    # how fast the survival odds fall off.
    service_load = (herd + 1) * actions / PARAMS["w_workforce_capacity"]
    survival = 1 / (1 + PARAMS["w_escape_risk"] * max(0.0, service_load - 1))
    produce *= survival
    fertilizer *= survival

    # crop_value spreads its (much smaller) seed cost over the crop's own
    # fixed cycle length, unaffected by the calendar except a hard cutoff
    # when there is no longer time to mature at all - so a wheat tile is
    # worth the same whether it is bought on day 1 or day 20. This instead
    # spread the (much larger) purchase price over days_left, the whole
    # remaining SEASON, which shrinks every single day regardless of the
    # animal's own economics - the same sheep looked steadily worse for no
    # reason but the calendar moving, dragging animal_value below
    # crop_value for nearly the entire game (measured in a real loss:
    # animal_value never won the BUILD_PASTURE comparison after day 12,
    # settling at 5 pens for the rest of a 30-day game against an opponent
    # who reached 13). w_animal_payback_days is a fixed floor on the
    # divisor, the same role crop_value's fixed cycle length plays - it
    # leaves every calculation with a full payback window ahead of it
    # unchanged, and only softens the penalty once days_left actually runs
    # low. 1 reproduces the old always-shrinking behaviour exactly (the
    # existing max(days_left, 1) zero-guard, unchanged).
    payback_divisor = max(days_left, PARAMS["w_animal_payback_days"])
    return (produce + fertilizer - feed - cost / payback_divisor
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
    hold back, which tuning decides.

    Missing until now: the cost of the next animal. A land purchase competes
    for the exact same cash as growing the herd, and real opponents spend
    theirs on animals and hands instead of a second quadrant - verified
    against 19 real games (day-0 money $3,000 -> day-1 $246, seed-starved
    while the opponent, on identical land, had already planted 3x as many
    tiles). w_land_reserve has sat at 0 through every tuning round because
    daily_burn never gave it a reason to move - it priced feeding the herd
    that exists, never growing it. The cheapest animal's cost is the
    smallest real stake in that competition, not an assumption about which
    animal or how many."""
    hires, a, b = 0, 1, 1
    for _ in range(HANDS_PER_DAY):
        hires += a
        a, b = b, a + b
    seed = sum(SEED_COST[c] * SEED_BUFFER for c in ranked[:2])
    feed = livestock * 3 * prices.get("WHEAT", 25)
    animal = min(cost for cost, *_ in ANIMAL_SPEC.values())
    return hires + seed + feed + animal


def _fib_hire_cost(n):
    """Cost of the (n+1)-th hire today - the env's own fib(n) schedule,
    1, 1, 2, 3, 5, 8, ... Exponential within a day, and resets to 1 again
    tomorrow, because hands vanish overnight - a hired hand is a same-day
    rental, not a season asset the way land or an animal is."""
    a, b = 1, 1
    for _ in range(n):
        a, b = b, a + b
    return a


def hire_value(n, best_task_value, turns_left):
    """UNUSED - kept as a documented false start, not wired into
    _market_orders. See the comment there for what was measured and why
    this formula's shape is wrong: it scaled best_task_value (a $/tile/day
    rate) by a fraction of today's remaining turns, which prices an extra
    hand as capable of only a fraction of one tile's work, when it can
    actually complete several separate full-value actions in that time.
    Reusable once that's fixed - the cost side (the fib schedule) is real
    and unaffected."""
    cost = _fib_hire_cost(n)
    value = best_task_value * (turns_left / TURNS_PER_DAY)
    return value - cost


# Lower score wins. The first six were hardcoded priority levels 1-6 scaled by
# the old PRIORITY_WEIGHT of 2; there was never a reason for them to be evenly
# spaced integers, so they are now searchable.
PARAMS = {
    # Hill-climbed against fitness.py's league soft-min score (SEARCH_LEAGUE),
    # then validated on 12 seeds the search never saw against the FULL league:
    # +0.82 overall, +0.90 mean, 79.2% win rate against v22 (our toughest
    # opponent, up from 75% before this round), 0 crashes over 120 games. See
    # tune_hillclimb.log / validate_tuned.log.
    "w_water_urgent": 8.418,
    "w_harvest_decay": 4.797,
    "w_harvest_ripe": 5.17,
    # Watering inside the bonus window earns a unit of yield; outside it, on a
    # plant in no danger, it earns nothing and only costs the walk.
    "w_water_bonus": 7.529,
    "w_water_idle": 30.0,
    "w_plant": 10.623,
    "w_dig": 12.865,
    "w_dist": 1.0,
    "w_dist_sq": 0.0,
    # PLANT only. Where an existing plant sits is already fixed, but choosing
    # where to plant fixes every future trip to that tile.
    #
    # The sign looks backwards next to w_pen_shed (same _shed_distance
    # formula, opposite sign) - tested flipping it directly: measured mean
    # plant-to-shed distance over 3 seeds barely moved (5.28 vs 5.07/4.94),
    # confirming why. Every empty tile eventually gets a PLANT candidate
    # regardless of distance (nothing filters candidate generation by shed
    # proximity, only the assignment score once a tile is already a
    # candidate), so this term only nudges transient turn-to-turn ordering,
    # not which tiles end up planted - not the mechanism behind the real gap
    # found the same day (see below), which is planting throughput, not
    # placement. Left as-is; revisit only with real supporting evidence,
    # not the sign argument alone.
    "w_shed": -0.499,
    # An unfed animal is gone permanently and cost $300-500, so feeding
    # outranks everything a crop can ask for.
    "w_feed": 0.562,
    # A second, strictly cheaper priority for an animal already one missed
    # feeding from escaping (consecutive_unfed >= 1) - see _pen_jobs for the
    # real escape this fixes: every routine w_feed job shares one flat
    # priority regardless of distance, so a pen far from the shed cluster
    # can lose the distance competition to every closer pen, every day, and
    # starve with wheat sitting untouched. Below every other candidate,
    # including w_feed itself, so the one day this animal cannot afford to
    # lose is never the day distance decides it for a farther pen.
    "w_feed_urgent": -10.0,
    "w_harvest_animal": 3.5,
    "w_collect": 3.5,
    # A brand new candidate, never offered before this round - not gated to
    # zero-effect like the other new terms, because this is a priority rank
    # (lower wins), not a multiplier with a meaningful "off" value. Started
    # near w_care/w_collect's level as a reasonable guess for tuning to
    # refine, not a considered answer.
    "w_fertilize": 4.0,
    "w_care": 3.0,
    "w_drop": 4.943,
    "w_place": -1.419,
    "w_build": 8.313,
    "w_pickup": 9.138,
    # Split from w_pickup - see _candidates for why. Started at a fraction of
    # w_pickup as a guess, not a considered answer; sweep.py should settle it.
    # Split from w_pickup - see _candidates for why. Tested by hand across
    # several values (1-9.138) head-to-head against the champion: none beat
    # the no-op baseline, and 6.0 measured clearly worse (-$2,886, 31.2% win
    # rate over 16 seeds). The diagnosis is real (bought animals sat unplaced
    # 10-17 days in real games) but a single-weight sweep can't see this
    # weight's interaction with the rest of the ecosystem - lowering it also
    # lets it beat genuinely urgent jobs like w_water_urgent (8.418), trading
    # a dying crop for an idle animal. Left at w_pickup's own value (a no-op)
    # until a proper multi-weight search can place it correctly.
    "w_pickup_animal": 9.138,
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
    "w_action_cost": 5.424,
    # How many empty pens to keep standing ahead of the animals waiting to
    # fill them. Building is free, so this only trades a tile against the
    # crop it could have grown instead - which the animal-vs-crop comparison
    # already prices per tile. Starts at the old hardcoded 2.
    "w_pen_ahead": 2.861,
    # Days of running costs to keep in the bank before buying an animal - the
    # same argument as w_land_reserve, on the same daily_burn. Starts at zero.
    "w_animal_reserve": -0.455,
    # How sharply feeding risk should bite once the herd needs more daily pen
    # chores than the farm's target headcount can supply. Zero means the herd
    # is assumed perfectly fed no matter how large it gets - the search turned
    # this on, so it now does bite.
    "w_escape_risk": 0.449,
    # Hands the farm can actually staff, for pricing how thin a bigger herd or
    # crop plot spreads the workforce - see animal_value's service_load and
    # crop_value's own equivalent. Was HANDS_PER_DAY (8) hardcoded into
    # service_load directly: real games (ours and opponents') routinely hire
    # past that once backlogs justify it - 12-17 hands by mid-game, verified
    # against real Kaggle replays and league_public/master_v3.py - so pricing
    # capacity at 8 forever capped the profitable herd size around 3-6
    # regardless of how positive the raw economics were. A tunable PARAM
    # instead of a second hardcoded constant, since the right number is an
    # empirical question sweep.py can refine, not one this comment can settle.
    "w_workforce_capacity": 15.0,
    # crop_value's own version of w_escape_risk - how sharply a crop tile's
    # value should fall once the board holds more planted tiles than the
    # workforce can reliably water. Crop planting had NO such discount at
    # all until now: crop_value priced every tile in isolation, so nothing
    # ever stopped planting once land was unlocked. Traced directly to a
    # live regression: a real loss (replays_v11/episode-109749026, vs Matt
    # Dowis) held 44-67 planted tiles against 12 hands - visibly more plot
    # than 3 quadrants' worth of hands can keep watered - while weed tiles
    # (a tile that missed watering two days running, verified in
    # test_model.py) climbed from 0 before day 15 to 10-13 a game from day 18
    # on. Started at w_escape_risk's own value as the nearest real analog
    # (same shape of risk, same headcount competing for it) and checked
    # directly against real games below, not left at the old always-plant
    # 0 - a genuine fix needs this actually turned on to do anything.
    "w_water_risk": 0.449,
    # How much of the town's daily drain we still trust once a real opponent
    # is adding supply to the same market and cancelling part of it. The
    # search landed almost exactly halfway between "ignore the town" (0) and
    # "we are the only other seller" (1) - which is what partial cancellation
    # from an actively-selling opponent should look like.
    "w_town_drift": 0.488,
    # Hard cap on extra hands hired beyond HANDS_PER_DAY in response to
    # today's urgent-job backlog (0 = never respond). Not a ratio - fib-
    # scaled hire cost means matching a large backlog 1:1 is ruinously
    # expensive (measured: -$18k to -$27k with an uncapped ratio). Started
    # at a small guess; sweep.py should settle it.
    "w_hire_backlog": 3.0,
    # Share of currently-unlocked land (plants + pens, out of everything
    # unlocked) required before the next quadrant is worth buying. 0
    # reproduces the old cash-only gate exactly (a no-op starting point);
    # see _market_orders for the real-game evidence this responds to.
    "w_land_utilization": 0.7,
    # Hard cap on extra hands hired for the animal-pickup backlog, separate
    # from w_hire_backlog's own cap - see _market_orders for why sharing
    # one cap between the two measured as a complete no-op. 0 = never
    # respond, matching pre-fix behaviour.
    "w_hire_pickup_backlog": 1.0,
    # Floor on animal_value's cost-amortization divisor - see animal_value
    # for why the plain days_left divisor made a new animal look steadily
    # worse for no reason but the calendar moving. Measured a complete
    # no-op against the current league at every tested value (1 through
    # 20) - the gate it was meant to help clear (animal_worth > crop_best)
    # stayed unmet regardless, because crop_value's own margin was simply
    # higher in every traced case, not because of this term. Harmless
    # everywhere tested, and more internally consistent with crop_value's
    # own fixed-cycle amortization, so kept - but this is a reasoned guess
    # at roughly 1-2 production cycles (goose ~5-8 days, cow ~10-14, sheep
    # ~9-15), not a considered answer; the real remaining question is why
    # animal_value's margin trails crop_value's before either cost term is
    # even subtracted.
    "w_animal_payback_days": 10.0,
}


def _distance(ax, ay, bx, by):
    return abs(ax - bx) + abs(ay - by)


def _land_utilization(farm):
    """Share of already-unlocked land actually put to work (a plant or a
    pen), out of everything that could be. Locked tiles do not count either
    way - they are not ours to use yet. A weed counts as unlocked-but-idle,
    same as bare dirt: it is not producing until cleared and replanted."""
    usable = occupied = 0
    for row in farm["tiles"]:
        for cell in row:
            if cell == "LOCKED":
                continue
            usable += 1
            if isinstance(cell, dict) and cell.get("kind") in ("PLANT", "PASTURE", "COOP"):
                occupied += 1
    return occupied / usable if usable else 0.0


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
    board_size = len(farm["tiles"])

    inventory = obs["market"]["inventory"]
    days_left = SEASON_DAYS - obs["day"]
    pending = {c: pending_units(farm, private, c) for c in CROP_SPEC}
    fert_pending = private["shed"].get("FERTILIZER", 0)

    found, plantable = [], []
    empty_pens = animals_placed = unfed = planted_tiles = 0
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
            planted_tiles += 1

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

            # A separate action from watering, offered alongside whatever
            # else this tile already qualified for - only while there is
            # still a bonus-watering day left to spend it on, and only if
            # the tile is not already covered by an earlier application
            # (fertilized_until_day spans 3 days, so one application can
            # cover several watering turns).
            if (water_earns and tile.get("fertilized_until_day", -1) < obs["day"]
                    and fertilize_value(crop, inventory, pending[crop], fert_pending) > 0):
                found.append(("w_fertilize", [FERTILIZE], x, y, 0, "FERTILIZER"))

    # Planting more tiles in a turn than we hold seeds for makes every PLANT
    # that turn fail, not just the surplus ones - so each planned planting must
    # be backed by a seed we actually have.
    budget = {c: seeds.get(c, 0) for c in CROP_SPEC}

    prices = obs["market"]["prices"]
    shops = obs["town"]["unlocked_shops"]
    _av = lambda a: animal_value(a, prices, days_left, inventory,
                                 animals_placed, private["shed"], shops)
    best_animal = max(ANIMAL_SPEC, key=_av)
    animal_worth = _av(best_animal)

    # Tried sorting plantable nearest-shed-first here, on the theory that a
    # pen built far from the shed cluster (a real traced escape: (8,2) lost
    # the daily distance competition to every closer pen and starved with
    # wheat untouched) only happens because board-scan order plants and pens
    # wherever the row/column loop reaches next, with no regard for repeat-
    # visit cost. Measured, not assumed: reverted, because it made things
    # much worse, not better - 79 of 60 seasons broke the starvation
    # invariant (up from 2), with peak herd rising to 17 (from 12-14). The
    # mechanism isn't understood - clustering every pen right at the shed
    # should shorten feeding walks, not lengthen the escape count - which
    # itself is a reason not to ship it: a fix whose own effect contradicts
    # its own reasoning needs to be understood before it is trusted, not
    # just measured once and kept because the number moved. w_feed_urgent
    # above still catches the (8,2)-style case after the fact and measured
    # clean (0/60, then 2/60 at a bigger sample) - real pen placement is
    # still an open question, not one a same-turn sort answered.

    # A pen only pays once an animal stands in it, so build them a tuned number
    # ahead of demand rather than covering the farm in empty structures.
    #
    # Counted on a separate variable from empty_pens, not empty_pens itself.
    # A planned-but-not-yet-built pen is not a place to put an animal - it is
    # still just bare ground with a BUILD candidate on it, since building is a
    # worker action like any other and has not happened yet this turn. Found
    # by tracing a fresh cash-drain bug: with w_pickup_animal's own check
    # below reading this same empty_pens, an animal bought this very turn
    # (before any real pen exists) looked pickupable immediately, sending a
    # worker to carry it off with nowhere to place it - which emptied the
    # shed, which made a from-scratch buy look needed again next turn, which
    # bought another - repeatedly, well before any pen was actually built.
    pens_ahead = empty_pens
    # Mirrors pens_ahead: counted separately from planted_tiles itself, since
    # a tile this same loop just decided to plant is not yet a real worker
    # commitment either - it is a PLANT candidate a worker still has to walk
    # to and execute, exactly the pens_ahead argument above.
    tiles_ahead = planted_tiles
    for x, y in plantable:
        if animal_worth > 0 and pens_ahead < PARAMS["w_pen_ahead"]:
            crop_best = max((crop_value(c, inventory, pending[c], days_left, shops,
                                        fert_pending, tiles_ahead)
                             for c in CROP_SPEC), default=0)
            if animal_worth > crop_best:
                pens_ahead += 1
                found.append(("w_build", ["BUILD_" + STRUCTURE_FOR[best_animal]],
                              x, y, 0, None))
                continue
        affordable = [c for c in CROP_SPEC if budget[c] > 0]
        best = max(
            affordable,
            key=lambda c: crop_value(c, inventory, pending[c], days_left, shops,
                                     fert_pending, tiles_ahead),
            default=None,
        )
        # Nothing left that can mature in time - stop planting entirely and
        # leave the workers free to harvest and sell.
        if (best is None
                or crop_value(best, inventory, pending[best], days_left, shops,
                              fert_pending, tiles_ahead) <= 0):
            break
        budget[best] -= 1
        # this tile's own output crowds the next one
        pending[best] += CROP_SPEC[best][0]
        tiles_ahead += 1
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
    # v21 offered a single DROP, at one shed tile, on the last day only. That is
    # one job for nine workers, so most of them still finished the season
    # holding produce: 28 units worth about $3,000 were measured still in hand
    # on turn 718. Offer a trip per shed-access tile, and start a day earlier so
    # what gets dropped still has turns left to be sold in.
    if obs["day"] >= SEASON_DAYS - 2:
        held = set()
        for inv in (private.get("inventories") or []):
            held.update(k for k, v in inv.items() if v > 0 and k in MARKET_PARAMS)
        for item in held:
            for sx, sy in _shed_tiles(board):
                found.append(("w_drop", [DROP], sx, sy, 0, item))
    # Count the workers who can feed, not the wheat they are holding. A FEED job
    # requires wheat in hand, so what limits feeding is how many workers carry
    # any - not the total units. v22 made the pickups parallel but left this
    # test counting units, and the farm settled at about two carriers holding
    # nine wheat between them against four hungry animals: unfed - units came
    # out negative, so no further pickup was ever offered, and the other seven
    # workers could not help. That switched the trigger off on 89% of the turns
    # where an animal was actually unfed, and animals still starved in sight of
    # a full shed.
    carriers = sum(1 for inv in (private.get("inventories") or [])
                   if inv.get("WHEAT", 0) > 0)
    hungry = unfed - carriers
    if hungry > 0 and shed.get("WHEAT", 0) > 0:
        # There are only 4 physical shed tiles, but nothing stops two workers
        # detouring through the same one - the assignment loop claims each
        # candidate independently. Capping at 4 candidates capped feeding
        # trips at 4 a turn too, which was enough for a 6-animal herd and not
        # for the herds this agent is now meant to grow: with more than 4
        # animals hungry at once, the rest went unfed with wheat sitting in
        # the shed, occasionally two days running. One candidate per hungry
        # animal, tiles reused, lets every worker who is needed go.
        tiles = _shed_tiles(board)
        for i in range(hungry):
            sx, sy = tiles[i % len(tiles)]
            found.append(("w_pickup", ["PICKUP", "WHEAT", max(hungry, 1)], sx, sy, 0, None))
    # A separate weight from w_pickup: PLACE ranks as the single best
    # candidate on the whole board once an animal is in hand (score ~0.6 of
    # ~15-70 in real games), so the bottleneck is entirely getting a worker
    # to detour and pick one up. Traced across real Kaggle replays: bought
    # animals sat in the shed 10-17 days at a stretch, in wins and losses
    # alike, because w_pickup's shared weight (9.138, tuned for the routine
    # WHEAT/FERTILIZER case) ranked animal pickup around 32nd of ~68
    # candidates every turn - never quite worth a detour, so it never
    # happened. An idle animal earns nothing every day it waits; that is a
    # bigger relative loss than a slightly delayed feed or fertilizer run,
    # so it gets its own, separately tunable priority.
    for animal in ANIMAL_SPEC:
        if shed.get(animal, 0) > 0 and empty_pens:
            sx, sy = _shed_tiles(board)[0]
            found.append(("w_pickup_animal", ["PICKUP", animal, 1], sx, sy, 0, None))

    # FERTILIZE needs fertilizer in hand, the same held-item constraint as
    # FEED - one pickup per tile that qualified and is not already carried.
    fert_jobs = sum(1 for f in found if f[0] == "w_fertilize")
    fert_carriers = sum(1 for inv in (private.get("inventories") or [])
                        if inv.get("FERTILIZER", 0) > 0)
    fert_needed = fert_jobs - fert_carriers
    if fert_needed > 0 and shed.get("FERTILIZER", 0) > 0:
        tiles = _shed_tiles(board)
        for i in range(fert_needed):
            sx, sy = tiles[i % len(tiles)]
            found.append(("w_pickup", ["PICKUP", "FERTILIZER", max(fert_needed, 1)], sx, sy, 0, None))
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
        # consecutive_unfed >= 1 means today is this animal's SECOND miss in a
        # row if it slips again - the escape threshold, verified against the
        # env source (kaggriculture.py: consecutive_unfed >= 2 triggers it).
        # Every routine w_feed job already shares the same flat priority
        # (0.562, the lowest of any candidate) regardless of which pen it is
        # on, so with a big enough herd spread across the board, Hungarian
        # picks whichever subset of feed jobs is CHEAPEST BY DISTANCE that
        # turn - and a pen built far from the shed cluster can lose that
        # distance competition to every closer pen, every single day, with
        # no relation to whether workers or wheat are actually short. Traced
        # directly to a real escape: one pen at (8,2), isolated from a
        # cluster near (0,0)-(5,4), went unfed two full days running while
        # 9 closer animals were fed every day and the shed held 55-69 wheat
        # the whole time. A separate, strictly cheaper priority for the
        # animal already one miss from escaping - the same rescue pattern
        # w_water_urgent already uses for a tile one day from dying - beats
        # every routine feed job on priority alone, so distance can no
        # longer cost it the one day it cannot afford to lose.
        key = "w_feed_urgent" if tile.get("consecutive_unfed", 0) >= 1 else "w_feed"
        jobs.append((key, ["FEED"], x, y, 0, "WHEAT"))
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


def _hungarian(cost):
    """Exact minimum-cost assignment: match every row to a distinct column,
    n rows by m columns, n <= m. O(n^2 * m) primal-dual method (Kuhn-Munkres
    with potentials) - the standard algorithm for the assignment problem,
    reimplemented in plain Python rather than imported from scipy, since a
    submission is one self-contained file with no guaranteed third-party
    packages in the judge's sandbox.

    Returns a list of length n: the column assigned to row i."""
    n, m = len(cost), len(cost[0])
    INF = float("inf")
    u = [0.0] * (n + 1)
    v = [0.0] * (m + 1)
    p = [0] * (m + 1)     # p[j] = the row (1-indexed) currently matched to j
    way = [0] * (m + 1)
    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = [INF] * (m + 1)
        used = [False] * (m + 1)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta, j1 = INF, -1
            for j in range(1, m + 1):
                if used[j]:
                    continue
                cur = cost[i0 - 1][j - 1] - u[i0] - v[j]
                if cur < minv[j]:
                    minv[j] = cur
                    way[j] = j0
                if minv[j] < delta:
                    delta, j1 = minv[j], j
            for j in range(m + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while j0:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
    result = [0] * (n + 1)
    for j in range(1, m + 1):
        if p[j]:
            result[p[j]] = j
    return [result[i] - 1 for i in range(1, n + 1)]


# A pair's cost when the worker cannot actually do that job (does not carry
# the required item). Large enough that the search never prefers it over any
# real score, but finite - the env silently no-ops an infeasible FEED/PLACE
# anyway, so the only cost of a forced pairing is one wasted turn, the same
# as PASS would have cost.
INFEASIBLE = 1e9


def _assign_actions(obs, farm, private, pool=None):
    """One action per worker, chosen to minimise the total score across every
    worker at once - the assignment problem, solved exactly with the
    Hungarian algorithm rather than the greedy pick-the-best-pair-repeatedly
    heuristic this replaced. Greedy can strand a worker on a long walk when a
    swap would have let two workers each take the tile nearer to them; exact
    assignment cannot.

    Idling is never given a competing score of its own - the weights in
    PARAMS have no meaningful zero point (most are positive; doing some job
    has always beaten doing none, at any score, as long as one is feasible)
    - so a worker only ends up on PASS when there are genuinely fewer usable
    candidates than workers, never because the search 'preferred' rest.

    pool may be passed in already computed (agent() shares one board scan
    between this and _market_orders' hiring decision) or left to compute its
    own, so direct calls and tests keep working unchanged."""
    board_size = len(farm["tiles"])
    workers = [tuple(farm["farmer"])] + [tuple(h) for h in farm["hands"]]
    carrying = private.get("inventories") or [{}] * len(workers)
    if pool is None:
        pool = _candidates(obs, farm, private)

    actions = [[PASS] for _ in workers]
    n, j_count = len(workers), len(pool)
    if j_count == 0:
        return actions[0], actions[1:]

    # Extra idle columns only appear when there are more workers than
    # candidates - a hard supply shortage, not a choice - so every real
    # candidate still gets filled first; padding is never large enough to
    # let a worker skip a real job it could have done.
    m = max(j_count, n)
    cost = [[0.0] * m for _ in range(n)]
    # A plant with consecutive_unwatered >= 1 (w_water_urgent's own trigger)
    # dies at today's day-boundary unless it is watered again before then -
    # verified against the environment (_daily_refresh_plants: unwatered a
    # second day in a row converts the tile straight to WEED). Reaching and
    # watering it costs distance + 1 turns; a worker who cannot finish both
    # before the day ends cannot save the tile, no matter what the plain
    # distance-scaled score says. Traced directly to a real death: a worker
    # standing on the tile (distance 0, could water it this turn) was scored
    # exactly tied with sending a worker two tiles away instead - who could
    # only take one step closer before the day rolled over and the tile
    # died regardless of "who" was assigned to it. Soft distance cost has no
    # way to express "impossible", so this is the same INFEASIBLE treatment
    # already used for a job whose required item is not held.
    turns_left_today = TURNS_PER_DAY - obs["hour"]
    for w in range(n):
        wx, wy = workers[w]
        held = carrying[w] if w < len(carrying) else {}
        for c, candidate in enumerate(pool):
            key, _action, tx, ty, _units, needs = candidate
            if needs and not held.get(needs, 0):
                cost[w][c] = INFEASIBLE
            elif (key == "w_water_urgent"
                    and _distance(wx, wy, tx, ty) + 1 > turns_left_today):
                cost[w][c] = INFEASIBLE
            else:
                cost[w][c] = _score(candidate, wx, wy, board_size)

    assignment = _hungarian(cost)
    for w, c in enumerate(assignment):
        if c >= j_count or cost[w][c] >= INFEASIBLE:
            continue  # a padding column, or no feasible job was left for it
        wx, wy = workers[w]
        _key, action, tx, ty, _units, _needs = pool[c]
        if (tx, ty) != (wx, wy):
            actions[w] = [_step_toward(wx, wy, tx, ty)]
        else:
            actions[w] = list(action)

    return actions[0], actions[1:]


def _market_orders(obs, farm, private, pool=None):
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

    prices = obs["market"]["prices"]
    days_left = SEASON_DAYS - obs["day"]
    pens = [t for row in farm["tiles"] for t in row
            if isinstance(t, dict) and t.get("kind") in ("COOP", "PASTURE")]
    empty_pens = sum(1 for t in pens if not t.get("animal"))
    livestock = len(pens) - empty_pens

    inventory = obs["market"]["inventory"]
    shops = obs["town"]["unlocked_shops"]
    fert_pending = private["shed"].get("FERTILIZER", 0)
    planted_tiles = sum(1 for row in farm["tiles"] for t in row
                        if isinstance(t, dict) and t.get("kind") == PLANT)
    ranked = sorted(
        CROP_SPEC,
        key=lambda c: crop_value(c, inventory, pending_units(farm, private, c),
                                 days_left, shops, fert_pending, planted_tiles),
        reverse=True,
    )

    # Tried pricing hires with hire_value() (below) instead of this flat
    # target - fib cost against best_task_value * turns_left/TURNS_PER_DAY,
    # the same $/tile/day crop_value and animal_value already use. Measured
    # worse across the board: v18 and v22 both fell out of a 100% win rate,
    # every already-solid matchup lost several thousand dollars, for a small
    # gain against one still-hopeless opponent. The formula was wrong, not
    # just unlucky - crop_value/animal_value price a FULL day's marginal
    # tile, and scaling that by a fraction of today's turns treats an extra
    # hand as able to do only a fraction of one tile's work, when in the
    # turns it has left it can actually walk to and complete several
    # separate full-value actions on different tiles. That systematically
    # undervalued every hire past the first one or two, and the agent
    # under-hired broadly (confirmed in test_behaviour.py: hires fell
    # 1440->1323, idle share rose). hire_value() is kept below, unused, as a
    # documented false start - the right fix needs a per-hand value that
    # scales with how many actions it can still take, not a day-rate.
    #
    # HANDS_PER_DAY alone is a flat target, blind to the one thing that
    # actually kills tiles: a temporary spike in urgent work outrunning
    # worker count. Traced directly to real deaths: a land purchase triggers
    # a planting burst (2 plants -> 56 plants in 3 days in one traced game),
    # and the wave of same-age tiles all needing water lands on the SAME
    # day - overwhelming the fixed 9-worker cap on that one day even though
    # most days have slack. HANDS_PER_DAY=8 is our own arbitrary constant,
    # not an environment limit (_do_hire only checks money) - fib-scaled
    # hire cost is still cheap against the cash on hand during exactly this
    # kind of burst, so staffing up temporarily should cost far less than
    # the tiles it saves. Respond to how many urgent (deadline-bound) jobs
    # exist right now, not a fixed count.
    # fib-scaled cost means the backlog itself cannot be the hire count -
    # matching a 20-tile burst 1:1 means paying fib(8..27), tens of
    # thousands of dollars for a handful of tiles worth a few hundred each.
    # w_hire_backlog is a hard cap on how many extra hands are ever worth
    # it, not a ratio - the backlog only decides whether to spend up to
    # that cap, never how far past it to go.
    #
    urgent_now = sum(1 for c in (pool or []) if c[0] in ("w_water_urgent", "w_harvest_decay"))
    workers_now = 1 + len(farm["hands"])
    extra_needed = max(0, urgent_now - workers_now)
    extra_hired = min(extra_needed, round(PARAMS["w_hire_backlog"]))

    # An idle animal is a real backlog too - traced across real games (and
    # confirmed present even in wins, so it is not itself what decides a
    # game, just money left on the table): a bought animal sitting in the
    # shed for 10-17 days straight because PICKUP never wins the routine
    # daily competition against watering/feeding, even though PLACE scores
    # as the single best move on the board the moment it is carried (rank 1
    # of ~70 candidates, measured). Lowering PICKUP's own weight to fix that
    # directly was tried and made things worse (-$1,361, 40.6% win rate) -
    # it also started beating water_urgent, trading a saved crop for an
    # idle animal. Hiring an EXTRA hand for it instead doesn't cost that
    # trade-off, PROVIDED it is not sharing a cap with the water/harvest
    # backlog above - the two were combined at first and measured, on the
    # same seeds, byte-for-byte identical to not fixing it at all, because
    # water/harvest alone routinely eats the whole shared cap and leaves
    # nothing for this. A separate cap of its own is the only way either
    # backlog is guaranteed a hand.
    animals_waiting = sum(private["shed"].get(a, 0) for a in ANIMAL_SPEC)
    pickup_backlog = min(empty_pens, animals_waiting)
    extra_pickup = min(pickup_backlog, round(PARAMS["w_hire_pickup_backlog"]))

    # Tried a third backlog term here, hiring extra hands for a placed herd's
    # daily chores (feed/care/collect/harvest), the same pattern as
    # extra_pickup just above. It made things WORSE - 6 of 20 seasons broke
    # the starvation invariant, up from 2 - despite (because of?) a bigger
    # peak herd (12, up from 10-11). Reverted: reactively hiring more once a
    # backlog is already visible cannot be the right fix for a herd the
    # AGENT ITSELF chose to grow past what it can actually service - that is
    # a belief animal_value should hold before recommending the purchase,
    # not a scramble to hire around afterward. See animal_value's
    # service_load for the real fix: its HANDS_PER_DAY denominator is the
    # same stale constant, so it already tries to capture exactly this
    # capacity limit - just calibrated against the wrong number.
    hire_target = HANDS_PER_DAY + extra_hired + extra_pickup
    orders += [["HIRE"]] * max(0, hire_target - farm["hires_today"])

    # Land is worth owning - removing it loses 36 of 40 games - but buying it
    # before the farm can afford to work it starves the seed budget for a third
    # of the season. Hold back a tuned number of days of running costs.
    #
    # That reserve only ever gated on CASH, never on whether the land we
    # already have is even in use - and w_land_reserve tuned to 0.0, so in
    # practice we buy the moment we can afford it, every time. Traced
    # against real Kaggle opponents: our toughest matchups all skip buying
    # land on day 1 and instead rush 6+ pens onto the single starting
    # quadrant first, banking a cash lead from early animal income that
    # compounds all game - we buy land immediately regardless, and never
    # catch up. w_land_utilization requires the CURRENT land to actually be
    # worked (plants + pens, not bare dirt or uncleared weeds) before the
    # next quadrant is worth it, the same argument as the cash reserve but
    # for capacity instead of money. 0 reproduces the old always-buy
    # behaviour exactly; sweep.py should find the real threshold.
    bought = len(farm["unlocked_quadrants"]) - 1
    if bought < MAX_QUADRANTS - 1 and obs["day"] <= LAND_LAST_DAY:
        reserve = PARAMS["w_land_reserve"] * daily_burn(
            farm, private, prices, ranked, livestock)
        if (farm["money"] >= 1000 * 2**bought + reserve
                and _land_utilization(farm) >= PARAMS["w_land_utilization"]):
            orders.append(["BUY_LAND"])
    _av = lambda a: animal_value(a, prices, days_left, inventory,
                                 livestock, private["shed"], shops)
    best_animal = max(ANIMAL_SPEC, key=_av)
    waiting = sum(private["shed"].get(a, 0) for a in ANIMAL_SPEC)
    # Buy for every pen that is standing empty and not already spoken for,
    # not just one at a time - a herd that only grows by one animal a day
    # cannot keep pace with pens that are free to build several at once.
    want = empty_pens - waiting
    # Bootstrapped to at least 1 even with zero pens yet. BUY_ANIMAL needs no
    # pen to exist (verified against the env source - it just lands in the
    # shed like any other purchase), but gating strictly on empty_pens made
    # the very first purchase wait on a WORKER to physically build one first,
    # and pen-building is worker-turn-limited while BUY_SEED is not. Every
    # real Kaggle game traced, wins and losses alike, spent all of day 0's
    # cash on HIRE + two BUY_SEED orders (which already fill all 10 market-
    # order slots) before a pen ever got built, so the first animal wasn't
    # bought until day 5-12 - while both public league opponents (hardcoded
    # day-0 animal buyers) beat us 0/56 games in fitness.py, by -$60k to -$90k.
    if want <= 0 and livestock == 0 and waiting == 0:
        want = 1
    reserve = PARAMS["w_animal_reserve"] * daily_burn(
        farm, private, prices, ranked, livestock)
    # Two narrower attempts - gating this on the single bootstrap call, then
    # on livestock == 0 - both still regressed test_behaviour.py's starvation
    # invariant (0/0 escapes -> 19/19). Traced directly: livestock flips from
    # 0 to 1 the moment the FIRST bought sheep is placed, same day, well
    # before the other 2 of w_pen_ahead's 3 pens finish filling - so a guard
    # scoped to "livestock == 0" turned itself off right as purchases 2 and 3
    # (still using the old, permissive reserve) rushed in and repeated the
    # exact same collapse: all 3 sheep occupied by day 0 hour 20, money at
    # exactly $0 by hour 21, and hiring a hand (as little as $1, the fib
    # schedule resets every day) no longer affordable either - so the lone
    # farmer ran the whole of day 1 alone. There is no game phase where
    # spending the till to zero is actually safe, only ones where it
    # happened not to get tested before now - w_animal_reserve is tuned
    # negative (permissive) on the assumption that BUY_SEED's own 2x-cost
    # restocking gate leaves enough behind, but nothing enforces that two
    # independent purchases (seed and animal) leave a SHARED minimum behind
    # together. Applying this reserve unconditionally costs nothing once the
    # farm is established - by then daily_burn's own hiring term already
    # dwarfs it - so there is no reason to scope it to early game at all.
    reserve = max(reserve, sum(_fib_hire_cost(i) for i in range(HANDS_PER_DAY)))
    # BUY_ANIMAL is processed per unit by the env (verified against source:
    # it buys until money runs out, then simply stops - never fails the
    # whole order), the same as BUY_SEED already relies on elsewhere. This
    # used to require affording all `want` units before submitting any -
    # traced directly to a real, confirmed stall: 3 empty pens, animal_worth
    # strongly positive (SHEEP worth ~$200-250/tile/day), $500-800 in the
    # bank for five straight days, and zero animals bought, because the gate
    # demanded $1,500 (3 x $500) before it would buy even one. Requiring
    # only the first unit's cost lets the env's own per-unit fulfilment do
    # the rest - it buys as many of `want` as it can actually afford.
    if (want > 0 and _av(best_animal) > 0
            and farm["money"] > ANIMAL_SPEC[best_animal][0] + reserve):
        orders.append(["BUY_ANIMAL", best_animal, want])

    # Feed is bought, never grown - a tile costs actions, which are scarcer than
    # money. Wheat is the most shop-drained product in the game, so the price
    # it's checked against is where the drift (w_town_drift) expects it to
    # be, not today's quote.
    #
    # This used to require affording the full `want` (times a 2x buffer, on
    # top of that) before buying any of it - the same all-or-nothing shape
    # traced and fixed for BUY_ANIMAL, here on the one purchase where it is
    # most dangerous: an unfed animal escapes for good after two consecutive
    # missed days. Traced directly: 48 consecutive turns blocked (affordable
    # in part, not in full) across every seed checked, 47 of them with an
    # animal already unfed and zero wheat in the shed - real escape risk,
    # not theoretical. BUY_PRODUCT is processed per unit by the environment
    # (verified against source), so requiring only the first unit's cost
    # lets it buy as much of `want` as it actually can.
    if livestock:
        want = livestock * 3 - private["shed"].get("WHEAT", 0)
        wheat_drift = PARAMS["w_town_drift"] * town_drain_per_day("WHEAT", shops) * (days_left / 2)
        feed_price = price_at("WHEAT", inventory.get("WHEAT", MARKET_I0) - wheat_drift)
        if want > 0 and farm["money"] > feed_price:
            orders.append(["BUY_PRODUCT", "WHEAT", want])

    # Only buy seed for a crop still worth planting. crop_value goes negative
    # once a crop cannot mature before the season ends, and the planting loop
    # already refuses those - but the buying did not, so the agent kept
    # stocking seed it could never use. Roughly $1,000 of dead stock by turn
    # 720, when unsold inventory scores nothing.
    for crop in ranked[:2]:
        if crop_value(crop, inventory, pending_units(farm, private, crop),
                      days_left, shops, fert_pending, planted_tiles) <= 0:
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
    pool = _candidates(obs, farm, private)
    farmer, hands = _assign_actions(obs, farm, private, pool)
    return {
        "farmer": farmer,
        "hands": hands,
        "market": _market_orders(obs, farm, private, pool),
    }
