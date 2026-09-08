# Game economics — derived reference

Working notes on Kaggriculture mechanics, worked out from the competition rules
and verified against the simulator where noted. This is the reasoning behind the
strategy choices in [VERSIONS.md](VERSIONS.md).

## Verified against the simulator

Not from the docs — measured by running episodes and inspecting `obs`:

- Farmer starts at `(4, 4)`. NW quadrant is `x: 0-4, y: 0-4`; everything else
  starts `"LOCKED"`.
- **NORTH decreases y, EAST increases x.** So SOUTH increases y, WEST decreases x.
- `obs["step"]` exists at top level alongside `day` and `hour`.
- Tile actions on a locked tile are a **silent no-op** — no error, no crash. They
  just waste the turn.

## Profit per tile per day

The number to rank crops by. Raw profit is misleading because tiles differ wildly
in how long they stay occupied:

```
profit_per_tile_per_day = (max_yield * base_price - seed_cost) / time_to_max_yield
```

| Crop | Seed | Price | Max yield | Days | Profit/tile/day |
|---|---|---|---|---|---|
| Wheat | $10 | $25 | 4 | 4 | **$35** |
| Melon | $80 | $250 | 6 | 10 | **$142** |

Melon looks 4x better. It is not, for the reason below.

## Why melon cannot be scaled

Melon's glut curve is `sq` with `above_target 3.60`, `T = 300`:

```
price = 250 * (1 - 3.6 * (x / 300)^2)        x = units above I0
```

- 36 melons above I0 → **$237** (barely moves — small scale is safe)
- Solving `3.6 * (x/300)^2 = 1` → **x ≈ 158 → price hits the $1 floor**

And melon appears in **zero** town shops. Its only demand sink is the town
centre, which consumes 1/day — **30 units across the entire season**. So melon
inventory is effectively a one-way ratchet: once you push it up, it never comes
back down, and the crash is permanent for the rest of the game.

At ~18 melons per tile per season, that caps melon at **8–9 tiles**, ever. It
cannot fill even one quadrant.

**Conclusion:** melon is a *burst* crop — a small plot sold early at $250.
Diversification is not a nicety here, it is forced by the price curves. Selling
100 melons + 100 carrots + 100 wheat beats selling 300 melons.

## Why wheat is the volume crop

- Glut side is `log` at `above_target 0.20`: dumping 800 units moves it $25 → $19.
  Practically uncrashable.
- Appears in **5 of the 8 shop types** (bakery, pizza, brunch, ice cream,
  farmers market). Each instance consumes 6/day, so wheat is drained continuously
  and the price is propped back up all season.
- One of only **two** things that can be bought back (with fertilizer).

**But the scarcity side is steep** — `sqrt` at `below_target 0.80`. Buying 400
units drives wheat $25 → $45. Combined with constant shop demand, wheat gets
*more expensive as the season runs*, so plans that assume $25/unit forever are
too optimistic.

## Market arbitrage is dead by design

Buy price is quoted at **post-buy** inventory; sell price at **pre-sell**
inventory. Both land on the same number, so an immediate buy-then-sell round trip
nets **exactly zero**. Pumping scarcity then dumping does not work.

## Grow feed or buy it?

Animals eat 1 wheat/day each, forever. For 6 animals:

- **Buy:** 6 × $25 = **$150/day**, costs zero farmer actions.
- **Grow:** wheat yields ~1 unit/tile/day, so 6 animals need ~6 tiles. Those
  tiles as melon would earn ~$852/day.

Buying wins by roughly 5-6x. **Do not grow your own feed** — with the caveat
above that heavy buying pushes wheat toward $45, which narrows the gap.

Note the animal income itself cancels out of this comparison — it is identical in
both branches.

## The real bottleneck: actions

24 turns/day, one action each. Watering 20 tiles costs 20 of them, before any
movement, harvesting, or planting. Measured on v2, **movement alone was 53%** of
all actions.

This is why the profit-per-tile-per-day table above overstates everything: it
assumes watering is free. Hitting a crop's max yield requires watering **every
day of the bonus window with zero misses** — one skipped day permanently caps
that harvest lower, and it cannot be made up later.

## Hiring is nearly free

`fib(n)` per hand, where n is hires already made **today**, resetting daily:
**1, 1, 2, 3, 5, 8, 13, 21, 34, 55**.

- First hand of the day: **$1**
- Five hands: **$12/day** → $360 for the whole season
- Ten hands: $143/day → ~$4,290 for the season

Each hand gets its own action every turn, i.e. up to 24 extra actions/day.
Hiring is a market order, so it costs no farmer action. Against a current
season profit of ~$4,300, this is the dominant untapped lever.

## Survival rules that bite

- A **fresh seed** starts at `consecutive_unwatered = 1`. Plant without watering
  the same day → weed by morning. No grace period.
- A **fresh animal** starts at `consecutive_unfed = 0` — it survives its first
  day unfed. Opposite of seeds.
- Two consecutive misses: plants become a recoverable weed (costs a `DIG`), but
  animals **escape permanently**. With animals at $300–500 versus seeds at
  $10–80, feeding should generally outrank watering.
- Shed caps at **100 non-seed items**; overflow is **silently discarded**. There
  is no error — the only way to notice is to check `private["shed"]` yourself.
