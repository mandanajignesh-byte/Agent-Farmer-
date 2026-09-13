# Every action, priced

One worker does one thing per turn. With the farmer plus 8 hands that is
**9 worker-actions per turn, 216 per day, 6,480 per season** — the real budget.
Money is recoverable; a spent action is not.

Every row below is checked against `kaggle_environments/envs/kaggriculture.py`,
not against memory. Where our model disagrees with the rules, it says so.

## The rules that set the prices

```
weeds        consecutive_unwatered >= 2  -> tile becomes WEED
             planting day counts as 1 unwatered, so a new plant must be
             watered on its planting day or the next one
escape       consecutive_unfed >= 2      -> animal gone, structure remains
water bonus  non-ongoing crops only, and only inside
             [(max_yield_day+1)//2, max_yield_day]
harvest gate day - planted_day >= first_yield_day, regardless of yield_units
shed         100 units, shared across all products
hire         fib(n) with mult 1: 1,1,2,3,5,8,13,21 = $54 for eight hands
land         $1,000 / $2,000 / $4,000
```

## Crop truth table

| crop | seed | first yield | max yield day | units (no fert) | $/unit @ base |
|---|---|---|---|---|---|
| WHEAT | $10 | day 2 | day 4 | 4 | 25 |
| CARROT | $20 | day 2 | day 3 | 3 | 35 |
| TOMATO | $50 | day 8 | day 11 | 4 (ongoing) | 60 |
| STRAWBERRY | $100 | day 10 | day 16 | 4 (ongoing) | 120 |
| MELON | $80 | day 10 | day 12 | 6 | 250 |

Units are derived, not looked up: a plant starts at 1 and each watering inside
the bonus window adds 1, capped at `max_yield`. Wheat gets 3 waterings (days
2-4) so 1+3 = 4. Melon's window is days 6-12, seven waterings, but the cap of 6
binds first — reached on day 10, which is why `MAX_YIELD_DAY["MELON"] = 10`
rather than the rulebook's 12. Harvesting on day 10 takes full yield two days
early.

**Ongoing crops (tomato, strawberry) do not respond to watering at all.**
Watering them only prevents weeds. `BONUS_START` computes a bonus window for
them anyway, so a watering on a tomato is scored `w_water_bonus` when it earns
nothing. Harmless today only because the crop ranking never picks them.

## Worker actions

| action | costs | earns | our weight | verdict |
|---|---|---|---|---|
| **CARE** | 1 action | banks +1 unit paid on next production — $160-200 | `w_care 3.0` | **best return on the farm** |
| **HARVEST** (animal) | 1 action | 1-4 units, up to `max_held` | `w_harvest_animal 3.5` | high |
| **COLLECT_FERTILIZER** | 1 action | 1 unit @ $100 base, free daily per animal, **does not accumulate** | `w_collect 3.5` | high, and was unclaimed until v17 |
| **HARVEST** (crop) | 1 action | full yield; non-ongoing frees the tile | `w_harvest_ripe 4.517` | high |
| **WATER** (rescue) | 1 action | saves a whole tile from becoming a weed | `w_water_urgent 5.907` | high |
| **FEED** | 1 action + 1 wheat (~$25) | prevents a $300-500 loss | `w_feed 0.562` | cheap insurance |
| **WATER** (in window) | 1 action | +1 unit of yield | `w_water_bonus 7.903` | medium |
| **PLANT** | 1 action + seed | starts a `crop_value` stream | `w_plant 10.623` | medium |
| **DIG** | 1 action | clears a weed back to plantable | `w_dig 10.988` | medium |
| **BUILD_COOP/PASTURE** | 1 action + a tile | enables an animal | `w_build 8.0` | conditional |
| **PICKUP / DROP / PLACE** | 1 action | pure logistics, earns nothing directly | `8.238 / 4.0 / -0.02` | overhead |
| **WATER** (outside window) | 1 action | **nothing** on a healthy plant | `w_water_idle 30.0` | correctly suppressed |
| **FERTILIZE** | 1 action + $100 of fertilizer | doubles the watering bonus for 3 days | **never emitted** | see below |
| **move** | 1 action | nothing | `w_dist 1.0` | 56% of all actions |
| **PASS** | 1 action | nothing | — | 18% of all actions |

### Why FERTILIZE stays unused — and it is not an oversight

Fertilizer makes each watering worth +2 instead of +1, for 3 days.

- **Wheat**: 3 waterings, 1+2+2+2 = 7, capped at 6. Yield goes 4 -> 6, so
  +2 units ≈ **$50**. The fertilizer itself sells for ≈ **$100**. Selling wins.
- **Melon**: already reaches its cap of 6. Extra yield is impossible, and
  hitting the cap earlier buys nothing because `HARVEST` is gated on
  `first_yield_day = 10` no matter how many units are banked.
- **Tomato / strawberry**: fertilizer genuinely does speed these up, reaching
  4 units by day 9 instead of day 11 — but the crop ranking never plants them.

So the action is correctly skipped **given our crop mix**. If the mix ever
shifts to ongoing crops, this flips.

## Market orders (10 per turn, and only ~3% of turns hit the cap)

| order | our rule | issue |
|---|---|---|
| **SELL** | everything in the shed except feed wheat | placed first, correctly |
| **HIRE** | top up to 8/day | $54/day total — not a real cost |
| **BUY_SEED** | top 2 crops by `crop_value`, buffer of 10 | **blocked by cash on 510 of 720 turns** |
| **BUY_ANIMAL** | when `animal_value > 0` and a pen is free | **blocked by cash on 244 turns** |
| **BUY_PRODUCT** | 3 days of feed per animal | never blocked |
| **BUY_LAND** | `money >= price + 500` | fires **day 0**, causing the two blocks above |

## Open defects

1. ~~`LAND_RESERVE = 500` is a guess.~~ **Fixed.** `daily_burn()` replaced the
   flat $500 with hire cost (the real fib schedule) + seed cost (what the
   ranking would actually buy) + feed cost (live wheat price), scaled by a
   tunable `w_land_reserve` (starts at 0). Same pattern now used for
   `w_animal_reserve` on animal purchases.

2. ~~`animal_value` ignores `first_yield_day`.~~ **Fixed.** `produce` is scaled
   by `(days_left - first_yield) / days_left`, so a late purchase that will
   never see a payout is valued at zero produce (fertilizer, which starts on
   placement day, is exempt).

3. ~~Ongoing crops are scored for a watering bonus they cannot receive.~~
   **Fixed.** `BONUS_START` excludes TOMATO and STRAWBERRY; only crops that
   actually have a bonus window are checked for one.

4. **Movement is ~56% and PASS is ~15%.** Together, most of every action the
   farm takes produces nothing. This is a task-allocation / routing question,
   not a value-function one — `w_dist`/`w_dist_sq` already price a trip's
   length, but nothing yet prices a worker's opportunity cost against standing
   still. Unstarted.

5. **`MAX_QUADRANTS = 3` and `HANDS_PER_DAY = 8` are structural constants, not
   `PARAMS` weights.** They were swept together in a controlled experiment
   (four quadrants loses at every staffing level, because the fib hire cost
   caps the workforce before that much land can be worked) rather than
   guessed, but they are not searchable by `tune.py`/`tune_cma.py` the way
   every dollar-weight is - both change the size of the action space itself
   (board width, hire schedule length), which is a different kind of variable
   than a score weight. Left as-is; revisit only if land/labour strategy
   changes enough to warrant re-sweeping them.

6. **One residual escape in 13 (test_behaviour.py) still starves an animal
   with wheat in the shed.** Down from 89% of unfed-turns being unaddressed
   before the carrier-count fix, and from a hard 4-workers-per-turn pickup
   cap before this session's fix. What is left looks like routing latency - a
   worker picks up wheat but cannot reach the pen before day-end - not a
   capacity gap. Unconfirmed; not yet worth its own diagnostic.

7. **Town drift (`w_town_drift`) and escape risk (`w_escape_risk`) are new,
   both start at 0.** `animal_value` and `crop_value` now price wheat, milk,
   wool, egg and carrot against the town's verified daily drain
   (`town_drain_per_day`, exact in `test_model.py`) averaged over the
   remaining horizon, and animal survival against a labour-capacity ratio -
   but at their default weight of zero, both are pure structure with no
   effect yet. Measured once against a strong opponent (v22): applying the
   town drift at full trust (as if we were the only other seller) lost
   $5,493/game, because the opponent's own selling cancels part of the drift.
   That is exactly the kind of thing tuning, not a guess, should set.
