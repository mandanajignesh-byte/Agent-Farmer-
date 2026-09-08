# Version log

Every version is a real commit — `git checkout <sha>` and re-run `bench.py`
to reproduce any row here.

Benchmarks use `episodeSteps: 720` (a full 30-day season). Later versions are
measured over 15 fixed seeds because single episodes are too noisy to compare:
weed spawns and town-shop unlocks are both random, and the run-to-run stdev is
around $436. **Differences under roughly $250 are noise, not signal.**

Starting money is $3,000, so subtract that to read profit.

| Version | Strategy change | vs starter | Notes |
|---|---|---|---|
| v1 `4f3ee6d` | Rule-based wheat farmer, strict priority | $3,911 | single episode |
| v2 `b695191` | Harvest at max yield, not first yield | $7,754 | single episode |
| v3 `ae22613` | Weigh priority against walking distance | $7,315 mean | 15 seeds, 15/15 — **no real gain** |
| v4 `70cbdde` | Hire 6 hands/day, claim-based assignment | $9,733 mean | 15 seeds, 15/15 |
| v5 `cbdc924` | Clear weeds with DIG | **$10,918 mean** | 15 seeds, 15/15 |
| v6 `1a244ba` | Land purchase — measured, left **disabled** | $10,918 mean | buying land *loses* money |

---

## v1 — rule-based wheat farmer

**Architecture** (unchanged since, and worth keeping): the agent is a pure
function called once per turn, with no memory between calls. All state is read
back out of `obs` — e.g. a plant's age comes from `obs["day"] - tile["planted_day"]`
rather than from anything the agent remembers.

Two layers:

1. **Choose a target** — scan every tile, bucket it by priority.
2. **Act or approach** — standing on the target, do the action; otherwise take
   one step toward it.

**Priority order**, ranked by cost of delaying one turn:

| # | Condition | Action | Why here |
|---|---|---|---|
| 1 | `consecutive_unwatered >= 1` and not watered today | WATER | Dies tonight — seed, actions, and the whole harvest are lost |
| 2 | Has yield and past `max_lifespan_step` | HARVEST | Actively shedding 1 unit every other turn |
| 3 | Has yield and ripe | HARVEST | Safe to postpone, costs nothing to wait |
| 4 | Not watered today | WATER | Yield optimisation, not survival |
| 5 | Empty tile, seed in hand, ≥2 turns left in day | PLANT | Needs a spare turn to water it the same day |
| 6 | — | PASS | |

Rule 1 outranks harvesting because the losses are asymmetric: a melon dying on
day 9 of 10 destroys ~$1,500 of nearly-realised value, while postponing a
harvest by a turn usually costs zero.

Rule 5's "≥2 turns left" exists because a fresh seed starts at
`consecutive_unwatered = 1` — plant it without watering the same day and it is
a weed by morning. There is no grace period.

**Market orders** run in parallel and cost no farmer action, so there is never a
reason to hold one back: sell the entire shed every turn, keep 3 wheat seeds in
reserve.

**Result:** $3,911 vs starter. Wins, but only $874 profit on a $3,000 stake.

---

## v2 — harvest at max yield

**The bug:** rule 3 fired as soon as `yield_units > 0`. For wheat that is day 2,
which collects **1 unit**. Waiting until day 4 collects **4**. The agent was
running 2-day cycles at 0.5 units/tile/day instead of 4-day cycles at 1.0 —
half throughput.

**The fix, and why it is age-based:** the obvious patch is "harvest when
`yield_units == max_yield`", but `max_yield` is not in the tile dict, and it
depends on fertilizer (wheat is 4 unfertilized, 6 fertilized). An agent that
never fertilizes and waits for 6 would wait forever, and the plant would decay
into a weed — strictly worse than harvesting early. Age is invariant to all of
that: `obs["day"] - tile["planted_day"] >= MAX_YIELD_DAY[crop]`.

**Second bug fixed:** a plant holding yield but not yet ripe was caught by the
harvest branch and so never reached the watering branch — it went dry while
waiting to be harvested.

**Result:** $3,911 → $7,754. One condition, revenue doubled.

---

## v3 — weigh priority against distance

**What the profiler showed** on v2 over 720 turns:

```
WATER     213   29.6%
movement  382   53.1%   <-- over half of every turn spent walking
PLANT      58    8.1%
HARVEST    49    6.8%

day 25: 9 of 25 tiles planted, 2 uncleared weeds
```

Strict priority thrashes. Standing on a tile that needs water (priority 4) with
a ripe harvest 5 tiles away (priority 3), it walks 5, harvests, walks back —
**11 actions for 2 useful ones**, when watering first would have cost 6.

**The fix:** score each candidate `priority * W + distance`, take the minimum.
`W` has to be large enough that a plant dying tonight is never skipped for a
convenient chore underfoot — with `W=1`, distance (0..18) swamps priority (1..5)
and the agent lets a dying plant go to save a few steps.

**Sweep over 15 seeds:**

| W | mean | | W | mean |
|---|---|---|---|---|
| 1 | $7,216 | | 3 | $7,132 |
| 2 | $7,315 | | strict | $7,104 |

**This is a negative result and is recorded as one.** The $211 spread sits
inside one standard error ($436 stdev). Weighting removes a pathological case;
it does not measurably earn money. Kept at `W=2` because it is the best point
estimate and costs nothing.

**Result:** mean $7,315, median $7,240, range $6,499–$7,900, **15/15 wins**.

---

## v4 — hire farm hands

Hiring is a market order, so it costs **no farmer action**, and `fib(n)` makes
the first few nearly free — the first hand each day costs **$1** for a full extra
action every turn.

**The design problem:** running v3's target selection once per worker sends
*every* worker to the same tile, since they all score the same pool from their
own position. Splitting workers into roles ("farmer waters, hands harvest")
doesn't fix it either — five hands with harvest duty and one ripe tile still
collide.

**The fix:** assign sequentially, farmer first, and **pop each claimed tile out
of the pool**. Workers spread out on their own, with no role assignment.

Also caps the plantable pool at seed count — planting more tiles in a turn than
you hold seeds for makes **every** PLANT that turn fail, not just the surplus.

**Sweep over hands/day** (12 seeds) — an inverted U, where 10 hands is worse
than hiring nobody:

| hands | 0 | 2 | 4 | **6** | 8 | 10 |
|---|---|---|---|---|---|---|
| mean | $7,012 | $9,021 | $8,935 | **$9,869** | $7,927 | $5,927 |

fib explodes at the tail: 6 hands cost $20/day, 10 cost $143/day — a $3,690
seasonal gap that closely tracks the observed collapse. With only 25 tiles there
is also very little for a 10th worker to do.

**Result:** $7,315 → $9,733.

---

## v5 — clear weeds

Profiling v4 showed weeds climbing **0 → 4 → 9 and sticking**: 9 of 25 tiles
permanently retired, with planted tiles falling 21 → 16 to match.

A detail worth noting: `empty` was 0 from day 5 onward, and weeds only spawn
randomly on *empty* ground — so those weeds weren't random. They were plants
dying and decaying in place.

`DIG` goes **last**, above only PASS. Weeds don't worsen with time so digging is
never urgent, and 31% of worker-actions were idle anyway — the work is free. It
also feeds planting, since a weed occupies a tile PLANT would otherwise use.

Weeds now go **0 → 3 → 1 → 0** and stay clear, for **26 DIG actions** all season.
Planted tiles rise 16 → 25 (full farm), idle falls 31% → 13%.

**Result:** $9,733 → **$10,918**.

---

## v6 — land purchase (negative result)

Buying land looked like the obvious next step once hiring provided the labour.
It isn't. Enabled, it cost more than a third of earnings and dropped the win rate
to **7/15**.

**Joint sweep** — land and labour are coupled, so tuning them separately would
have been the mistake (12 seeds):

| | hands=6 | hands=8 | hands=10 |
|---|---|---|---|
| **1 quad** (25 tiles) | **$11,048** | $10,128 | $7,445 |
| **2 quads** (50 tiles) | $9,555 | $10,765 | $9,237 |

The coupling is real — with land the optimum *does* shift to more hands (8 rather
than 6). But even optimally staffed, land never catches up.

**Why:** labour has a hard ceiling because fib cost explodes (11th hand $89/day,
14th $377/day), and ~65% of every worker's turn is already spent walking a 5×5
quadrant. Doubling the area doubles the walking without adding workers to absorb
it, so tiles miss waterings and decay. **25 tiles is already past the optimum.**

Left behind `MAX_QUADRANTS = 1` with the numbers recorded. Cheap to revisit — but
the blocker is **movement efficiency**, not the land price.

---

## Known gaps

Roughly in order of expected value:

1. **Movement is 65% of all actions** — the single dominant cost, and the thing
   blocking land from ever paying off. Two angles: assignment is greedy and
   sequential (a globally optimal worker→tile matching would beat it), and
   nothing encourages planting in clusters near the shed where workers spawn.
2. **Wheat only.** The highest-margin crop is unused. Melon is worth ~$142/tile/day
   against wheat's $35 — but caps at 8-9 tiles for the whole season
   (see [NOTES.md](NOTES.md#why-melon-cannot-be-scaled)), so it's a small plot
   alongside wheat, not a replacement.
3. **No fertilizer.** Wheat maxes at 4 units unfertilized versus 6 fertilized —
   a 50% yield increase on every tile, currently untouched.
4. **No animals.** Ongoing income, and they produce fertilizer as a free byproduct.
5. **Dumps the whole shed every turn** with no regard for price impact. Safe for
   wheat, which barely moves on glut; would be ruinous for melon or strawberry.
6. **"Nearest" ignores value** — a wheat plant 1 step away outranks a melon 4
   steps away when both are dying.
7. **No endgame liquidation.** Unsold inventory scores $0, so the last day should
   dump everything; currently nothing special happens at the end.
