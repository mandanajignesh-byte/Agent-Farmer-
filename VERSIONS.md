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
| v3 `ae22613` | Weigh priority against walking distance | $7,315 mean | 15 seeds, 15/15 wins |

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

## Known gaps

Deliberately unimplemented, roughly in order of expected value:

1. **No hired hands.** Hiring is a market order, so it costs no farmer action,
   and the cost is `fib(n)` per hand per day — the first hand of the day costs
   **$1**. Each hand gets its own action every turn. This is almost certainly
   the largest single lever available and it is completely untouched.
2. **Only 9 of 25 tiles used.** `PLANT` is last in priority, so the farm never
   fills. One farmer cannot water 25 tiles anyway (25 waterings > 24 actions),
   which is why hands come first.
3. **Weeds never cleared.** No `DIG` rule at all; weeds permanently retire tiles.
4. **Wheat only.** No melon plot, so the highest-margin crop is unused.
5. **No animals, no land purchase.**
6. **Dumps the whole shed every turn** with no regard for price impact. Safe for
   wheat, which barely moves on glut; would be ruinous for melon or strawberry.
7. **"Nearest" ignores value** — a wheat plant 1 step away outranks a melon 4
   steps away when both are dying.
