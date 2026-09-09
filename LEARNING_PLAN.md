# Learning plan

The goal of this project is not the agent. It is being able to reason about the
agent — to predict what a change will do, explain why it worked, and design the
next one without help. The leaderboard is how we check that reasoning against
reality.

**Target:** a bronze medal. For a competition this size that means roughly the
top 10% of teams, but confirm the exact threshold on the competition's rules
page before treating that as the number.

## What went wrong in session 1, and the rule that follows

Session 1 went v1 → v14 and improved the agent about 3x. But somewhere around
the tuning work, the sessions stopped being taught and became me running sweeps
and reporting results. Fast progress that leaves the learner behind is a failed
session even when the agent improves.

**The rule for every session from here:** predict before measuring. Before any
experiment runs, write down what you expect and why. The gap between prediction
and result is the entire point — a sweep whose result you couldn't have
predicted teaches nothing unless you first committed to a guess.

---

## Phase 1 — Understand what already exists

Nothing new gets built until the current agent is fully legible.

### 1.1 The agent loop
- What `obs` actually contains, and which fields we use versus ignore
- Why the agent is stateless between turns, and what that forces
- The two output channels: one farmer/hand action versus up to ten market orders

**You should be able to:** given a board state, say what the agent does next turn
and why — before running it.

### 1.2 Candidate generation (`_candidates`)
- How every tile becomes zero or one scored candidate
- The five categories and what distinguishes them
- Why plantable tiles are capped by seed count (planting more than you hold
  makes *every* plant that turn fail)

### 1.3 Scoring and assignment (`_score`, `_assign_actions`)
- The weighted-sum formula and what each term means
- Why the target choice and the action are two separate layers
- Why assignment takes the best *(worker, tile)* pair globally rather than
  letting workers pick in order

### 1.4 Market orders (`_market_orders`)
- Hiring, land, selling, seed buying — and why these cost no farmer action

---

## Phase 2 — Understand the method

This is the part that was rushed, and it matters more than any single change.

### 2.1 Why measurement is hard here
- Per-episode noise exceeds $1,000 — a single game proves nothing
- **Common random numbers:** why both agents must play identical seeds
- **Seat swapping:** why every seed is played from both sides
- What a **binomial p-value** tells you, and what it does not

### 2.2 Why win rate, not money
- Ranking is Elo over wins; a $1 win scores the same as a $50,000 win
- So why do we still *tune* on money margin? (Because win rate over a dozen
  games is too coarse to climb.) When can optimising a proxy mislead you?

### 2.3 The search algorithm
- What **hill climbing** is: propose, evaluate, keep if better
- Why not gradient descent (no gradient), why not a neural net (no labels)
- Why `w_dist` is frozen: scaling all weights equally leaves the decision
  unchanged, so that dimension is redundant
- **Local optima** — why acceptance fell from 11/150 to 2/150, and what random
  restarts or CMA-ES would do about it

### 2.4 Overfitting
- Why we tune on one seed range and validate on another
- **Ablation:** a feature starts at 0.0 and stays only if the search turns it on.
  This is how `w_yield` and `w_sticky` were deleted.

### 2.5 The lesson that produced most of the gains
Four separately-measured conclusions turned out to be **conditional on the
configuration they were measured in**, not permanent:

| Conclusion | True when | False when |
|---|---|---|
| "10 hands is worse than none" | 25 tiles | 75 tiles |
| "Land loses money" | v6 | v10 — worth +$6,590 |
| "Skipping useless watering is pointless" | idle capacity spare | action-starved — +$1,282 |
| "Melon caps at 8-9 tiles" | paper estimate | measured — 14 is better |

**Re-test disabled features after any significant change.**

---

## Phase 3 — The economics (revision)

Mostly derived already, in [NOTES.md](NOTES.md). Worth re-deriving rather than
re-reading.

- Profit per tile per day, and why it beats raw profit for comparing crops
- Why melon cannot be scaled: its glut curve, and having no town shop demand
- Why wheat is the volume crop despite the worst per-tile numbers
- The action budget, and why movement is largely inherent rather than waste

---

## Phase 4 — The work ahead

Ordered by expected value, based on what actually beat us.

### 4.1 Animals — the priority
We lost to an animal-heavy agent, $78,653 to $35,176. Animals were analysed in
session 1 and then set aside for v1 simplicity, and never revisited — exactly
the pattern in 2.5.

To work out together, before writing code:
- Income per animal per day versus wheat feed cost
- The action chain: `BUILD_PASTURE` → `BUY_ANIMAL` → `PLACE` → `FEED` daily
- Why animals should outrank crops when a worker must choose (permanent loss
  versus a recoverable weed)
- Free fertilizer as a byproduct — and whether collecting it pays for the action

### 4.2 Fertilizer
Dead for wheat by arithmetic ($100 for +2 units worth $50). Possibly strong for
melon, where it reaches the yield cap two days earlier — 20% faster cycling on
the most valuable tiles.

### 4.3 Endgame liquidation
Unsold stock scores $0 and nothing special happens on the final day yet.

### 4.4 Better search
Random restarts or CMA-ES, once there are more features worth tuning.

### 4.5 Opponent awareness
Their farm is visible in `obs`. Entirely unexplored.

---

## How to study the replays

```bash
pip install kaggle
# token from https://www.kaggle.com/settings/api
kaggle competitions submissions kaggriculture
kaggle competitions episodes <SUBMISSION_ID> -v
kaggle competitions replay <EPISODE_ID> -p ./replays
```

The replay JSON holds the full turn-by-turn state of both farms. Compare money
curves over time rather than just the final number — *when* they pulled ahead
says far more than *that* they did.

---

## Session shape

1. Review — one question on something from last time
2. One concept, taught and checked
3. Predict — commit to an expected result in writing
4. Measure — run it
5. Reconcile — where prediction and result differ, and why

If a session produces a working change but you could not explain it afterwards,
it did not go well.

---

## Candidate improvements found while studying

Things spotted by reading the code rather than by measurement. Untested — each
needs a held-out benchmark before it ships.

**Urgent watering outranks harvesting a decaying plant.** In `_candidates`, the
`consecutive_unwatered >= 1` branch fires before the `decaying` branch, without
checking whether watering still earns anything. Watering only adds yield inside
the bonus window (days 2-4 for wheat), so a plant at age 6 that is both dying and
decaying gets watered — spending an action to preserve a plant whose yield is
shrinking, instead of banking that yield now.

Worth noting the tuner could never find this: it adjusts the *weights*, while
this is a flaw in the *conditions*. No weight value fixes a branch that should
not have fired. The search can tune what you give it; it cannot restructure the
logic.

**The opponent's farm is never read.** `farms[1 - obs["player"]]` is fully
visible every turn — their crops, their money, their land — and the agent ignores
it entirely. Since crops take days to mature, seeing 40 tiles of wheat planted on
day 3 predicts a price crash around day 7, giving four days to sell ahead of it
and to plant something else. This directly attacks what the replays showed:
wheat at $19 while strawberry sat at $252.

---

## Experiment queued: is the distance term the right shape?

Raised by Jignesh while working through `_score`. Distance is currently linear —
one step costs exactly one unit, and `w_dist` is frozen at 1.0 so every other
weight is denominated in walking steps.

The case for linear: walking N steps really does cost N actions. There is no
economy of scale in walking, so the cost is genuinely linear in the game's own
currency.

The case against: a long walk carries risk a short one does not. Over ten turns
of walking, other plants dry out, jobs become urgent, and another worker may
reach the target first. That risk plausibly grows faster than linearly, which a
linear term cannot express.

**The test.** Add a second distance term, inert at zero:

```python
score = PARAMS[key] + PARAMS["w_dist"] * d + PARAMS["w_dist_sq"] * d**2
```

Then tune. Ablation decides it: if `w_dist_sq` stays at 0.0 the linear model was
right and the term gets deleted, exactly as `w_yield` and `w_sticky` were. If it
grows, long trips deserve a disproportionate penalty.

**Prediction, recorded 2026-09-09 before running.** Jignesh: `w_dist_sq` will
grow � long walks are riskier because there is always plenty else to be doing, so
the opportunity cost of a long trip is worse than the steps alone suggest.

Claude: uncertain, leaning toward it staying near zero. Global assignment already
matches close workers to close jobs, so genuinely long trips may be rare enough
that the quadratic term has little to act on. If it does grow, the interesting
question is whether it is capturing risk or just re-expressing something the
existing weights already encode.

Note this is a change to the *shape* of the formula, not to a weight — the same
category as the branch-ordering flaw above. The search can only tune what it is
given; deciding what to give it is the part that has to be reasoned.

**Seed buffer may throttle early growth.** We buy up to `SEED_BUFFER =
HANDS_PER_DAY + 2` (currently 10) seeds at a time, but the farm has 75 tiles to
fill. Empty tiles compound — a tile idle on day 3 loses its whole first cycle —
so if planting is seed-limited rather than action-limited in the opening days,
raising the buffer is nearly free. Measure whether plant candidates are being
capped by seeds before changing it.

**Workers are never dispersed deliberately.** All workers respawn at the shed
each morning (forced by the game) and fan out with nothing stopping them
crisscrossing. Global assignment gives some of this implicitly by pairing near
workers with near jobs, but there is no notion of a worker owning a region.
Explicit zoning might cut travel further — though note the v9 finding that
movement is largely inherent rather than wasteful, so the ceiling here may be
low.
