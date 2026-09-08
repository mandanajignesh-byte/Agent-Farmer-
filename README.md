# Agent Farmer — Kaggriculture

An agent for the Kaggle [Kaggriculture](https://www.kaggle.com/competitions/kaggriculture)
simulation competition: a 1v1 turn-based farming game, 720 turns (24 turns/day ×
30 days). Most cash in the bank at the end wins. Unsold inventory counts for
nothing.

Not an LLM — submissions run offline in a sandbox with no network, 1.6 vCPUs and
a 100 MiB limit. This is a rule-based agent: a function called once per turn that
reads the game state and returns one action.

```
main.py       the agent (submission entry point)
bench.py      paired head-to-head benchmark with a significance test
tune.py       hill climber over the scoring weights
VERSIONS.md   what each version changed, and what it scored
NOTES.md      derived game economics - the reasoning behind the strategy
TOOLKIT.md    the learning-toolkit README this repo was started from
```

## Current state

**v6** — mean **$10,918** over 15 seeds vs the built-in `starter` agent,
**15/15 wins** (stdev $1,317). Starting money is $3,000, so ~$7,900 profit per
season.

Wheat only, 25 tiles, 6 hired hands per day. No animals, no fertilizer, and
deliberately **no land** — buying it is a measured loss
([why](VERSIONS.md#v6--land-purchase-negative-result)).

The dominant remaining cost is **movement at 65% of all actions**. That is also
what blocks land from paying off, so it is the next thing worth attacking. Full
list in [Known gaps](VERSIONS.md#known-gaps).

## Setup

`kaggle-environments` pulls in `orbax-checkpoint`, which ships paths long enough
to break installs on Windows without long-path support. Those dependencies are
only needed by other environments, so install without them:

```bash
python -m venv .venv
.venv/Scripts/python.exe -m pip install --no-deps kaggle-environments
```

## Running

Single game against a built-in agent (`pass`, `random`, or `starter`):

```bash
python -c "from kaggle_environments import make; env = make('kaggriculture', configuration={'episodeSteps': 720}); env.run(['main.py', 'starter']); print([s.reward for s in env.steps[-1]])"
```

Benchmark over N seeded episodes — **use this to evaluate any change.** Single
episodes are too noisy to compare, since weed spawns and town-shop unlocks are
both random:

```bash
python bench.py 15 starter
```

## Submitting

```bash
kaggle competitions submit kaggriculture -f main.py -m "v8 tuned weights"
```

5 submissions/day; only the latest 2 stay active. Ranking is Elo-style across
many episodes against similarly-rated opponents, finalised with a Bradley-Terry
tournament — so **win rate matters, not margin**. A $1 win counts the same as a
$50,000 win.
