"""What do prices do across a real game, and when do we sell into them?

The town drains market inventory all season, which lifts prices. We sell the
moment produce reaches the shed. If those two run in opposite directions, we
are systematically selling at the worst prices of the game.
"""
import collections, multiprocessing as mp, statistics, sys
from kaggle_environments import make
import main as agent

WATCH = ("WHEAT", "MELON", "CARROT", "EGG", "MILK", "WOOL", "FERTILIZER")

def play(seed):
    price = collections.defaultdict(dict)
    sold = collections.defaultdict(lambda: collections.defaultdict(int))
    revenue = collections.Counter()
    shops = []
    def spy(obs):
        act = agent.agent(obs)
        d = obs["day"]
        for p in WATCH:
            price[p][d] = obs["market"]["prices"].get(p, 0)
        for o in act["market"]:
            if o[0] == "SELL":
                sold[o[1]][d] += o[2]
                revenue[o[1]] += o[2] * obs["market"]["prices"].get(o[1], 0)
        shops[:] = obs["town"]["unlocked_shops"]
        return act
    env = make("kaggriculture", configuration={"episodeSteps":720, "seed":seed})
    env.run([spy, "champion.py"])
    mine, _ = (s.reward for s in env.steps[-1])
    return {"price": {k: dict(v) for k, v in price.items()},
            "sold": {k: dict(v) for k, v in sold.items()},
            "revenue": dict(revenue), "shops": list(shops), "mine": mine}

if __name__ == "__main__":
    with mp.Pool(8) as pool:
        games = pool.map(play, range(3000, 3008))
    days = [0, 4, 8, 12, 16, 20, 24, 28]
    print("PRICE BY DAY (mean over 8 games)")
    for p in WATCH:
        print(f"  {p:<11}" + "  ".join(
            f"d{d}={statistics.mean(g['price'][p].get(d,0) for g in games):>6.0f}"
            for d in days))
    print("\nUNITS WE SELL BY DAY")
    for p in WATCH:
        tot = sum(sum(g["sold"].get(p,{}).values()) for g in games)/len(games)
        if tot < 1: continue
        print(f"  {p:<11}" + "  ".join(
            f"d{d}={statistics.mean(g['sold'].get(p,{}).get(d,0) for g in games):>6.1f}"
            for d in days) + f"   total {tot:.0f}")
    print("\n  revenue: " + "  ".join(
        f"{p}=${statistics.mean(g['revenue'].get(p,0) for g in games):>8,.0f}"
        for p in WATCH if statistics.mean(g['revenue'].get(p,0) for g in games) > 100))
