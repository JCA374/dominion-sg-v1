"""The ablation's report: per arm score (kingdom-pair bootstrap 95% interval), capped and errored games, rollouts,
overrides by decision kind, latency p50/p95; and paired score differences between arms on the same games.

    python ablation_report.py      -> results/ablation.json
"""
import json
import os

import numpy as np

from sg.paths import RESULTS

ARMS = ["mirror", "buy", "action", "both"]
PAIRS = [("buy", "mirror"), ("action", "mirror"), ("both", "mirror"), ("both", "buy"), ("both", "action"),
         ("action", "buy")]


def load(arm):
    p = os.path.join(RESULTS, "ablation", f"match_sg_abl_{arm}.jsonl")
    if not os.path.exists(p):
        return None
    rows = [json.loads(x) for x in open(p)]
    by = {}
    for r in rows:
        by.setdefault(r["i"], {})[r["seat"]] = r
    return {i: v for i, v in by.items() if len(v) == 2}


def boot(x, n=4000, seed=0):
    rng = np.random.default_rng(seed)
    x = np.asarray(x, float)
    if not len(x):
        return [None, None]
    m = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(n)]
    return [round(float(np.percentile(m, 2.5)), 4), round(float(np.percentile(m, 97.5)), 4)]


def arm_stats(P):
    rows = [r for v in P.values() for r in v.values()]
    pts = [np.mean([r["points"] for r in v.values()]) for v in P.values()]
    lat = np.array([x for r in rows for x in r.get("lat", [])])
    buys = sum(r["buys"] for r in rows)
    ag = sum(r.get("act_graded", 0) for r in rows)
    return {"pairs": len(P), "games": len(rows), "score": round(float(np.mean(pts)), 4), "ci95": boot(pts),
            "capped_games": sum(r["capped"] for r in rows), "errored_games": sum(r["error"] for r in rows),
            "score_without_capped_pairs": round(float(np.mean([np.mean([r["points"] for r in v.values()])
                                                               for v in P.values() if not any(r["capped"] for r in v.values())])), 4),
            "rollouts": sum(r["rollouts"] for r in rows), "rollouts_per_game": round(sum(r["rollouts"] for r in rows) / len(rows)),
            "buy_decisions": buys, "buy_overrides": sum(r["overrides"] for r in rows),
            "buy_override_rate": round(sum(r["overrides"] for r in rows) / max(buys, 1), 4),
            "action_graded": ag, "action_overrides": sum(r.get("act_overrides", 0) for r in rows),
            "action_override_rate": round(sum(r.get("act_overrides", 0) for r in rows) / max(ag, 1), 4),
            "action_override_kinds": {"play where c2 stopped": sum(r.get("act_play_not_stop", 0) for r in rows),
                                      "stop where c2 played": sum(r.get("act_stop_not_play", 0) for r in rows),
                                      "another action": sum(r.get("act_other_action", 0) for r in rows)},
            "latency_s": ({"decisions": int(len(lat)), "p50": round(float(np.percentile(lat, 50)), 4),
                           "p95": round(float(np.percentile(lat, 95)), 4)} if len(lat) else None),
            "secs_per_game": round(float(np.mean([r["secs"] for r in rows])), 2)}


def main():
    data = {a: load(a) for a in ARMS}
    out = {"arms": {a: arm_stats(P) for a, P in data.items() if P}, "paired": {}}
    for x, y in PAIRS:
        if not data.get(x) or not data.get(y):
            continue
        common = sorted(set(data[x]) & set(data[y]))
        d = [np.mean([r["points"] for r in data[x][i].values()]) - np.mean([r["points"] for r in data[y][i].values()])
             for i in common]
        out["paired"][f"{x} - {y}"] = {"pairs": len(common), "diff": round(float(np.mean(d)), 4), "ci95": boot(d, seed=1)}
    json.dump(out, open(os.path.join(RESULTS, "ablation.json"), "w"), indent=1)
    for a, s in out["arms"].items():
        print(f"{a:7s} {100 * s['score']:.2f}% {s['ci95']} n={s['games']} buy-ov {s['buy_override_rate']:.3f} "
              f"act-ov {s['action_override_rate']:.3f} capped {s['capped_games']} err {s['errored_games']} lat {s['latency_s']}")
    for k, v in out["paired"].items():
        print(f"{k}: {100 * v['diff']:+.2f} points {v['ci95']} ({v['pairs']} pairs)")


if __name__ == "__main__":
    main()
