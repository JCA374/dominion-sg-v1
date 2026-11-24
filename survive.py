"""Do the teacher's proposed overrides survive what a player cannot know? On the frozen bank, every position where the
label's own teacher (its grades, the deployment margin and z gate) would override c2 is regraded with --k fresh draws
three ways: the simulator sampler with the actual opponent (as labelled), the public sampler (the opponent's hidden
cards pooled: bv-v1 bv_set_obs 1) and a frozen opponent mixture in the rollouts (c2, c1, RL-v1, GA2 in turn, for a
policy that is not told who it plays). An override survives when its winner still beats the incumbent by > margin.

    python survive.py --bank sg_bank --k 64 [--workers 7]
Out: results/survive_<bank>.json
"""
import argparse
import json
import multiprocessing as mp
import os

import numpy as np

from sg.paths import BV, RESULTS, use_bv

VIEWS = {"simulator": {"obs": 0, "mix": False}, "public": {"obs": 1, "mix": False},
         "mixture": {"obs": 0, "mix": True}, "public+mixture": {"obs": 1, "mix": True}}


def job(item):
    use_bv()
    from bv import core
    key, didx, kind, b, inc, k, seed = item
    out = {}
    for view, v in VIEWS.items():
        core.configure(k0=k, k1=k, topn=34, obs=v["obs"], act=1 if kind == core.ACTION else 0)
        core.set_mix([core.seat_for(x, key["kingdom"]) for x in ("c2", "c1", "rl1", "ga2")] if v["mix"] else [])
        try:
            g = core.regrade(key, didx, k=k, seed=seed, all_legal=True, kind=kind)
        finally:
            core.set_mix([])
        out[view] = None if g is None else float(g["Q"][b] - g["Q"][inc])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", default="sg_bank")
    ap.add_argument("--k", type=int, default=64)
    ap.add_argument("--margin", type=float, default=0.02)
    ap.add_argument("--z", type=float, default=2.0)
    ap.add_argument("--workers", type=int, default=7)
    a = ap.parse_args()
    B = dict(np.load(os.path.join(BV, "data", "bank", f"{a.bank}.npz")))
    n = len(B["Q"])
    q = np.where(B["CAND16"].astype(bool), np.nan_to_num(B["Q16"], nan=-1), -1)
    inc = B["INC"].astype(int)
    b = q.argmax(1)
    g16 = q[np.arange(n), b] - q[np.arange(n), inc]
    se = np.nan_to_num(B["SE16"][np.arange(n), b], nan=0)
    over = np.where((b != inc) & (g16 > a.margin) & (g16 > a.z * se))[0]
    items = []
    for i in over:
        kk = json.loads(str(B["KEY"][i]))
        items.append((kk["key"], kk["didx"], kk["kind"], int(b[i]), int(inc[i]), a.k, 7_000_003 + int(i)))
    print(f"{len(items)} proposed overrides of {n} bank positions", flush=True)
    with mp.get_context("spawn").Pool(a.workers) as p:
        res = p.map(job, items, chunksize=1)
    bank_gain = B["Q"][over, b[over]] - B["Q"][over, inc[over]]
    out = {"bank": a.bank, "k": a.k, "margin": a.margin, "overrides": len(over), "positions": n,
           "bank_gain": {"mean": round(float(bank_gain.mean()), 4), "survive": round(float((bank_gain > a.margin).mean()), 4)},
           "views": {}}
    for view in VIEWS:
        d = np.array([r[view] for r in res if r[view] is not None])
        out["views"][view] = {"n": int(len(d)), "mean_gain": round(float(d.mean()), 4),
                              "survive": round(float((d > a.margin).mean()), 4),
                              "positive": round(float((d > 0).mean()), 4)}
    kinds = B["KIND"][over]
    out["by_kind"] = {}
    for name, kv in (("buy", 1), ("action", 0)):
        m = kinds == kv
        if m.any():
            out["by_kind"][name] = {view: round(float(np.mean([r[view] > a.margin for r, mm in zip(res, m)
                                                                if mm and r[view] is not None])), 4) for view in VIEWS}
    json.dump(out, open(os.path.join(RESULTS, f"survive_{a.bank}.json"), "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
