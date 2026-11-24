"""Anchor states for the supervised step: positions frozen c2 reaches in its own games (every decision kind: actions,
buys and the card-specific subdecisions), against the league's frozen opponents. The supervised update is pulled
toward c2's policy and value on these, so it changes c2 only where the verified labels say so.

    python anchors.py --games 600 --name a1 [--workers 6]
Out: data/anchors_<name>.npz (X, LEGAL, KIND, TURN, GAME, SPLIT: game % 10 == 0 held out) + META.
Half the games c2 plays its argmax (as deployed), half it samples at temperature 1 (as in PPO collection). Kingdoms
avoid manifests/exclusions.json.
"""
import argparse
import json
import multiprocessing as mp
import os
import random
import sys

import numpy as np

from sg.paths import C1, C2, EXCLUSIONS, RL1, ROOT, use_rl3

OPPS = [{"kind": "net", "path": C2}, {"kind": "net", "path": C1}, {"kind": "net", "path": RL1}, {"kind": "ga2"}]


def job(gm):
    use_rl3()
    from rl.core import play
    from rl.jobs import _seat
    k = gm["kingdom"]
    me = _seat({"kind": "net", "path": C2}, k, sample=gm["sample"], record=True)
    op = _seat(OPPS[gm["opp"]], k, sample=False)
    seats = [me, op] if gm["seat"] == 0 else [op, me]
    r = play(seats, k, gm["seed"], rng_seed=gm["seed"] * 31 + 7)
    rec = r["rec"]
    if r["error"] or r["overflow"]:
        return None
    m = rec["SEAT"] == gm["seat"]
    return {"X": rec["X"][m], "LEGAL": rec["LEGAL"][m], "KIND": rec["KIND"][m], "TURN": rec["TURN"][m],
            "GAME": np.full(m.sum(), gm["g"], np.int32)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=600)
    ap.add_argument("--name", default="a1")
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    use_rl3()
    sys.path.insert(0, os.path.join(ROOT, "rl-v3"))
    from rl.core import random_kingdom
    ex = set(json.load(open(EXCLUSIONS))["kingdoms"])
    rng = random.Random(f"sg-v1 anchors {a.name}")
    games = []
    while len(games) < a.games:
        k = random_kingdom(rng)
        if ",".join(sorted(k)) in ex:
            continue
        g = len(games)
        games.append({"g": g, "kingdom": k, "seed": rng.getrandbits(32), "seat": g % 2, "opp": g // 2 % len(OPPS),
                      "sample": g % 4 >= 2})
    with mp.get_context("spawn").Pool(a.workers) as p:
        parts = [x for x in p.imap(job, games, chunksize=4) if x is not None]
    D = {k: np.concatenate([x[k] for x in parts]) for k in parts[0]}
    D["SPLIT"] = (D["GAME"] % 10 == 0).astype(np.int8)          # 1 = held out
    os.makedirs(os.path.join(ROOT, "data"), exist_ok=True)
    meta = {"games": a.games, "kept": len(parts), "states": int(len(D["X"])), "opps": ["c2", "c1", "rl1", "ga2"],
            "policy": "c2 2654, half argmax, half sampled at temp 1"}
    np.savez(os.path.join(ROOT, "data", f"anchors_{a.name}.npz"), META=np.array(json.dumps(meta)), **D)
    print(meta)


if __name__ == "__main__":
    main()
