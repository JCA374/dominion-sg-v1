#!/usr/bin/env python3
"""Record games of branch B against branch A for the analysis -> results/divergence/<model>/episodes.jsonl.gz

    python experiments/record_games.py B_s2 --vs A_s2 [--episodes 120]

Analysis kingdoms: a fresh stream ("sg-v1 analysis"), none of manifests/exclusions.json. Each kingdom is played in
both seats, B arg-max against A arg-max. Every decision of B with two or more options is a frame: its features, its
choice, and what A and frozen c2 would choose on the same position (their arg-max on the same inputs). Outcome:
failed = B lost the game. The analysis's event is a divergence: a frame where B's choice differs from A's.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)
from far import Recorder  # noqa: E402
from sg.paths import C2, EXCLUSIONS, RL3, use_rl3  # noqa: E402

use_rl3()
import torch  # noqa: E402
from rl.core import NET, NAME, KINDS, Seat, base_genome, outcome, play, random_kingdom  # noqa: E402
from rl.net import from_flat  # noqa: E402

N = len(NAME)


def net_path(run):
    return os.path.join(RL3, "runs", run, "net_100.npy")


def argmax(net, X, L):
    with torch.no_grad():
        lg, v = net(torch.from_numpy(X), torch.from_numpy(L).bool())
        p = torch.softmax(lg, -1)
    return lg.argmax(1).numpy(), p.numpy(), v.numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--vs", default="A_s2")
    ap.add_argument("--episodes", type=int, default=120)
    a = ap.parse_args()
    pb, pa = np.load(net_path(a.model)), np.load(net_path(a.vs))
    nb, na, nc = from_flat(pb).eval(), from_flat(pa).eval(), from_flat(np.load(C2)).eval()
    ex = set(json.load(open(EXCLUSIONS))["kingdoms"])
    rng = random.Random("sg-v1 analysis")
    out = os.path.join(ROOT, "results", "divergence", a.model, "episodes.jsonl.gz")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    header = {"model": a.model, "vs": a.vs, "fps": None, "failure": f"{a.model} loses the game against {a.vs}",
              "event": f"a decision where {a.model}'s choice differs from {a.vs}'s on the same position"}
    with Recorder(out, header) as rec:
        k = 0
        while k < a.episodes:
            kg = random_kingdom(rng)
            seed = rng.getrandbits(32)
            if ",".join(sorted(kg)) in ex:
                continue
            for s in (0, 1):
                g = base_genome(kg)["ints"]
                me, op = Seat(NET, g, pb, record=True), Seat(NET, g, pa)
                r = play([me, op] if s == 0 else [op, me], kg, seed, rng_seed=seed ^ 0x5EED)
                rec_ = r["rec"]
                m = rec_["SEAT"] == s
                X, L, A = rec_["X"][m], rec_["LEGAL"][m], rec_["A"][m]
                ia, _, _ = argmax(na, X, L)
                ic, _, _ = argmax(nc, X, L)
                _, pbp, vb = argmax(nb, X, L)
                top2 = np.sort(pbp, 1)[:, -2:]
                frames = {"kind": rec_["KIND"][m], "turn": rec_["TURN"][m], "b": A, "a": ia, "c2": ic,
                          "coins": X[:, 5 * N] * 8, "buys": X[:, 5 * N + 1] * 2, "vp_diff": X[:, 5 * N + 6],
                          "provinces": X[:, 5 * N + 7], "empty": X[:, 5 * N + 8], "legal": L.sum(1),
                          "b_conf": top2[:, 1] - top2[:, 0], "b_value": vb}
                res = outcome(r, s)
                rec.add({"seed": seed, "seat": s, "kingdom": kg, "failed": res < 0, "cause": "loss" if res < 0 else
                         "tie" if res == 0 else "win", "steps": int(m.sum()), "score": r["scores"][s] - r["scores"][1 - s]},
                        frames)
                k += 1
            print(f"\r{k}/{a.episodes}", end="", flush=True)
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
