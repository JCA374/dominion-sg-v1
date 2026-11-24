"""Frozen game manifests and the shared kingdom exclusion list, written once before any match.

    python manifests.py            # writes manifests/{ablation,dev,matrix,promotion}.json + exclusions.json

ablation   300 kingdoms   the matched teacher ablation (step 1)
dev        300 kingdoms   checkpoint tests while developing (supervised step, PPO checkpoints)
matrix     500 kingdoms   the head-to-head matrix (A vs B, each vs c2, c1, RL-v1, GA2)
promotion 2000 kingdoms   the promotion match (once, at the end)

Each game is (i, kingdom, seed) and is played in both seats. Kingdoms are fresh: none is in another manifest or in any
of bv-v1's evaluation streams. exclusions.json lists every one of them plus bv-v1's streams: PPO training (rl-v3 cfg
exclude) and label collection (bv-v1 collect --exclude) never deal them. Refuses to overwrite existing manifests.
"""
import hashlib
import json
import os
import random
import sys

from sg.paths import BV, EXCLUSIONS, MANIFESTS, use_bv

SIZES = {"ablation": 300, "dev": 300, "matrix": 500, "promotion": 2000}


def canon(k):
    return ",".join(sorted(k))


def main():
    use_bv()
    sys.path.insert(0, BV)
    from bv.core import random_kingdom
    from match import eval_kingdoms
    os.makedirs(MANIFESTS, exist_ok=True)
    if any(os.path.exists(os.path.join(MANIFESTS, f"{n}.json")) for n in SIZES):
        raise SystemExit("manifests exist: they are frozen (delete them only if no game was played on them)")
    taken = set(eval_kingdoms())
    bv_streams = sorted(taken)
    out = {}
    for name, n in SIZES.items():
        rng = random.Random(f"sg-v1 {name} kingdoms")
        games = []
        while len(games) < n:
            k = random_kingdom(rng)
            s = rng.getrandbits(32)
            if canon(k) in taken:
                continue
            taken.add(canon(k))
            games.append({"i": len(games), "kingdom": k, "seed": s})
        body = {"name": name, "games": games, "note": f"{n} fresh kingdoms, both seats each; sg-v1 manifests.py"}
        body["sha"] = hashlib.sha256(json.dumps(games).encode()).hexdigest()[:16]
        json.dump(body, open(os.path.join(MANIFESTS, f"{name}.json"), "w"), indent=0)
        out[name] = body["sha"]
    ex = sorted(taken)
    json.dump({"note": "kingdoms never trained or labelled on: every sg-v1 manifest + bv-v1's evaluation streams",
               "manifests": out, "bv_streams": len(bv_streams), "kingdoms": ex}, open(EXCLUSIONS, "w"), indent=0)
    print({**out, "excluded": len(ex)})


if __name__ == "__main__":
    main()
