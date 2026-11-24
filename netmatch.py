"""Plain-network matches on a frozen manifest (both seats per kingdom), through bv-v1's match.py (manifest written
before game one, completed seat pairs, kingdom-pair bootstrap interval).

    python netmatch.py --a models/s1/net.npy --b c2 --manifest dev [--name s1_vs_c2] [--workers 7]
--a / --b: a .npy path, or c2 / c1 / rl1 / ga2. Out: results/matches/match_<name>.json(l).
"""
import argparse
import json
import os
import shutil
import subprocess

from sg.paths import BV, MANIFESTS, PY, RESULTS


def spec(x):
    return x if x in ("c2", "c1", "rl1", "ga", "ga2") else "net:" + os.path.abspath(x)


def play(a, b, manifest, name, workers=7):
    man = os.path.join(MANIFESTS, f"{manifest}.json")
    n = len(json.load(open(man))["games"])
    out = os.path.join(RESULTS, "matches")
    os.makedirs(out, exist_ok=True)
    cmd = [PY, "-u", "match.py", "--name", name, "--base", spec(a), "--opp", spec(b), "--bv", "0", "--games", man,
           "--kingdoms", str(n), "--workers", str(workers), "--resume"]
    subprocess.run(cmd, cwd=BV, check=True, stdout=subprocess.DEVNULL)
    for ext in (".json", ".jsonl"):
        shutil.copy2(os.path.join(BV, "results", f"match_{name}{ext}"), os.path.join(out, f"match_{name}{ext}"))
    return json.load(open(os.path.join(out, f"match_{name}.json")))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--manifest", default="dev")
    ap.add_argument("--name", default="")
    ap.add_argument("--workers", type=int, default=7)
    x = ap.parse_args()
    name = x.name or f"{os.path.basename(os.path.dirname(x.a)) or x.a}_vs_{x.b}_{x.manifest}"
    r = play(x.a, x.b, x.manifest, name, x.workers)
    print(f"{name}: {100 * r['score']:.2f}% {r['ci95']} pairs {r['pairs']} capped {r['capped']} errors {r['errors']}")


if __name__ == "__main__":
    main()
