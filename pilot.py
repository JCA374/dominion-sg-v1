"""Step 2: the pilot dataset, then its frozen held-out bank and the survival check.

    python -u pilot.py --act 0 --states 20000 [--workers 7]        # act 0 buys, 1 buys + actions, 2 actions only

1. Labels (bv-v1 collect.py, run sg_pilot): c2 plays (labels only), the teacher grades with the ablation's settings
   (fixed 32 draws, top-5 shortlist) and validates every graded winner on 32 independent draws; simulator sampler,
   the actual opponent (c2 / c1 / RL-v1 / GA2). Kingdoms: half uniform random, half the predeclared attack / Gardens /
   trasher / engine slices (collect.SLICES); opening (turn <= 4) and endgame (turn > 12) positions are tagged by turn.
   Never a kingdom of manifests/exclusions.json. One cohort id for all shards (checked).
2. The bank (bv-v1 experiments/bank.py): 1,000 held-out positions (test split: kingdoms the update never trains on),
   30% decisive (label spread > 0.2), every legal option at 128 independent draws, the first 100 at 512; replay
   identity required.
3. Survival (survive.py): the label teacher's proposed overrides on the bank under the public sampler and an opponent
   mixture.
Out: bv-v1/data/sg_pilot/, bv-v1/data/bank/sg_bank.npz, results/{pilot,sg_bank,survive_sg_bank}.json
"""
import argparse
import glob
import json
import os
import shutil
import subprocess
import sys

import numpy as np

from sg.paths import BV, EXCLUSIONS, PY, RESULTS, ROOT

RUN = "sg_pilot"


def cohorts():
    out = {}
    for f in glob.glob(os.path.join(BV, "data", RUN, "*.npz")):
        c = json.loads(str(np.load(f)["META"]))["cohort"]
        out[c] = out.get(c, 0) + 1
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--act", type=int, default=0)
    ap.add_argument("--states", type=int, default=20000)
    ap.add_argument("--workers", type=int, default=7)
    ap.add_argument("--skip-collect", action="store_true")
    a = ap.parse_args()
    if not a.skip_collect:
        have = sum(len(np.load(f)["K"]) for f in glob.glob(os.path.join(BV, "data", RUN, "*.npz")))
        if have < a.states:
            cmd = [PY, "-u", "collect.py", "--run", RUN, "--states", str(a.states - have), "--workers", str(a.workers),
                   "--k0", "32", "--k1", "32", "--topn", "5", "--kval", "32", "--act", str(a.act), "--obs", "0",
                   "--exclude", EXCLUSIONS]
            print(" ".join(cmd[2:]), flush=True)
            st = subprocess.run(cmd, cwd=BV, check=True, capture_output=True, text=True).stdout
            print(st[-2000:], flush=True)
    co = cohorts()
    if len(co) != 1:
        raise SystemExit(f"pilot shards are of {len(co)} cohorts: {co}")
    cohort = next(iter(co))
    kinds = {0: "buy", 1: "buy,action", 2: "action"}[a.act]
    json.dump({"run": RUN, "cohort": cohort, "shards": co[cohort], "act": a.act, "kinds": kinds},
              open(os.path.join(RESULTS, "pilot.json"), "w"), indent=1)
    if not os.path.exists(os.path.join(BV, "data", "bank", "sg_bank.npz")):
        subprocess.run([PY, "-u", "experiments/bank.py", "--run", RUN, "--cohort", cohort, "--split", "1", "--kinds",
                        kinds, "--n", "1000", "--k", "128", "--n-hi", "100", "--k-hi", "512", "--decisive-share", "0.3",
                        "--name", "sg_bank", "--workers", str(a.workers)], cwd=BV, check=True)
        shutil.copy2(os.path.join(BV, "results", "sg_bank.json"), os.path.join(RESULTS, "sg_bank.json"))
    if not os.path.exists(os.path.join(RESULTS, "survive_sg_bank.json")):
        subprocess.run([PY, "-u", "survive.py", "--bank", "sg_bank", "--k", "64", "--workers", str(a.workers)],
                       cwd=ROOT, check=True)


if __name__ == "__main__":
    sys.exit(main())
