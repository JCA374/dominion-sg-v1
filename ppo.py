"""Step 4: matched PPO. Branch A starts from plain c2, branch B from the supervised c2; both get the same PPO settings,
the same game budget and, per training seed, the same training stream (rl-v3 cfg seed: the same kingdoms, seeds,
seats and opponents at every iteration). Runs go in rl-v3/runs/<name> (the pinned submodule's train.py).

    python -u ppo.py --b models/s1/net.npy --seeds s1,s2 --iterations 100 [--workers 6]

Order: A_s1, B_s1, A_s2, B_s2 (interleaved, so a stop leaves comparable pairs); each resumes with --continue.
Settings (c2's own, from runs/c2/config.json): open league (--stage 2), lr 2e-4, card drill 0.3, temperature 1, fresh
Adam; plus frozen c2 and c1 in the league (10% each), evaluation also against them, the shared exclusion manifest,
no rl-v3 goal checks. A finished run's net_<it>.npy are its checkpoints.
"""
import argparse
import json
import os
import subprocess
import sys

from sg.paths import C2, EXCLUSIONS, PY, RL3, ROOT

COMMON = ["drill_share=0.3", "lr=0.0002", "temp=1.0", "frozen=c2:0.1,c1:0.1", "eval_vs=c2,c1",
          f"exclude={EXCLUSIONS}", "goal_every=0", "games=3000", "chunk=50"]


def run(name, init, seed, iterations, workers):
    d = os.path.join(RL3, "runs", name)
    done = 0
    if os.path.exists(os.path.join(d, "ckpt.pt")):
        import torch
        done = torch.load(os.path.join(d, "ckpt.pt"), weights_only=False)["iter"]
    if done >= iterations:
        print(f"{name}: done ({done} iterations)", flush=True)
        return
    sets = COMMON + [f"seed={seed}", f"iterations={iterations}", f"workers={workers}"]
    if done or os.path.exists(os.path.join(d, "ckpt.pt")):
        cmd = [PY, "-u", "train.py", "--run", name, "--continue", "--set", f"iterations={iterations}",
               f"workers={workers}"]
    else:
        cmd = [PY, "-u", "train.py", "--run", name, "--init_from", os.path.abspath(init), "--stage", "2", "--set", *sets]
    print(" ".join(cmd[2:]), flush=True)
    with open(os.path.join(ROOT, "logs", f"ppo_{name}.log"), "a") as f:
        subprocess.run(cmd, cwd=RL3, stdout=f, stderr=subprocess.STDOUT, check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", default=C2, help="branch A's start (plain c2)")
    ap.add_argument("--b", required=True, help="branch B's start (the supervised c2, .npy)")
    ap.add_argument("--seeds", default="s1,s2")
    ap.add_argument("--iterations", type=int, default=100)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--tag", default="", help="run name suffix")
    a = ap.parse_args()
    plan = []
    for s in a.seeds.split(","):
        plan += [(f"A{a.tag}_{s}", a.a, s), (f"B{a.tag}_{s}", a.b, s)]
    json.dump({"plan": plan, "iterations": a.iterations, "common": COMMON},
              open(os.path.join(ROOT, "results", f"ppo_plan{a.tag}.json"), "w"), indent=1)
    for name, init, seed in plan:
        run(name, init, seed, a.iterations, a.workers)


if __name__ == "__main__":
    sys.exit(main())
