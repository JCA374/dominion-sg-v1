"""Step 1: the matched teacher ablation. Four arms on one frozen game manifest (manifests/ablation.json, both seats)
and one frozen teacher build (build/bv_ablation.so, copied once from bv-v1's bv.so; its sha is in every match
manifest): the plain c2 mirror, the buy teacher, the action teacher and both, all on c2 against plain c2, with the
handoff's recorded settings --k0 32 --k1 32 --topn 5 --margin 0.02 --z 2.0.

    python -u ablation.py [--workers 7] [--arms mirror,buy,action,both]
Out: results/ablation/match_sg_abl_<arm>.json(l), then ablation_report.py -> results/ablation.json
"""
import argparse
import os
import shutil
import subprocess
import sys

from sg.paths import BV, MANIFESTS, PY, RESULTS, ROOT, use_bv

ARMS = {"mirror": ["--bv", "0", "--act", "0"], "buy": ["--bv", "1", "--act", "0"],
        "action": ["--bv", "1", "--act", "2"], "both": ["--bv", "1", "--act", "1"]}
SETTINGS = ["--k0", "32", "--k1", "32", "--topn", "5", "--margin", "0.02", "--z", "2.0"]
LIB = os.path.join(ROOT, "build", "bv_ablation.so")


def frozen_lib():
    if not os.path.exists(LIB):
        use_bv()
        from bv import core                              # builds bv-v1/bv.so from the pinned sources
        os.makedirs(os.path.dirname(LIB), exist_ok=True)
        shutil.copy2(core.SO, LIB)
    return LIB


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=7)
    ap.add_argument("--arms", default="mirror,buy,action,both")
    a = ap.parse_args()
    lib = frozen_lib()
    out = os.path.join(RESULTS, "ablation")
    os.makedirs(out, exist_ok=True)
    man = os.path.join(MANIFESTS, "ablation.json")
    for arm in a.arms.split(","):
        name = f"sg_abl_{arm}"
        for ext in (".json", ".jsonl"):                  # resume from our copy if the submodule's was cleaned
            src, dst = os.path.join(out, f"match_{name}{ext}"), os.path.join(BV, "results", f"match_{name}{ext}")
            if os.path.exists(src) and not os.path.exists(dst):
                shutil.copy2(src, dst)
        cmd = [PY, "-u", "match.py", "--name", name, "--base", "c2", "--opp", "c2", *ARMS[arm], *SETTINGS,
               "--games", man, "--kingdoms", "300", "--workers", str(a.workers), "--resume"]
        print(" ".join(cmd[2:]), flush=True)
        subprocess.run(cmd, cwd=BV, env={**os.environ, "BV_SO": lib}, check=True)
        for ext in (".json", ".jsonl"):
            shutil.copy2(os.path.join(BV, "results", f"match_{name}{ext}"), os.path.join(out, f"match_{name}{ext}"))
    subprocess.run([PY, os.path.join(ROOT, "ablation_report.py")], cwd=ROOT, check=True)


if __name__ == "__main__":
    sys.exit(main())
