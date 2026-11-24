"""A network scored on a frozen bank (bv-v1/data/bank/<bank>.npz: held-out positions with independent many-draw values
of every legal option): where it plays something other than c2's choice (the bank's incumbent), what that is worth.

    python bank_eval.py --net models/s1/net.npy --bank sg_bank
Prints and returns: change rate, mean gain over c2's choice, harmful (< -0.01) and good (> +0.01) changes, agreement
with the bank's best option, overall and by decision kind and phase.
"""
import argparse
import json
import os

import numpy as np

from sg.paths import BV, use_rl3


def evaluate(net_path, bank):
    use_rl3()
    import torch
    from rl.net import from_flat
    B = dict(np.load(os.path.join(BV, "data", "bank", f"{bank}.npz")))
    net = from_flat(np.load(net_path)).eval()
    legal = torch.from_numpy(B["CAND"].astype(bool))
    with torch.no_grad():
        lg, _ = net(torch.from_numpy(B["X"]), legal)
    pick = lg.argmax(1).numpy()
    Q, inc = B["Q"], B["INC"]
    n = len(Q)
    gain = Q[np.arange(n), pick] - Q[np.arange(n), inc]
    ch = pick != inc
    best = np.nanargmax(np.where(B["CAND"].astype(bool), Q, -1), 1)
    turn = B["TURN"]
    phase = np.where(turn <= 4, "opening", np.where(turn <= 12, "middle", "late"))

    def block(m):
        if not m.any():
            return None
        return {"n": int(m.sum()), "changed": round(float(ch[m].mean()), 4), "mean_gain": round(float(gain[m].mean()), 5),
                "harmful": round(float((ch & (gain < -0.01))[m].mean()), 4),
                "good": round(float((ch & (gain > 0.01))[m].mean()), 4),
                "agrees_with_best": round(float((pick == best)[m].mean()), 4),
                "c2_agrees_with_best": round(float((inc == best)[m].mean()), 4)}

    kind = B["KIND"]
    return {"net": net_path, "bank": bank, "all": block(np.ones(n, bool)),
            "by_kind": {"buy": block(kind == 1), "action": block(kind == 0)},
            "by_phase": {p: block(phase == p) for p in ("opening", "middle", "late")}}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--net", required=True)
    ap.add_argument("--bank", default="sg_bank")
    a = ap.parse_args()
    print(json.dumps(evaluate(a.net, a.bank), indent=1))
