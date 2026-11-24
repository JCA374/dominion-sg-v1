"""Step 4's tests and the decision, after the four PPO runs (A_s1, B_s1, A_s2, B_s2 at their last iteration).

    python -u final.py --run pipeline [--iteration 100]

1. Matrix (manifests/matrix.json, 500 fresh kingdoms x 2 seats): each run's last net against frozen c2, c1, RL-v1 r2
   2200 and GA2, and A against B within each seed; plus plain c2 against GA2 (the regression baseline).
2. Candidate per branch: the seed with the higher matrix score against c2.
3. Promotion (manifests/promotion.json, 2,000 fresh kingdoms x 2 seats, played once): each branch's candidate against
   frozen c2 and against GA2, and plain c2 against GA2. Gate: >= 53% vs c2 with the paired 95% lower bound above 50%;
   no regression vs GA2 (score not more than 1 point below plain c2's on the same games); no card slice (kingdoms
   holding a card, >= 150 kingdom pairs) more than 3 points below plain c2's vs GA2.
4. results/decision.json: per branch the scores, the gate, the card slices; A vs B; label cost; promoted model(s).
"""
import argparse
import json
import os
import subprocess

import numpy as np

from netmatch import play
from sg.paths import RESULTS, RL3, ROOT

OPPS = ["c2", "c1", "rl1", "ga2"]


def last_net(run, it):
    p = os.path.join(RL3, "runs", run, f"net_{it}.npy")
    if not os.path.exists(p):
        raise SystemExit(f"{run}: no net_{it}.npy")
    return p


def pairs_of(name):
    rows = [json.loads(x) for x in open(os.path.join(RESULTS, "matches", f"match_{name}.jsonl"))]
    by = {}
    for r in rows:
        by.setdefault(r["i"], {})[r["seat"]] = r
    return {i: v for i, v in by.items() if len(v) == 2}


def slices(name, base):
    """Per kingdom card: score of match `name` and of the baseline match `base` on the same kingdom pairs."""
    a, b = pairs_of(name), pairs_of(base)
    out = {}
    common = set(a) & set(b)
    cards = sorted({c for i in common for c in a[i][0]["kingdom"]})
    for c in cards:
        ids = [i for i in common if c in a[i][0]["kingdom"]]
        if len(ids) < 150:
            continue
        sa = np.mean([np.mean([r["points"] for r in a[i].values()]) for i in ids])
        sb = np.mean([np.mean([r["points"] for r in b[i].values()]) for i in ids])
        out[c] = {"pairs": len(ids), "score": round(float(sa), 4), "baseline": round(float(sb), 4),
                  "diff": round(float(sa - sb), 4)}
    return out


def journal(*args):
    subprocess.run([os.path.join(ROOT, ".venv", "bin", "python"), os.path.join(ROOT, "experiments", "journal.py"), *args],
                   check=True, capture_output=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="pipeline")
    ap.add_argument("--iteration", type=int, default=100)
    ap.add_argument("--seeds", default="s1,s2")
    ap.add_argument("--workers", type=int, default=7)
    a = ap.parse_args()
    seeds = a.seeds.split(",")
    nets = {f"{b}_{s}": last_net(f"{b}_{s}", a.iteration) for b in "AB" for s in seeds}
    mat = {}
    for run, net in nets.items():
        for o in OPPS:
            mat[f"{run} vs {o}"] = play(net, o, "matrix", f"mx_{run}_it{a.iteration}_vs_{o}", a.workers)
    for s in seeds:
        mat[f"B_{s} vs A_{s}"] = play(nets[f"B_{s}"], nets[f"A_{s}"], "matrix", f"mx_B_{s}_vs_A_{s}_it{a.iteration}", a.workers)
    mat["c2 vs ga2"] = play("c2", "ga2", "matrix", "mx_c2_vs_ga2", a.workers)
    summ = {k: {"score": v["score"], "ci95": v["ci95"], "pairs": v["pairs"]} for k, v in mat.items()}
    json.dump(summ, open(os.path.join(RESULTS, "matrix.json"), "w"), indent=1)
    journal("Matrix done", "; ".join(f"{k} {100 * v['score']:.1f}% [{100 * v['ci95'][0]:.1f}, {100 * v['ci95'][1]:.1f}]"
                                     for k, v in summ.items()))
    cand = {b: max(seeds, key=lambda s: summ[f"{b}_{s} vs c2"]["score"]) for b in "AB"}
    base = play("c2", "ga2", "promotion", "promo_c2_vs_ga2", a.workers)
    dec = {"iteration": a.iteration, "candidates": cand, "branches": {}, "matrix": summ,
           "baseline_c2_vs_ga2": {"score": base["score"], "ci95": base["ci95"]}}
    for b, s in cand.items():
        run = f"{b}_{s}"
        vc2 = play(nets[run], "c2", "promotion", f"promo_{run}_it{a.iteration}_vs_c2", a.workers)
        vga = play(nets[run], "ga2", "promotion", f"promo_{run}_it{a.iteration}_vs_ga2", a.workers)
        sl = slices(f"promo_{run}_it{a.iteration}_vs_ga2", "promo_c2_vs_ga2")
        worst = min(sl.items(), key=lambda kv: kv[1]["diff"]) if sl else None
        gate = {"vs_c2": vc2["score"] >= 0.53 and vc2["ci95"][0] > 0.5,
                "vs_ga2": vga["score"] >= base["score"] - 0.01,
                "card_slices": all(v["diff"] >= -0.03 for v in sl.values())}
        dec["branches"][b] = {"run": run, "net": os.path.relpath(nets[run], ROOT), "vs_c2": {"score": vc2["score"], "ci95": vc2["ci95"]},
                              "vs_ga2": {"score": vga["score"], "ci95": vga["ci95"]}, "worst_card_slice": worst,
                              "card_slices": sl, "gate": gate, "promoted": all(gate.values())}
    pa = os.path.join(RESULTS, "pilot_cost.json")
    dec["label_cost"] = json.load(open(pa)) if os.path.exists(pa) else None
    A, B = dec["branches"]["A"], dec["branches"]["B"]
    ab = [summ[f"B_{s} vs A_{s}"] for s in seeds]
    consistent = all(x["score"] > 0.5 for x in ab) and B["vs_c2"]["score"] > A["vs_c2"]["score"]
    dec["b_beats_a"] = consistent
    dec["promoted"] = [b for b in "AB" if dec["branches"][b]["promoted"]]
    dec["summary"] = (f"Promotion (2,000 fresh kingdoms x 2 seats) vs frozen c2: A ({A['run']}) {100 * A['vs_c2']['score']:.1f}% "
                      f"[{100 * A['vs_c2']['ci95'][0]:.1f}, {100 * A['vs_c2']['ci95'][1]:.1f}], B ({B['run']}) "
                      f"{100 * B['vs_c2']['score']:.1f}% [{100 * B['vs_c2']['ci95'][0]:.1f}, {100 * B['vs_c2']['ci95'][1]:.1f}]. "
                      f"B vs A head to head (matrix): " + ", ".join(f"{s} {100 * x['score']:.1f}%" for s, x in zip(seeds, ab)) +
                      f". Promoted: {', '.join(dec['promoted']) or 'none'}. B consistently better than A: {consistent}.")
    json.dump(dec, open(os.path.join(RESULTS, "decision.json"), "w"), indent=1)
    journal("--decision", "Promotion result", dec["summary"], "")
    print(dec["summary"])


if __name__ == "__main__":
    main()
