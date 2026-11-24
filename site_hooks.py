"""Project hooks for publish.py (ml-project-site): the pipeline view (track.py), the Results page's data, the Play tab's
game clips (rl-v3's renderer: one frame per buy, both decks, what each net wanted) and pictures for The model tab.
Drawing lives here only; no policy imports this file."""
import glob
import importlib.util
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
from sg.paths import BV, C2, C1, RESULTS, RL3, use_rl3  # noqa: E402

try:                                          # the pipeline's run files, before publish.py reads runs/
    import track
    track.update()
except Exception as ex:                        # never stop the publisher
    print("track.update failed:", ex)

use_rl3()
_spec = importlib.util.spec_from_file_location("rl3_site_hooks", os.path.join(RL3, "site_hooks.py"))
RLH = importlib.util.module_from_spec(_spec)
try:
    _spec.loader.exec_module(RLH)
except Exception as ex:
    print("rl-v3 site hooks unavailable:", ex)
    RLH = None


def _newest(run, every=10):
    nets = []
    for p in glob.glob(os.path.join(RL3, "runs", run, "net_*.npy")):
        it = int(p.split("_")[-1][:-4])
        if it % every == 0 and it > 0:
            nets.append((it, p))
    return max(nets) if nets else None


def models():
    return []                                  # no game clips (the Play tab is off)


def games():
    return []


def cached(out_png, source_path):
    return os.path.exists(out_png) and os.path.exists(source_path) and os.path.getmtime(out_png) >= os.path.getmtime(source_path)


def model_viz(out_dir):
    """The net's architecture and weights (c2; every model here has its shape), and for the supervised c2 what changed."""
    if not RLH:
        return []
    import torch
    from rl.net import from_flat
    figs = []
    arch, hist = os.path.join(out_dir, "arch.png"), os.path.join(out_dir, "weights.png")
    if not (cached(arch, C2) and cached(hist, C2)):
        net = from_flat(np.load(C2))
        sd = net.state_dict()
        RLH.fig_arch([("input: what c2 knows at a decision", net.nf), ("dense ReLU", net.nh), ("dense ReLU", net.nh),
                      ("34 logits + 1 win estimate", 35)], arch, title="RL-v3 c2 (and every net in this project)")
        RLH.fig_weight_hist([("W1", sd["l1.weight"].numpy()), ("W2", sd["l2.weight"].numpy()),
                             ("policy", sd["pi.weight"].numpy()), ("value", sd["v.weight"].numpy())], hist,
                            title="c2's weights per layer")
    figs += [{"src": "model/arch.png", "title": "Architecture", "caption": "One trunk, two heads: 34 logits (a card, "
              "or pass) for whatever decision is asked, and a win estimate. 181k parameters. The supervised step and "
              "PPO change these weights; the shape never changes."},
             {"src": "model/weights.png", "title": "c2's weights", "caption": "A wide, centred histogram means the "
              "layer is used; a spike at 0 means dead units."}]
    sup = sorted(glob.glob(os.path.join(ROOT, "models", "s*", "net.npy")), key=os.path.getmtime)
    if sup:
        dpng = os.path.join(out_dir, "delta.png")
        if not cached(dpng, sup[-1]):
            a, b = from_flat(np.load(C2)).state_dict(), from_flat(np.load(sup[-1])).state_dict()
            names = [("W1", "l1.weight"), ("W2", "l2.weight"), ("policy", "pi.weight"), ("value", "v.weight")]
            RLH.fig_bars([n for n, _ in names], [float((b[k] - a[k]).norm() / a[k].norm()) for _, k in names], dpng,
                         title="How much the supervised step changed each layer", ylabel="relative change |Δw| / |w|",
                         fmt="{:.4f}")
        figs.append({"src": "model/delta.png", "title": "What the supervised step changed",
                     "caption": "Relative size of the weight change per layer, supervised c2 against c2. The anchors "
                                "keep it small; the value head should barely move."})
    return figs


def _load(p):
    try:
        return json.load(open(p))
    except Exception:
        return None


def extra(data_dir):
    """data/sg.json for the Results page: every result file, summarised."""
    out = {"ablation": _load(os.path.join(RESULTS, "ablation.json")), "pilot": _load(os.path.join(RESULTS, "pilot.json")),
           "bank": _load(os.path.join(RESULTS, "sg_bank.json")), "survive": _load(os.path.join(RESULTS, "survive_sg_bank.json")),
           "decision": _load(os.path.join(RESULTS, "decision.json")), "supervise": {}, "matches": {}, "ppo": {}}
    live = {}
    for arm in track.ARMS:
        rows, _ = track.arm_rows(arm)
        n, s = track.paired(rows)
        live[arm] = {"pairs": n, "score": s}
    out["ablation_live"] = live
    for p in sorted(glob.glob(os.path.join(RESULTS, "supervise_*.json"))):
        s = _load(p)
        if s:
            out["supervise"][os.path.basename(p)[10:-5]] = {"args": s["args"], "verified": s["verified"],
                                                            "first": s["epochs"][0], "last": s["epochs"][-1],
                                                            "bank": s.get("bank"), "dev_match": s.get("dev_match")}
    for p in sorted(glob.glob(os.path.join(RESULTS, "matches", "match_*.json"))):
        s = _load(p)
        if s:
            out["matches"][os.path.basename(p)[6:-5]] = {k: s.get(k) for k in ("games", "pairs", "score", "ci95", "capped",
                                                                                 "errors", "done", "t")}
    import csv
    for d in sorted(glob.glob(os.path.join(RL3, "runs", "[AB]*_s*"))):
        p = os.path.join(d, "stats.csv")
        if os.path.exists(p):
            rows = list(csv.DictReader(open(p)))
            if rows:
                last = rows[-1]
                out["ppo"][os.path.basename(d)] = {k: last.get(k) for k in ("iter", "eval_vs_c2", "eval_vs_c1", "eval_GA_base",
                                                                            "eval_champion", "games_total", "cpu_hours")}
    json.dump(out, open(os.path.join(data_dir, "sg.json"), "w"), indent=0, default=str)
