"""The site's view of the pipeline (called by site_hooks before every build): runs/pipeline/{curriculum.json,
events.jsonl, stats.csv} from the result files, and runs/<A|B>_<seed> linked to the PPO runs in rl-v3/runs/.

events.jsonl is rebuilt from the files each time (deterministic); stats.csv gets one progress row per build (at most
one per 10 minutes): games played per ablation arm and their running scores, labels collected, PPO iterations.
"""
import csv
import glob
import json
import os
import time

import numpy as np

from sg.paths import BV, RESULTS, RL3, ROOT

D = os.path.join(ROOT, "runs", "pipeline")
ARMS = ["mirror", "buy", "action", "both"]
STEPS = [
    {"name": "1 Teacher ablation", "world": "c2 + teacher vs plain c2 on 300 fresh kingdoms x 2 seats, one frozen build: "
     "mirror, buy teacher, action teacher, both (32 draws, top-5 buys, margin 0.02, z 2)",
     "pass": "all four arms played; the simplest arm whose paired interval is above 50% chooses what the pilot labels",
     "expect": "buy search repeats its earlier gain (59%); action search adds something on top (earlier +4 points, "
               "interval -2.5 to +10.5)"},
    {"name": "2 Pilot dataset", "world": "c2 plays, the teacher labels (fixed 32 draws) and validates each graded winner "
     "on 32 independent draws; random kingdoms + attack / Gardens / trasher / engine slices; opponents c2, c1, RL-v1, GA2",
     "pass": "20,000 positions of one cohort; a frozen held-out bank (1,000 positions, all legal options, 128 draws, 100 "
             "at 512) replays exactly; survival under the public sampler and an opponent mixture measured",
     "expect": "a few percent of positions are verified improvements; most survive the public view"},
    {"name": "3 Supervised c2", "world": "c2 itself as the student: a masked pairwise loss raises each verified winner "
     "above c2's choice, weighted by the verified gain; KL to frozen c2 and its value on c2's own positions as anchors",
     "pass": "held-out pairs move toward the winner with small anchor drift; the bank says the changes help; a dev match",
     "expect": "a small, targeted change: anchor argmax agreement stays above 99%"},
    {"name": "4 Matched PPO", "world": "A from plain c2, B from supervised c2; the same open-league PPO, game budget and "
     "training stream per seed (two seeds); frozen c2 and c1 in the league",
     "pass": "promotion: >= 53% vs frozen c2 on 2,000 fresh kingdoms, paired lower bound > 50%, no regression vs GA2",
     "expect": "B starts ahead of A; the question is whether it stays ahead after the same PPO"},
]


def mtime_t(p):
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(os.path.getmtime(p)))


def jl(p):
    return [json.loads(x) for x in open(p)] if os.path.exists(p) else []


def paired(rows):
    by = {}
    for r in rows:
        by.setdefault(r["i"], {})[r["seat"]] = r["points"]
    pts = [np.mean(list(v.values())) for v in by.values() if len(v) == 2]
    return len(pts), (float(np.mean(pts)) if pts else None)


def arm_rows(arm):
    for p in (os.path.join(BV, "results", f"match_sg_abl_{arm}.jsonl"),
              os.path.join(RESULTS, "ablation", f"match_sg_abl_{arm}.jsonl")):
        if os.path.exists(p):
            return jl(p), p
    return [], None


def link_ppo_runs():
    for d in glob.glob(os.path.join(RL3, "runs", "[AB]*_s*")):
        dst = os.path.join(ROOT, "runs", os.path.basename(d))
        if not os.path.exists(dst):
            os.symlink(os.path.relpath(d, os.path.dirname(dst)), dst)


def events():
    ev = []
    ex = os.path.join(ROOT, "manifests", "exclusions.json")
    if os.path.exists(ex):
        ev.append({"t": mtime_t(ex), "kind": "run_start", "note": "Frozen manifests written (ablation 300, dev 300, "
                   "matrix 500, promotion 2,000 fresh kingdoms) and the shared exclusion list (8,197 kingdoms)."})
    lib = os.path.join(ROOT, "build", "bv_ablation.so")
    if os.path.exists(lib):
        ev.append({"t": mtime_t(lib), "kind": "stage_start", "stage": STEPS[0]["name"], "expect": STEPS[0]["expect"]})
    for arm in ARMS:
        p = os.path.join(RESULTS, "ablation", f"match_sg_abl_{arm}.json")
        if os.path.exists(p):
            s = json.load(open(p))
            if s.get("done"):
                ev.append({"t": s["t"].replace(" ", "T") + ":00", "kind": "note",
                           "title": f"Ablation arm done: {arm} {100 * s['score']:.1f}%",
                           "text": f"{s['games']} games, 95% interval {100 * s['ci95'][0]:.1f}–{100 * s['ci95'][1]:.1f}%; "
                                   f"buy overrides {100 * s.get('override_rate', 0):.1f}% of buys, action overrides "
                                   f"{s.get('act_overrides', 0)} of {s.get('act_graded', 0)} graded; capped {s['capped']}, "
                                   f"errors {s['errors']}; latency {s.get('latency_s')}."})
    rep = os.path.join(RESULTS, "ablation.json")
    if os.path.exists(rep):
        r = json.load(open(rep))
        if len(r["arms"]) == 4:
            txt = "; ".join(f"{k} {100 * v['diff']:+.1f} points [{100 * v['ci95'][0]:+.1f}, {100 * v['ci95'][1]:+.1f}]"
                            for k, v in r["paired"].items())
            ev.append({"t": mtime_t(rep), "kind": "note", "title": "Ablation: paired differences", "text": txt})
            b = r["arms"]["both"]
            ev.append({"t": mtime_t(rep), "kind": "stage_pass", "stage": STEPS[0]["name"],
                       "check": {"score": b["score"], "n": b["games"], "label": "c2 + buy and action teacher vs c2"}})
            ev.append({"t": mtime_t(rep), "kind": "stage_start", "stage": STEPS[1]["name"], "expect": STEPS[1]["expect"]})
    for name, title in (("pilot.json", "Pilot labels collected"), ("sg_bank.json", "Held-out bank built"),
                        ("survive_sg_bank.json", "Survival check done")):
        p = os.path.join(RESULTS, name)
        if os.path.exists(p):
            ev.append({"t": mtime_t(p), "kind": "note", "title": title, "text": json.dumps(json.load(open(p)))[:600]})
    sup = sorted(glob.glob(os.path.join(RESULTS, "supervise_*.json")), key=os.path.getmtime)
    if sup:
        if os.path.exists(os.path.join(RESULTS, "survive_sg_bank.json")):
            ev.append({"t": mtime_t(os.path.join(RESULTS, "survive_sg_bank.json")), "kind": "stage_pass",
                       "stage": STEPS[1]["name"]})
        ev.append({"t": mtime_t(sup[0]), "kind": "stage_start", "stage": STEPS[2]["name"], "expect": STEPS[2]["expect"]})
    for p in sup:
        s = json.load(open(p))
        e = s["epochs"][-1]
        ev.append({"t": mtime_t(p), "kind": "note", "title": f"Supervised c2: {os.path.basename(p)[10:-5]}",
                   "text": f"verified pairs {s['verified']}; held-out pair accuracy {e['test']}; anchor drift "
                           f"{e['anchor_test']['all']}"})
    plan = os.path.join(RESULTS, "ppo_plan.json")
    if os.path.exists(plan):
        ev.append({"t": mtime_t(plan), "kind": "stage_pass", "stage": STEPS[2]["name"]})
        ev.append({"t": mtime_t(plan), "kind": "stage_start", "stage": STEPS[3]["name"], "expect": STEPS[3]["expect"]})
    for p in sorted(glob.glob(os.path.join(RESULTS, "goal_*.json"))):
        pass                                          # goal checks are read by publish.py itself
    fin = os.path.join(RESULTS, "decision.json")
    if os.path.exists(fin):
        r = json.load(open(fin))
        ev.append({"t": mtime_t(fin), "kind": "stage_pass" if r.get("promoted") else "note",
                   "stage": STEPS[3]["name"], "title": "Decision", "text": r.get("summary", "")})
    for p in sorted(glob.glob(os.path.join(ROOT, "logs", "*.log"))):
        pass
    ev.sort(key=lambda e: e["t"])
    return ev


def current_stage():
    st = [e["stage"] for e in events() if e["kind"] == "stage_start"]
    return st[-1] if st else ""


def progress_row():
    row = {"t": time.strftime("%Y-%m-%dT%H:%M:%S"), "stage": current_stage()}
    t0p = os.path.join(ROOT, "manifests", "exclusions.json")
    row["hours"] = round((time.time() - os.path.getmtime(t0p)) / 3600, 3) if os.path.exists(t0p) else 0
    for arm in ARMS:
        n, s = paired(arm_rows(arm)[0])
        row[f"abl_{arm}_pairs"] = n
        row[f"abl_{arm}_score"] = round(s, 4) if s is not None else ""
    lab = 0
    for f in glob.glob(os.path.join(BV, "data", "sg_pilot", "*.npz")):
        try:
            lab += len(np.load(f)["K"])
        except Exception:
            pass
    row["pilot_labels"] = lab
    its = 0
    for d in glob.glob(os.path.join(RL3, "runs", "[AB]*_s*", "stats.csv")):
        its += max(0, sum(1 for _ in open(d)) - 2)
    row["ppo_iterations"] = its
    return row


def update():
    os.makedirs(D, exist_ok=True)
    json.dump({"note": "The handoff's run order: ablation -> pilot labels -> supervised c2 -> matched PPO (A vs B).",
               "stages": [{**s, "reward": "match score: win 1, tie 1/2, loss 0", "cap": "none"} for s in STEPS]},
              open(os.path.join(D, "curriculum.json"), "w"), indent=1)
    with open(os.path.join(D, "events.jsonl"), "w") as f:
        for e in events():
            f.write(json.dumps(e) + "\n")
    p = os.path.join(D, "stats.csv")
    rows = list(csv.DictReader(open(p))) if os.path.exists(p) else []
    if not rows or time.time() - time.mktime(time.strptime(rows[-1]["t"], "%Y-%m-%dT%H:%M:%S")) > 600:
        row = progress_row()
        fields = list(row)
        new = not rows
        with open(p, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields if new else list(rows[0]), extrasaction="ignore")
            if new:
                w.writeheader()
            w.writerow(row)
    link_ppo_runs()


if __name__ == "__main__":
    update()
