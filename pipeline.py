"""The pipeline driver (the site shows it as run "pipeline"): waits for the running step, then goes on by the
predeclared rules, logging each decision to the journal.

    python -u pipeline.py --run pipeline

1. waits for the ablation (unit sg-ablation) and writes its report;
2. chooses the pilot's labels by the handoff's rule "start collection with the simplest teacher arm that reproduces a
   gain": buy-only if the buy arm's interval is above 50%; else action-only if the action arm's is; else both if the
   combined arm's is; else stop (no arm reproduces a gain);
3. runs pilot.py (labels, bank, survival). The supervised step's settings are chosen on the dev split afterwards.
"""
import argparse
import json
import os
import subprocess
import time

from sg.paths import PY, RESULTS, ROOT


def journal(*args):
    subprocess.run([PY, os.path.join(ROOT, "experiments", "journal.py"), *args], check=True, capture_output=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="pipeline")
    ap.add_argument("--workers", type=int, default=7)
    a = ap.parse_args()
    while subprocess.run(["systemctl", "--user", "is-active", "--quiet", "sg-ablation"]).returncode == 0:
        time.sleep(60)
    subprocess.run([PY, "ablation_report.py"], cwd=ROOT, check=True)
    r = json.load(open(os.path.join(RESULTS, "ablation.json")))
    arms = r["arms"]
    act = None
    for arm, code in (("buy", 0), ("action", 2), ("both", 1)):
        if arm in arms and arms[arm]["ci95"][0] > 0.5:
            act = code
            break
    txt = ", ".join(f"{k} {100 * v['score']:.1f}% [{100 * v['ci95'][0]:.1f}, {100 * v['ci95'][1]:.1f}]" for k, v in arms.items())
    if act is None:
        journal("--decision", "No teacher arm reproduces a gain", f"Ablation: {txt}. The pilot is not collected.")
        return
    name = {0: "buy-only", 2: "action-only", 1: "buys and actions"}[act]
    journal("--decision", f"Pilot labels: {name}", f"The simplest arm whose paired interval is above 50% (the "
            f"handoff's rule). Ablation: {txt}. Paired differences: " +
            "; ".join(f"{k} {100 * v['diff']:+.1f} [{100 * v['ci95'][0]:+.1f}, {100 * v['ci95'][1]:+.1f}]"
                      for k, v in r["paired"].items()), "the other arms, as the first pilot")
    subprocess.run([PY, "-u", "pilot.py", "--act", str(act), "--states", "20000", "--workers", str(a.workers)],
                   cwd=ROOT, check=True)
    journal("Pilot done", "Labels, the held-out bank and the survival check are in results/ (pilot.json, sg_bank.json, "
            "survive_sg_bank.json). Next: the supervised step, tuned on the dev split.")


if __name__ == "__main__":
    main()
