#!/usr/bin/env python3
"""Where do branch B's choices differ from branch A's, and do the differences help or hurt?
-> results/divergence/<model>/report.html + summary.json

    python experiments/analyze_games.py B_s2

Reads episodes.jsonl.gz (record_games.py): games of B against A on fresh analysis kingdoms, every decision of B with
its features and what A and frozen c2 would choose on the same position. Event = a divergence (B's choice differs
from A's). The six figures say where divergences happen and what announces them; two extra figures score them with
exact values on the held-out bank (every legal buy at 128-512 rollouts): the gain of B's buy over A's buy where the
two nets differ, by phase, opponent and kingdom card, and how a game's divergences relate to its result.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)
from far import Analysis, Signal, figures, load, style  # noqa: E402
from sg.paths import BV, RL3, use_rl3  # noqa: E402

style.apply()
HORIZON, WINDOW = 3, 40
KIND_NAMES = ["play action", "buy"]


def signals(f, ep):
    n = ep["n"]
    noise = np.random.default_rng(ep["outcome"]["seed"] + ep["outcome"]["seat"]).normal(size=n)
    return {
        "coins": Signal(f["coins"], "input", "coins"), "buys": Signal(f["buys"], "input", "buys left"),
        "turn": Signal(f["turn"], "input", "turn"), "vp_diff": Signal(f["vp_diff"], "input", "VP lead (scaled)"),
        "provinces": Signal(f["provinces"], "input", "Provinces left (scaled)"),
        "empty": Signal(f["empty"], "input", "empty piles (scaled)"), "legal": Signal(f["legal"], "input", "legal options"),
        "is_buy": Signal((f["kind"] == 1).astype(float), "input", "a buy decision"),
        "is_action": Signal((f["kind"] == 0).astype(float), "input", "an action decision"),
        "conf": Signal(f["b_conf"], "candidate", "B's margin between its top two options"),
        "value": Signal(f["b_value"], "candidate", "B's win estimate"),
        "c2_moves": Signal((f["a"] != f["c2"]).astype(float), "candidate", "A differs from c2 here"),
        "noise": Signal(noise, "candidate", "random noise (control)"),
    }


def events(ep):
    f = ep["f"]
    return list(np.where(f["b"] != f["a"])[0])


def draw_episode(ax, ep):
    f = ep["f"]
    ax.plot(f["turn"], np.cumsum(f["b"] != f["a"]), color=style.BLUE, lw=1.5)
    ax.set_xticks([]); ax.set_yticks([])


def episode_title(ep):
    o = ep["outcome"]
    return f"{o['cause']} {o['score']:+d}", (style.RED if o["failed"] else style.INK)


def draw_state(ax, ep, i):
    f = ep["f"]
    lo = max(0, i - 30)
    ax.step(range(lo, min(ep["n"], i + 21)), (f["b"] != f["a"])[lo:i + 21], color=style.BLUE, where="mid")
    ax.axvline(i, color=style.INK, lw=1)
    ax.set_title(f"turn {int(f["turn"][i])}, {KIND_NAMES[int(f['kind'][i])] if f['kind'][i] < 2 else 'subdecision'}", fontsize=9)
    ax.set_yticks([0, 1]); ax.set_yticklabels(["same", "differs"])


def bank_divergence(pairs):
    """Exact values on the bank: for each (B, A) pair of nets, the gain of B's buy over A's where they differ."""
    use_rl3()
    import torch
    from rl.net import from_flat
    B = dict(np.load(os.path.join(BV, "data", "bank", "sg_bank.npz")))
    X, L, Q = torch.from_numpy(B["X"]), torch.from_numpy(B["CAND"].astype(bool)), B["Q"]
    n = len(Q)
    turn, opp = B["TURN"], B["OPP"]
    cards = [json.loads(str(k))["key"]["kingdom"] for k in B["KEY"]]
    rows = []
    for b, a in pairs:
        with torch.no_grad():
            pb = from_flat(np.load(os.path.join(RL3, "runs", b, "net_100.npy")))(X, L)[0].argmax(1).numpy()
            pa = from_flat(np.load(os.path.join(RL3, "runs", a, "net_100.npy")))(X, L)[0].argmax(1).numpy()
        d = pb != pa
        g = Q[np.arange(n), pb] - Q[np.arange(n), pa]
        for i in np.where(d)[0]:
            rows.append({"pair": f"{b} vs {a}", "gain": float(g[i]), "turn": int(turn[i]), "opp": int(opp[i]),
                         "cards": cards[i]})
    return rows, n


def bars(labels, values, counts, title, xlabel):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 1.2 + 0.36 * len(labels)))
    ys = np.arange(len(labels))[::-1]
    cols = [style.BLUE if v >= 0 else style.ORANGE for v in values]
    ax.barh(ys, values, color=cols, height=0.7)
    ax.axvline(0, color=style.INK, lw=1)
    ax.set_yticks(ys); ax.set_yticklabels([f"{l} (n={c})" for l, c in zip(labels, counts)])
    for y, v in zip(ys, values):
        ax.text(v, y, f" {100 * v:+.1f}", va="center", ha="left" if v >= 0 else "right", fontsize=9)
    ax.set_xlabel(xlabel); ax.set_title(title)
    fig.tight_layout()
    return style.png(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--open", action="store_true")
    a = ap.parse_args()
    out = os.path.join(ROOT, "results", "divergence", a.model)
    header, episodes = load(os.path.join(out, "episodes.jsonl.gz"))
    an = Analysis(header, episodes, out_dir=out, event_name="divergence", horizon=HORIZON, window=WINDOW, fps=None,
                  warmup=0)
    an.signals(signals).events(events).compute()
    an.cause("On a buy", lambda ep, i: ep["f"]["kind"][i] == 1, lambda ep: ep["f"]["kind"] == 1)
    an.cause("On an action decision", lambda ep, i: ep["f"]["kind"][i] == 0, lambda ep: ep["f"]["kind"] == 0)
    an.cause("B keeps c2's choice (A moved away)", lambda ep, i: ep["f"]["b"][i] == ep["f"]["c2"][i], None)
    an.cause("Late game (turn > 12)", lambda ep, i: ep["f"]["turn"][i] > 12, lambda ep: ep["f"]["turn"] > 12)
    an.figure("gallery", figures.gallery(episodes, draw_episode, episode_title,
                                         key=lambda ep: (not ep["outcome"]["failed"], -int((ep["f"]["b"] != ep["f"]["a"]).sum()))))
    an.figure("start", figures.start(an.examples(3, min_lead=5), draw_state, offsets=(-10, -3, 0, 5)))
    kinds = ["play action", "buy", "subdecisions"]
    kof = lambda k: np.minimum(k, 2)  # noqa: E731
    ev = np.array([sum(((kof(ep["f"]["kind"]) == j) & (ep["f"]["b"] != ep["f"]["a"])).sum() for ep in episodes) for j in range(3)])
    hc = np.array([sum((kof(ep["f"]["kind"]) == j).sum() for ep in episodes) for j in range(3)])
    an.figure("where", figures.where_bars(kinds, ev, hc, "decision kind", ylabel="divergences per 1000\ndecisions"))

    # help or hurt: exact bank values
    rows, nbank = bank_divergence([("B_s1", "A_s1"), ("B_s2", "A_s2")])
    g = np.array([r["gain"] for r in rows])
    ph = np.array(["opening" if r["turn"] <= 4 else "middle" if r["turn"] <= 12 else "late" for r in rows])
    op = np.array([["c2", "c1", "RL-v1", "GA2"][r["opp"]] for r in rows])
    lab, val, cnt = [], [], []
    for name, m in [("all", np.ones(len(g), bool))] + [(p, ph == p) for p in ("opening", "middle", "late")] + \
                   [(f"vs {o}", op == o) for o in ("c2", "c1", "RL-v1", "GA2")]:
        if m.any():
            lab.append(name); val.append(float(g[m].mean())); cnt.append(int(m.sum()))
    an.figure("bank", bars(lab, val, cnt, "Exact gain of B's buy over A's where they differ (held-out bank)",
                           "mean gain, percentage points of win chance"),
              "On the 1,000 held-out bank positions, every legal buy has a 128-512 rollout value. For both seeds, "
              "where B's and A's top buys differ, the bar is the mean value of B's buy minus A's. Blue: B's choice is "
              "better on average; orange: worse.", title="Do B's differences help? (by phase and opponent)")
    cl = {}
    for r in rows:
        for c in r["cards"]:
            cl.setdefault(c, []).append(r["gain"])
    cs = sorted([(c, float(np.mean(v)), len(v)) for c, v in cl.items() if len(v) >= 15], key=lambda t: t[1])
    if cs:
        an.figure("cards", bars([c for c, _, _ in cs], [v for _, v, _ in cs], [n for _, _, n in cs],
                                "Gain of B's buy over A's where they differ, by kingdom card", "mean gain, points"),
                  "The same differences grouped by the kingdom cards present (15+ differing positions). Few positions "
                  "per card: read the sign of large bars only.", title="By kingdom card")
    nd = np.array([int((ep["f"]["b"] != ep["f"]["a"]).sum()) for ep in episodes])
    lost = np.array([ep["outcome"]["failed"] for ep in episodes])
    q = np.quantile(nd, [0.25, 0.5, 0.75])
    grp = np.digitize(nd, q)
    names = ["fewest", "fewer", "more", "most"]
    an.figure("games", bars([f"{names[j]} divergences ({nd[grp == j].min()}-{nd[grp == j].max()})" for j in range(4) if (grp == j).any()],
                            [float(lost[grp == j].mean()) for j in range(4) if (grp == j).any()],
                            [int((grp == j).sum()) for j in range(4) if (grp == j).any()],
                            "B's loss rate by how often it diverged from A in the game", "share of games B lost"),
              "Games split into quarters by the number of B's divergences. A rising bar would mean B's differences "
              "cost games; but long games have more decisions, so read it together with the turn signal.",
              title="Divergences and results")
    an.summary["bank"] = {"positions": nbank, "differences": len(rows), "mean_gain": float(g.mean()),
                          "helps": float((g > 0.01).mean()), "hurts": float((g < -0.01).mean()),
                          "by_phase": {p: float(g[ph == p].mean()) for p in ("opening", "middle", "late") if (ph == p).any()},
                          "by_opponent": {o: float(g[op == o].mean()) for o in ("c2", "c1", "RL-v1", "GA2") if (op == o).any()},
                          "worst_cards": cs[:5], "best_cards": cs[-5:]}
    n_ev = len(an.event_list)
    steps = sum(ep["n"] for ep in episodes)
    an.tiles([(f"{sum(lost)}/{len(episodes)}", "games B lost"), (f"{n_ev}", "divergences from A"),
              (f"{100 * n_ev / steps:.1f}%", "of B's decisions"), (f"{100 * g.mean():+.2f}", "bank gain per difference (points)")])
    an.write(f"Divergence analysis: {a.model} against {header['vs']}",
             f"{len(episodes)} games ({len(episodes) // 2} fresh analysis kingdoms, both seats), {steps} decisions of B; "
             f"divergence = B's arg-max differs from A's on the same position. Help or hurt: exact values on the "
             f"held-out bank ({nbank} buy positions, both seeds).", open_report=a.open)


if __name__ == "__main__":
    main()
