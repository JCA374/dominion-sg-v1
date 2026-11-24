"""The orchestrator: episodes + signals + events -> the frame table, the numbers, the figures, the report.

    from far import Analysis, Signal, load, style, figures
    style.apply()
    header, episodes = load("results/fall_analysis/champion/episodes.jsonl.gz")
    an = Analysis(header, episodes, out_dir="results/fall_analysis/champion",
                  event_name="fall", horizon=30, window=150, fps=50, warmup=20)
    an.signals(my_signals)                 # (f, ep) -> {key: Signal(values, "input"|"candidate", "Label")}
    an.events(lambda ep: [ep["n"] - 1] if ep["outcome"]["failed"] else [])     # the default
    an.compute()
    an.cause("Fell forwards", lambda ep, i: ep["f"]["angle"][i] > 0, lambda ep: ep["f"]["angle"] > 0)
    an.figure("gallery", figures.gallery(episodes, draw, title))               # the two project-drawn figures
    an.figure("start", figures.start(an.examples(3), draw_state))
    an.tiles([(f"{n_fell}/{len(episodes)}", "episodes that fell")])
    an.write("Fall analysis: champion", "60 episodes on fresh seeds; fall = hull touches the ground ...")

Vocabulary. event: the failure (an index into the episode's frames; terminal failures are the last frame).
healthy: frames where the model is still in control (the frame table; default: every frame before the event,
after `warmup`). normal: healthy frames further than `window` steps from any event (the baseline), plus the
recorded snapshots if any. label: "an event comes within `horizon` steps". Signals of kind "input" are what
the model already sees; "candidate" signals are the hypotheses.
"""
from __future__ import annotations

import json
import os
from collections import namedtuple

import numpy as np

from . import figures as F
from . import stats as ST
from .report import build, privacy_scan

Signal = namedtuple("Signal", "values kind label")

DEFAULT_SECTIONS = [
    ("gallery", "1. What does {event} look like?", "Every episode, worst first."),
    ("where", "2. Where does it happen?", "Event rate by context, against the healthy baseline."),
    ("drift", "3. What drifts before it?",
     "Median and IQR of each signal over the last {window} steps before each {event}; dashed line and grey band: "
     "the normal level (median, IQR) on healthy steps far from any {event}."),
    ("predicts", "4. What predicts it?",
     "Left: AUC of each signal alone for '{event} within {horizon} steps'; the dashed line is all the model's inputs "
     "together (a cross-validated logistic model). Right: what a signal adds beyond the inputs (partial correlation "
     "with log time-to-{event}, inputs regressed out). A bar on the right is new information."),
    ("causes", "5. What causes it?", "Share of {event} events with each property; black ticks show the same share among healthy steps."),
    ("start", "6. How does it start?", "A few random {event} events: the state before, at, and after the event."),
]


class Analysis:
    def __init__(self, header: dict, episodes: list, out_dir: str, event_name: str = "failure", horizon: int = 30,
                 window: int = 150, fps: float | None = None, warmup: int = 0, ttf_cap: int | None = None):
        self.header, self.episodes, self.out_dir = header, episodes, out_dir
        self.event_name, self.horizon, self.window, self.fps, self.warmup = event_name, horizon, window, fps, warmup
        self.ttf_cap = ttf_cap or max(4 * horizon, window)
        self._signals = None
        self._events = lambda ep: [ep["n"] - 1] if ep["outcome"].get("failed") else []
        self._healthy = None
        self._inputs = None
        self.causes_table = {}
        self.figs, self.captions = {}, {}
        self.extra_sections = []
        self.tile_list = []
        self.summary = {}
        self.rows = None

    # ---------------------------------------------------------------- plug-ins
    def signals(self, fn):
        """fn(f, ep) -> {key: Signal(values, kind, label)} where f is a dict of per-step arrays."""
        self._signals = fn
        return self

    def events(self, fn):
        """fn(ep) -> list of frame indices where the event happens (default: last frame if outcome.failed)."""
        self._events = fn
        return self

    def healthy(self, fn):
        """fn(ep) -> bool mask over frames the model is still in control of (default: all frames before an event)."""
        self._healthy = fn
        return self

    def inputs(self, fn):
        """fn(ep) -> 2-D array (steps x features) of the model's raw inputs, for the reference AUC and partial
        correlations (default: the signals of kind 'input')."""
        self._inputs = fn
        return self

    # ---------------------------------------------------------------- compute
    def compute(self):
        if self._signals is None:
            raise RuntimeError("call .signals(fn) first")
        eps = self.episodes
        for ep in eps:
            ep["sig"] = {k: Signal(np.asarray(v[0], dtype=float), v[1], v[2]) for k, v in self._signals(ep["f"], ep).items()}
            if "snap" in ep:
                ep["snap_sig"] = {k: np.asarray(v[0], dtype=float) for k, v in self._signals(ep["snap"], ep).items()}
            ep["events"] = [int(e) for e in self._events(ep)]
            n = ep["n"]
            ep["tte"] = ST.time_to_event(n, ep["events"])
            base = (np.arange(n) >= self.warmup) & (ep["tte"] > 0)
            ep["healthy"] = base & (np.asarray(self._healthy(ep), bool) if self._healthy else base)
        self.meta = {k: Signal(None, s.kind, s.label) for k, s in eps[0]["sig"].items()}
        self.keys = list(self.meta)
        self.input_keys = [k for k in self.keys if self.meta[k].kind == "input"]

        X, label, ttf, groups, Z = [], [], [], [], []
        for g, ep in enumerate(eps):
            ok = ep["healthy"]
            X.append(np.c_[[ep["sig"][k].values[ok] for k in self.keys]].T if ok.any() else np.zeros((0, len(self.keys))))
            label.append(ep["tte"][ok] <= self.horizon)
            ttf.append(np.log(np.minimum(ep["tte"][ok], self.ttf_cap) + 1))
            groups.append(np.full(int(ok.sum()), g))
            Zi = np.asarray(self._inputs(ep), float)[ok] if self._inputs else np.c_[[ep["sig"][k].values[ok] for k in self.input_keys]].T
            Z.append(Zi if Zi.ndim == 2 else Zi.reshape(int(ok.sum()), -1))
        self.X = np.nan_to_num(np.vstack(X))
        self.label, self.ttf, self.groups = np.concatenate(label), np.concatenate(ttf), np.concatenate(groups)
        self.Z = np.nan_to_num(np.vstack(Z)) if Z and Z[0].shape[1] else np.zeros((len(self.label), 0))

        # normal baseline: healthy frames far from any event, plus snapshots
        far = np.concatenate([(ep["tte"][ep["healthy"]] > self.window) for ep in eps])
        normal_vals = {}
        for j, k in enumerate(self.keys):
            v = self.X[far, j] if far.any() else self.X[:, j]
            if any("snap_sig" in ep for ep in eps):
                v = np.concatenate([v] + [ep["snap_sig"][k] for ep in eps if "snap_sig" in ep])
            normal_vals[k] = v
        self.normal = {k: (float(np.median(v)), float(np.percentile(v, 25)), float(np.percentile(v, 75))) if len(v)
                       else (float("nan"),) * 3 for k, v in normal_vals.items()}

        # run-up windows (NaN-padded when the event comes early)
        self.event_list = [(ep, i) for ep in eps for i in ep["events"]]
        W = {k: np.full((len(self.event_list), self.window + 1), np.nan) for k in self.keys}
        for r, (ep, i) in enumerate(self.event_list):
            lo = max(0, i - self.window)
            for k in self.keys:
                seg = ep["sig"][k].values[lo:i + 1]
                W[k][r, self.window + 1 - len(seg):] = seg
        self.windows = W

        # the numbers
        has_pos = self.label.any() and (~self.label).any()
        self.ref_auc = ST.logistic_auc(self.Z, self.label.astype(float), self.groups) if has_pos and self.Z.shape[1] else float("nan")
        self.rows = []
        for j, k in enumerate(self.keys):
            a = ST.auc(self.X[:, j], self.label) if has_pos else float("nan")
            pr = ST.partial_corr(self.X[:, j], self.ttf, self.Z) if len(self.label) > self.Z.shape[1] + 2 else float("nan")
            at = float(np.nanmedian(W[k][:, -1])) if self.event_list else None
            self.rows.append({"key": k, "label": self.meta[k].label, "kind": self.meta[k].kind,
                              "auc": _r(a), "auc_dir": _r(max(a, 1 - a)) if not np.isnan(a) else None,
                              "higher_is_risk": bool(a >= 0.5) if not np.isnan(a) else True,
                              "partial_r": _r(pr), "normal_median": _r(self.normal[k][0]),
                              "normal_iqr": [_r(self.normal[k][1]), _r(self.normal[k][2])],
                              "median_at_event": _r(at), "leaves_normal": None})
        d_png, leave = F.drift(self.keys, self.meta, W, self.normal, self.window, self.fps, self.event_name)
        for r in self.rows:
            r["leaves_normal"] = leave.get(r["key"])
        d_png_auc = F.predicts(self.rows, self.ref_auc, self.horizon, self.event_name, f"time to {self.event_name}")
        self.figs.setdefault("drift", d_png)
        self.figs.setdefault("predicts", d_png_auc)
        return self

    # ---------------------------------------------------------------- after compute
    def cause(self, label: str, at_event, among_healthy=None):
        """at_event(ep, i) -> bool for each event; among_healthy(ep) -> bool mask over frames (chance level), or None."""
        share = float(np.mean([bool(at_event(ep, i)) for ep, i in self.event_list])) if self.event_list else float("nan")
        chance = None
        if among_healthy is not None:
            vals = [np.asarray(among_healthy(ep), bool)[ep["healthy"]] for ep in self.episodes]
            chance = float(np.mean(np.concatenate(vals))) if vals else None
        self.causes_table[label] = (share, chance)
        return self

    def examples(self, k: int = 3, seed: int = 0, min_lead: int | None = None):
        """k random (ep, event index) pairs for figure 6, preferring events with at least min_lead steps before."""
        cands = [(ep, i) for ep, i in self.event_list if i >= (min_lead if min_lead is not None else 0)]
        if not cands:
            return []
        rng = np.random.default_rng(seed)
        pick = rng.choice(len(cands), size=min(k, len(cands)), replace=False)
        return [cands[int(p)] for p in pick]

    def last_healthy(self, ep, i: int, keys, pct: float = 75):
        """The onset: the last frame before event i where every signal in `keys` is within the normal range
        (below its healthy `pct` percentile). Returns the index, or None."""
        ok = np.ones(ep["n"], bool)
        for k in keys:
            thr = np.percentile(self.X[:, self.keys.index(k)], pct)
            ok &= ep["sig"][k].values <= thr
        idx = np.where(ok[:i])[0]
        return int(idx[-1]) if len(idx) else None

    def figure(self, key: str, png: str, caption: str | None = None, title: str | None = None):
        """Add or replace a figure. Keys gallery/where/drift/predicts/causes/start get the standard titles;
        any other key becomes an extra section after the six (give it a title)."""
        self.figs[key] = png
        if caption:
            self.captions[key] = caption
        if key not in dict((s[0], s) for s in DEFAULT_SECTIONS) and title:
            self.extra_sections.append((key, title))
        return self

    def tiles(self, tiles):
        self.tile_list = list(tiles)
        return self

    # ---------------------------------------------------------------- output
    def write(self, title: str, subtitle: str, notes_path: str | None = None, open_report: bool = False):
        if self.rows is None:
            raise RuntimeError("call .compute() first")
        os.makedirs(self.out_dir, exist_ok=True)
        if self.causes_table and "causes" not in self.figs:
            self.figs["causes"] = F.causes(self.causes_table, self.event_name)
        n_fail = sum(1 for ep in self.episodes if ep["events"])
        s = {
            "title": title, "subtitle": subtitle, "far_version": __import__("far").__version__,
            "header": {k: v for k, v in self.header.items() if not isinstance(v, (list, dict)) or len(json.dumps(v)) < 500},
            "episodes": len(self.episodes), "episodes_with_event": n_fail, "events": len(self.event_list),
            "healthy_steps": int(len(self.label)), "positive_steps": int(self.label.sum()),
            "horizon": self.horizon, "window": self.window, "fps": self.fps, "event": self.event_name,
            "reference_auc_all_inputs": _r(self.ref_auc),
            "signals": self.rows,
            "causes": {k: {"share": _r(v[0]), "chance": _r(v[1])} for k, v in self.causes_table.items()},
            "outcomes": [ep["outcome"] for ep in self.episodes],
            **self.summary,
        }
        with open(os.path.join(self.out_dir, "summary.json"), "w") as fh:
            json.dump(s, fh, indent=1)
        notes = {}
        notes_path = notes_path or os.path.join(self.out_dir, "notes.json")
        if os.path.exists(notes_path):
            notes = json.load(open(notes_path))
        fmt = dict(event=self.event_name, window=self.window, horizon=self.horizon)
        sections = [(k, t.format(**fmt), self.captions.get(k, notes.get("captions", {}).get(k, c.format(**fmt))))
                    for k, t, c in DEFAULT_SECTIONS if self.figs.get(k)]
        sections += [(k, t, self.captions.get(k, notes.get("captions", {}).get(k, ""))) for k, t in self.extra_sections if self.figs.get(k)]
        html = build(s, self.tile_list, sections, self.figs, notes)
        path = os.path.join(self.out_dir, "report.html")
        with open(path, "w") as fh:
            fh.write(html)
        hits = privacy_scan(self.out_dir)
        self._print(s)
        if hits:
            print("PRIVACY: personal information in the output, fix before publishing:")
            for h in hits:
                print("  ", h)
        if open_report:
            import subprocess
            subprocess.Popen(["xdg-open", path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return path

    def _print(self, s):
        print(f"{s['title']}: {s['episodes']} episodes, {s['events']} {self.event_name} events, "
              f"{s['healthy_steps']:,} healthy steps ({s['positive_steps']:,} within {self.horizon} of an event); "
              f"all inputs together AUC {s['reference_auc_all_inputs']}")
        for r in sorted(self.rows, key=lambda r: -(r["auc_dir"] or 0)):
            print(f"  {r['key']:>18} {r['kind']:>9} auc {r['auc_dir'] if r['auc_dir'] is not None else 'nan':>5} "
                  f"partial {r['partial_r'] if r['partial_r'] is not None else 'nan':>6} normal {r['normal_median']} "
                  f"at event {r['median_at_event']} leaves {r['leaves_normal']}")
        for k, (share, chance) in self.causes_table.items():
            print(f"  cause {k}: {share:.0%}" + (f" (chance {chance:.0%})" if chance is not None else ""))
        print(f"-> {os.path.join(self.out_dir, 'report.html')}")


def _r(v, nd=3):
    if v is None:
        return None
    try:
        if np.isnan(v):
            return None
    except TypeError:
        pass
    return round(float(v), nd)
