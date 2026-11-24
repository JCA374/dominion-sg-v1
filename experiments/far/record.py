"""Recording failures with context: one .jsonl.gz per model, a header line, then one episode per line.

    rec = Recorder("results/fall_analysis/champion/episodes.jsonl.gz", header={"model": ..., "fps": 50, ...})
    for seed in seeds:
        ... run the episode, appending per-step values to lists ...
        rec.add(outcome={"seed": seed, "failed": True, "cause": "fell", "steps": n},
                frames={"obs": obs_list, "action": act_list, "x": xs, ...},
                extra={"track": {...}})          # anything static for the episode (optional)
    rec.close()

Keep EVERY step of every episode when the data allows it (tens of MB compressed is fine); the analysis cuts the
windows it wants. If episodes are very long, keep the last `window` steps plus snapshots every `every` steps
(Recorder(..., keep_last=300, snapshot_every=100)) and the analysis uses the snapshots as the normal baseline.
Round floats (3 decimals by default) so the file stays small. Use a seed range that is never used for
training, validation or the final test.
"""
from __future__ import annotations

import gzip
import json
import os
import time

import numpy as np


def _round(v, nd):
    if isinstance(v, (list, tuple)):
        return [_round(x, nd) for x in v]
    if isinstance(v, np.ndarray):
        return np.round(v.astype(float), nd).tolist()
    if isinstance(v, (float, np.floating)):
        return round(float(v), nd)
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.bool_,)):
        return bool(v)
    return v


class Recorder:
    def __init__(self, path: str, header: dict, decimals: int = 3, keep_last: int | None = None,
                 snapshot_every: int | None = None):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self.path, self.decimals = path, decimals
        self.keep_last, self.snapshot_every = keep_last, snapshot_every
        self.n = 0
        self.fh = gzip.open(path, "wt")
        hdr = dict(header)
        hdr.setdefault("format", "far-episodes-1")
        hdr.setdefault("recorded_at", time.strftime("%Y-%m-%d %H:%M"))
        hdr["keep_last"], hdr["snapshot_every"] = keep_last, snapshot_every
        self.fh.write(json.dumps(hdr) + "\n")

    def add(self, outcome: dict, frames: dict, extra: dict | None = None):
        """outcome: small dict (seed, failed, cause, steps, score...). frames: name -> list/array, one entry per step.
        Lengths must agree. With keep_last set, only the last keep_last steps are kept in `frames` and every
        snapshot_every-th step goes to `snapshots` (same keys), with `snapshot_steps` giving their step indices."""
        lens = {k: len(v) for k, v in frames.items()}
        n = max(lens.values()) if lens else 0
        if any(l != n for l in lens.values()):
            raise ValueError(f"frame arrays differ in length: {lens}")
        rec = {"outcome": _round(outcome, self.decimals), "n_steps": n}
        if self.keep_last is not None and n > self.keep_last:
            lo = n - self.keep_last
            rec["frames_start"] = lo
            rec["frames"] = {k: _round(list(v)[lo:], self.decimals) for k, v in frames.items()}
            if self.snapshot_every:
                idx = list(range(0, lo, self.snapshot_every))
                rec["snapshot_steps"] = idx
                rec["snapshots"] = {k: _round([list(v)[i] for i in idx], self.decimals) for k, v in frames.items()}
        else:
            rec["frames_start"] = 0
            rec["frames"] = {k: _round(list(v), self.decimals) for k, v in frames.items()}
        if extra:
            rec["extra"] = _round(extra, self.decimals)
        self.fh.write(json.dumps(rec) + "\n")
        self.n += 1

    def close(self):
        self.fh.close()
        return self.path

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()


class Episode(dict):
    """One recorded episode. ep["f"] holds per-step float arrays, ep["outcome"] the outcome dict,
    ep["n"] the number of steps recorded in f, ep["start"] the step index of f[0], ep["snap"] the snapshot arrays
    (if any), ep["extra"] the static extras. Analysis results are added as ep["sig"], ep["events"], ep["healthy"]."""


def load(path: str) -> tuple[dict, list[Episode]]:
    with gzip.open(path, "rt") as fh:
        header = json.loads(fh.readline())
        episodes = []
        for line in fh:
            if not line.strip():
                continue
            d = json.loads(line)
            ep = Episode(outcome=d["outcome"], n=len(next(iter(d["frames"].values()), [])),
                         start=d.get("frames_start", 0), extra=d.get("extra", {}), n_steps=d.get("n_steps"))
            ep["f"] = {k: np.asarray(v, dtype=float) for k, v in d["frames"].items()}
            if "snapshots" in d:
                ep["snap"] = {k: np.asarray(v, dtype=float) for k, v in d["snapshots"].items()}
                ep["snap_steps"] = np.asarray(d["snapshot_steps"])
            episodes.append(ep)
    return header, episodes
