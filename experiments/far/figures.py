"""The six figures, one per question. Project-specific drawing (what a state looks like) comes in as callbacks."""
from __future__ import annotations

import math
import warnings

import numpy as np
import matplotlib.pyplot as plt

from . import style as S
from .stats import leaves_normal


def gallery(episodes, draw, title, cols: int = 10, cell=(1.55, 1.75), key=None):
    """1. What does failure look like? One small panel per episode, sorted by `key` (worst first).
    draw(ax, ep) draws the episode; title(ep) -> (text, color)."""
    order = sorted(episodes, key=key) if key else list(episodes)
    rows = max(1, math.ceil(len(order) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * cell[0], rows * cell[1]), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for ax, ep in zip(axes.flat, order):
        draw(ax, ep)
        t, c = title(ep)
        ax.set_title(t, fontsize=7.5, color=c, fontweight="normal")
    fig.tight_layout(pad=0.3)
    return S.png(fig)


def where_bars(labels, event_counts, healthy_counts, xlabel, ylabel="events per 1000\nhealthy steps", title=""):
    """2. Where does it happen? Event rate per 1000 healthy steps in each bin of a context variable."""
    ev, he = np.asarray(event_counts, float), np.asarray(healthy_counts, float)
    rate = 1000 * ev / np.maximum(he, 1)
    fig, ax = plt.subplots(figsize=(7.5, 3.2))
    x = np.arange(len(labels))
    bars = ax.bar(x, rate, color=S.CANDIDATE, width=0.7)
    for b, r, n in zip(bars, rate, ev):
        ax.text(b.get_x() + b.get_width() / 2, r, f"{r:.1f}\n({int(n)})", ha="center", va="bottom", fontsize=8.5, color=S.INK2)
    ax.set_xticks(x, labels)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    if title:
        ax.set_title(title)
    ax.set_ylim(0, (rate.max() if len(rate) else 1) * 1.3 + 0.1)
    return S.png(fig), {str(l): round(float(r), 2) for l, r in zip(labels, rate)}


def where_heatmaps(grids, titles, extent=None, cmap=None, vmax=None, xlabel="", ylabel=""):
    """2 (alt). Normal vs at-failure heatmaps (and their difference if 2 grids are given), one hue ramp."""
    grids = [np.asarray(g, float) for g in grids]
    if len(grids) == 2:
        grids = grids + [grids[1] - grids[0]]
        titles = list(titles) + ["difference (failure − normal)"]
    fig, axes = plt.subplots(1, len(grids), figsize=(3.6 * len(grids), 3.4))
    axes = np.atleast_1d(axes)
    vmax = vmax or max(np.nanmax(np.abs(g)) for g in grids[:2])
    for ax, g, t in zip(axes, grids, titles):
        diff = t.startswith("difference")
        im = ax.imshow(g, cmap="RdBu_r" if diff else (cmap or S.BLUES), vmin=-vmax if diff else 0, vmax=vmax,
                       interpolation="nearest", origin="lower", extent=extent, aspect="auto")
        ax.set_title(t)
        ax.grid(False)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    fig.tight_layout()
    return S.png(fig)


def drift(keys, meta, windows, normal, window: int, fps: float | None, event_name="failure", cols: int = 4):
    """3. What drifts before it? Median + IQR of each signal over the last `window` steps before each event,
    with the normal level (median, IQR on healthy steps) dashed. Returns (png, {key: leave index or None})."""
    n = len(keys)
    rows = max(1, math.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.3, rows * 2.4), sharex=True, squeeze=False)
    t = np.arange(-window, 1) / (fps or 1.0)
    unit = "seconds" if fps else "steps"
    leave = {}
    for ax, k in zip(axes.flat, keys):
        W = windows[k]
        ax.set_title(meta[k].label, fontsize=9.5)
        if W.size == 0 or np.all(np.isnan(W)):
            ax.text(0.5, 0.5, "no events", transform=ax.transAxes, ha="center", color=S.INK2)
            ax.set_yticks([])
            leave[k] = None
            continue
        if np.nanmax(W) == np.nanmin(W) == normal[k][0]:
            ax.text(0.5, 0.5, f"always {normal[k][0]:g}", transform=ax.transAxes, ha="center", color=S.INK2)
            ax.set_yticks([])
            leave[k] = None
            continue
        with np.errstate(all="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)       # NaN-padded columns before short episodes
            med, lo, hi = np.nanmedian(W, 0), np.nanpercentile(W, 25, 0), np.nanpercentile(W, 75, 0)
        c = S.kind_color(meta[k].kind)
        ax.fill_between(t, lo, hi, color=c, alpha=0.18, lw=0)
        ax.plot(t, med, color=c)
        nm, nlo, nhi = normal[k]
        ax.axhline(nm, color=S.INK3, ls="--", lw=1.2)
        ax.axhspan(nlo, nhi, color=S.INK3, alpha=0.06, lw=0)
        idx = leaves_normal(med, nlo, nhi)
        leave[k] = None if idx is None else idx - window      # steps before the event (negative)
        if leave[k] is not None and leave[k] < 0:
            ax.axvline(t[idx], color=c, lw=1, ls=":")
    for ax in list(axes.flat)[n:]:
        ax.axis("off")
    for ax in axes[-1]:
        ax.set_xlabel(f"{unit} before {event_name}")
    fig.tight_layout()
    return S.png(fig), leave


def predicts(rows, ref, horizon: int, event_name="failure", target_name="time to failure"):
    """4. What predicts it? Left: AUC per signal (folded to >= 0.5, '(low)' marks when low values are the risk),
    dashed = all inputs together. Right: partial r with log(time to event) after regressing out the inputs."""
    rows = sorted(rows, key=lambda r: -abs((r["auc_dir"] or 0.5) - 0.5))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 0.32 * len(rows) + 1.2), sharey=True)
    y = np.arange(len(rows))[::-1]
    for yy, r in zip(y, rows):
        c = S.kind_color(r["kind"])
        ad = r["auc_dir"]
        if ad is None:
            a1.text(0.505, yy, "no events", va="center", fontsize=8, color=S.INK2)
        else:
            a1.barh(yy, ad - 0.5, left=0.5, color=c, height=0.62)
            a1.text(ad + 0.005, yy, f"{ad:.2f}{'' if r['higher_is_risk'] else ' (low)'}", va="center", fontsize=8, color=S.INK2)
        pr = 0.0 if r["partial_r"] is None or np.isnan(r["partial_r"]) else r["partial_r"]
        a2.barh(yy, pr, color=c, height=0.62)
        a2.text(pr + (0.004 if pr >= 0 else -0.004), yy, f"{pr:+.2f}", va="center",
                ha="left" if pr >= 0 else "right", fontsize=8, color=S.INK2)
    if ref is not None and not np.isnan(ref):
        a1.axvline(ref, color=S.INK, ls="--", lw=1.2)
        a1.text(ref, len(rows) - 0.3, f" all inputs together: {ref:.2f}", fontsize=8, color=S.INK, va="bottom")
    a1.set_yticks(y, [r["label"] for r in rows])
    a1.set_xlim(0.5, max(1.0, (ref or 0) + 0.05))
    a1.set_xlabel(f"AUC: {event_name} within {horizon} steps")
    a2.axvline(0, color=S.INK3, lw=1)
    a2.set_xlabel(f"partial r with log({target_name}),\ninputs regressed out (− = risk)")
    lim = max(0.1, max(abs(r["partial_r"] or 0) for r in rows) * 1.35)
    a2.set_xlim(-lim, lim)
    fig.tight_layout()
    return S.png(fig)


def causes(table, event_name="failure"):
    """5. What causes it? table: {label: (share among events, chance share among healthy steps or None)}."""
    labels = list(table)
    fig, ax = plt.subplots(figsize=(8, 0.42 * len(labels) + 0.8))
    y = np.arange(len(labels))[::-1]
    for yy, k in zip(y, labels):
        share, chance = table[k]
        ax.barh(yy, share * 100, color=S.CANDIDATE, height=0.6)
        ax.text(share * 100 + 1, yy, f"{share * 100:.0f}%", va="center", fontsize=8.5, color=S.INK2)
        if chance is not None and not np.isnan(chance):
            ax.plot([chance * 100] * 2, [yy - 0.38, yy + 0.38], color=S.INK, lw=2)
    ax.set_yticks(y, labels)
    ax.set_xlim(0, 105)
    ax.set_xlabel(f"% of {event_name} events (black tick = the same share among healthy steps)")
    fig.tight_layout()
    return S.png(fig)


def start(examples, draw, offsets=(-30, -10, 0, 20), cell=(3.0, 3.1), title=None):
    """6. How does it start? One row per example (ep, event index), one column per offset around the event.
    draw(ax, ep, i) draws the state at step i; title(ep, i, off) -> str (optional)."""
    if not examples:
        return ""
    fig, axes = plt.subplots(len(examples), len(offsets), figsize=(cell[0] * len(offsets), cell[1] * len(examples)), squeeze=False)
    for row, (ep, i0) in zip(axes, examples):
        n = ep["n"]
        for ax, off in zip(row, offsets):
            i = int(np.clip(i0 + off, 0, n - 1))
            draw(ax, ep, i)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(False)
            t = title(ep, i, off) if title else ("event" if off == 0 else f"{off:+d} steps")
            ax.set_title(t, fontsize=8.5, fontweight="normal", color=S.INK)
    fig.tight_layout()
    return S.png(fig)


def histogram(values, title, xlabel, bins=30):
    """A plain histogram in house style (e.g. steps from the last healthy state to failure)."""
    fig, ax = plt.subplots(figsize=(6, 2.8))
    ax.hist(values, bins=bins, color=S.BLUE, rwidth=0.85)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    fig.tight_layout()
    return S.png(fig)
