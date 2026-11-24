"""House style for every figure (dataviz reference palette) and the PNG helper."""
from __future__ import annotations

import base64
import io

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

SURFACE, INK, INK2, INK3, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#7a7974", "#e4e3df"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
BLUE, ORANGE, GREEN, YELLOW, PINK, DARKGREEN, PURPLE, RED = SERIES
INPUT, CANDIDATE = BLUE, ORANGE          # current inputs blue, candidate signals orange, everywhere
BLUES = LinearSegmentedColormap.from_list("far_blues", [SURFACE, "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])

RC = {
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": INK3, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.titlecolor": INK, "axes.titlesize": 11, "axes.titleweight": "bold",
    "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8, "font.size": 10,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 2, "legend.frameon": False,
    "font.family": "DejaVu Sans",
}


def apply():
    """Call once at import time of the analysis script."""
    plt.rcParams.update(RC)


def kind_color(kind: str) -> str:
    return INPUT if kind == "input" else CANDIDATE


def png(fig, dpi: int = 110) -> str:
    """Render a figure to a base64 PNG string (for the self-contained report) and close it."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def label_bars(ax, bars, values, fmt="{:.2f}", color=INK2, fontsize=8.5, horizontal=False):
    """Direct value labels at the end of each bar."""
    for b, v in zip(bars, values):
        if horizontal:
            ax.text(b.get_width() + (b.get_x() if b.get_x() else 0), b.get_y() + b.get_height() / 2, " " + fmt.format(v),
                    va="center", ha="left", fontsize=fontsize, color=color)
        else:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + (b.get_y() if b.get_y() else 0), fmt.format(v),
                    ha="center", va="bottom", fontsize=fontsize, color=color)
