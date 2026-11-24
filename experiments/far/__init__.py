"""far: the failure-analysis-report package.

Record how a model fails, compute candidate signals, measure which ones warn of failure and which add
information beyond the model's own inputs, and write one self-contained report.html.

    from far import Analysis, Recorder, load, style
    style.apply()

A project writes two short scripts on top of this package (see templates/):
    record_<thing>.py   -> results/<analysis>/<model>/episodes.jsonl.gz   (Recorder)
    analyze_<thing>.py  -> results/<analysis>/<model>/report.html + summary.json   (Analysis)
"""
from .analysis import Analysis, Signal
from .record import Recorder, load
from . import figures, stats, style, report

__version__ = "1.0.0"
__all__ = ["Analysis", "Signal", "Recorder", "load", "figures", "stats", "style", "report", "__version__"]
