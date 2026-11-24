"""One self-contained report.html (light and dark), and the privacy scan of an output folder."""
from __future__ import annotations

import html as _html
import os
import re

CSS = """
:root{--bg:#fcfcfb;--surface:#fff;--border:#e4e2dc;--text:#0b0b0b;--text-2:#52514e;--text-3:#7a7974;
--input:#2a78d6;--cand:#eb6834;--bad:#c0362c;--good:#1d7a3a}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#141413;--surface:#1a1a19;--border:#34332f;
--text:#fff;--text-2:#c3c2b7;--text-3:#8f8e86;--input:#3987e5;--cand:#d95926;--bad:#e8695f;--good:#4cbf6e;color-scheme:dark}}
:root[data-theme="dark"]{--bg:#141413;--surface:#1a1a19;--border:#34332f;--text:#fff;--text-2:#c3c2b7;
--text-3:#8f8e86;--input:#3987e5;--cand:#d95926;--bad:#e8695f;--good:#4cbf6e;color-scheme:dark}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--text);font:15px/1.55 system-ui,-apple-system,"Segoe UI",sans-serif}
main{max-width:1120px;margin:0 auto;padding:28px 16px 64px}
h1{font-size:26px;margin:0 0 4px;letter-spacing:-.01em}
h2{font-size:19px;margin:40px 0 6px}
.sub{color:var(--text-2);margin:0 0 20px}
.caption{color:var(--text-2);max-width:78ch;margin:0 0 12px}
.fig{background:#fcfcfb;border:1px solid var(--border);border-radius:10px;padding:10px;overflow:hidden}
.fig img{display:block;width:100%;height:auto}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:10px;margin:18px 0}
.tile{background:var(--surface);border:1px solid var(--border);border-radius:10px;padding:12px 14px}
.tile b{display:block;font-size:24px;letter-spacing:-.01em}
.tile span{color:var(--text-3);font-size:13px}
.findings{background:var(--surface);border:1px solid var(--border);border-radius:10px;padding:6px 20px 10px;margin:18px 0}
.findings li{margin:8px 0}
.next{border-left:4px solid var(--cand);background:var(--surface);padding:12px 18px;border-radius:0 10px 10px 0;margin:16px 0}
.next pre{white-space:pre-wrap;font-size:13px;color:var(--text-2);margin:8px 0 0}
.tablewrap{overflow-x:auto;border:1px solid var(--border);border-radius:10px}
table{border-collapse:collapse;width:100%;font-size:13.5px;font-variant-numeric:tabular-nums}
th,td{padding:7px 10px;text-align:right;border-bottom:1px solid var(--border);white-space:nowrap}
th:first-child,td:first-child{text-align:left}
th{color:var(--text-3);font-weight:600;background:var(--surface)}
.dot{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:7px;vertical-align:0}
.in{background:var(--input)}.ca{background:var(--cand)}
.legend{color:var(--text-2);font-size:13px;margin:6px 0 10px}
.foot{color:var(--text-3);font-size:12px;margin-top:40px}
"""


def esc(s):
    return _html.escape(str(s))


def fmt(v, nd=2):
    if v is None:
        return "–"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return esc(v)


def build(s: dict, tiles, sections, figs: dict, notes: dict) -> str:
    """s: summary dict (title, subtitle, signals rows, ...). tiles: [(value, label)]. sections: [(key, title, caption)].
    figs: key -> base64 png. notes: {"findings": [...html...], "next_step": {"title", "text"}, "captions": {...}}."""
    legend = ("<p class='legend'><span class='dot in'></span>the model's current inputs"
              "&nbsp;&nbsp;&nbsp;<span class='dot ca'></span>candidate signals (not inputs)</p>")
    sec = []
    for key, title, caption in sections:
        if key == "drift":
            sec.append(legend)
        sec.append(f"<h2>{esc(title)}</h2><p class='caption'>{caption}</p>"
                   f"<div class='fig'><img alt='{esc(title)}' src='data:image/png;base64,{figs[key]}'></div>")

    unit = "s" if s.get("fps") else "steps"
    table = ["<div class='tablewrap'><table><tr><th>Signal</th><th>Normal (median)</th><th>At event</th>"
             "<th>AUC</th><th>Added info (partial r)</th><th>Leaves normal</th></tr>"]
    for r in sorted(s["signals"], key=lambda r: -(r["auc_dir"] or 0)):
        dot = "in" if r["kind"] == "input" else "ca"
        lv = r["leaves_normal"]
        leave = "–" if lv is None else (f"{lv / s['fps']:+.2f} s" if s.get("fps") else f"{lv:+d} steps")
        aucs = "–" if r["auc_dir"] is None else f"{r['auc_dir']:.2f}{'' if r['higher_is_risk'] else ' ↓'}"
        pr = "–" if r["partial_r"] is None else f"{r['partial_r']:+.2f}"
        table.append(f"<tr><td><span class='dot {dot}'></span>{esc(r['label'])}</td><td>{fmt(r['normal_median'])}</td>"
                     f"<td>{fmt(r['median_at_event'])}</td><td>{aucs}</td><td>{pr}</td><td>{leave}</td></tr>")
    table.append(f"</table></div><p class='legend'>↓ = low values are the risky ones. Partial r below zero = higher "
                 f"signal, sooner {esc(s['event'])}, beyond what the inputs explain. 'Leaves normal' = how long before the "
                 f"event the run-up median leaves the normal IQR for good ({unit}).</p>")

    findings = "".join(f"<li>{f}</li>" for f in notes.get("findings", []))
    nxt = notes.get("next_step")
    nxt_html = (f"<div class='next'><b>Proposed next step: {esc(nxt.get('title', ''))}</b>"
                f"<pre>{esc(nxt.get('text', nxt.get('toml', '')))}</pre></div>" if nxt else "")
    tiles_html = "".join(f"<div class='tile'><b>{esc(v)}</b><span>{esc(k)}</span></div>" for v, k in tiles)
    return f"""<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{esc(s['title'])}</title><style>{CSS}</style></head><body><main>
<h1>{esc(s['title'])}</h1>
<p class="sub">{s['subtitle']}</p>
{f"<div class='tiles'>{tiles_html}</div>" if tiles else ""}
{f"<div class='findings'><h2 style='margin-top:12px'>Key findings</h2><ol>{findings}</ol></div>" if findings else ""}
{nxt_html}
<h2>Summary</h2>{''.join(table)}
{''.join(sec)}
<p class="foot">Findings are leads, not results: validate by implementing the change, retraining, and a paired comparison on
held-out data with bootstrap confidence intervals. Numbers in this page come from summary.json next to it.</p>
</main></body></html>"""


def privacy_scan(out_dir: str, extra_terms=()):
    """Lines in the output folder's text files that contain the login name, a /home/<user> path, or extra terms."""
    user = os.environ.get("USER") or os.path.basename(os.path.expanduser("~"))
    terms = [t for t in (user, *extra_terms) if t and len(t) > 2]
    pats = [re.compile(r"/home/[A-Za-z0-9_.-]+")] + [re.compile(re.escape(t), re.I) for t in terms]
    hits = []
    for name in sorted(os.listdir(out_dir)):
        if not name.endswith((".html", ".json", ".md", ".txt", ".csv")):
            continue
        p = os.path.join(out_dir, name)
        with open(p, errors="ignore") as fh:
            for ln, line in enumerate(fh, 1):
                if len(line) > 5000:   # base64 figures
                    line = re.sub(r"data:image/png;base64,[A-Za-z0-9+/=]+", "", line)
                for pat in pats:
                    m = pat.search(line)
                    if m:
                        hits.append(f"{name}:{ln}: ...{line[max(0, m.start() - 30):m.end() + 30].strip()}...")
                        break
    return hits
