"""Main results table (LaTeX, booktabs-free to match the paper's style) + CSV/Markdown.

Multi-seed cells are ``\\ms{mean}{std}`` (sample std over seeds); single-run cells are
plain numbers and the model name gets a dagger.  The best value per (model, metric)
among the non-excluded methods is bold (``\\bms`` / ``\\textbf``); ties at the
displayed precision are all bold.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .results import method_label, model_label, HIGHER_BETTER, METHOD_LABELS, METRIC_KEYS, MODEL_LABELS, SCALE, order

PREAMBLE = r"""% Requires in the preamble:
% \newcommand{\ms}[2]{#1{\tiny$\pm$#2}}
% \newcommand{\bms}[2]{\textbf{#1}{\tiny$\pm$\textbf{#2}}}
% \newcommand{\Ctok}{C_{\mathrm{tok}}} \newcommand{\Cphr}{C_{\mathrm{phr}}}
% \newcommand{\Ccfg}{C_{\mathrm{cfg}}} \newcommand{\Cbar}{\bar{C}}
"""


def summarize(runs: pd.DataFrame) -> pd.DataFrame:
    """mean / std / n over seeds for every (model, method, metric)."""
    g = runs.groupby(["model", "method"])
    rows = []
    for (model, method), df in g:
        rec = {"model": model, "method": method, "n_seeds": len(df), "seeds": ",".join(map(str, sorted(df.seed)))}
        for k in METRIC_KEYS:
            v = df[k].dropna().astype(float) * SCALE[k]
            rec[f"{k}_mean"] = v.mean() if len(v) else np.nan
            rec[f"{k}_std"] = v.std(ddof=1) if len(v) > 1 else np.nan
        rows.append(rec)
    return pd.DataFrame(rows)


def _fmt(x: float, k: str) -> str:
    return f"{x:.1f}" if k == "halluc" else f"{x:.3f}"


def latex_table(summary: pd.DataFrame, domain: str, models: List[str], methods: List[str],
                bold_exclude: Optional[List[str]] = None, caption: str = "", label: str = "") -> str:
    bold_exclude = set(bold_exclude or ["vanilla"])
    lines = [PREAMBLE, r"\begin{table*}[ht]", r"\centering", r"\scriptsize",
             r"\caption{" + (caption or f"Results on the \\textbf{{{domain.upper()}}} test set. "
                             r"\textbf{Bold}: best per model. Hall.\%~$\downarrow$; all others~$\uparrow$. "
                             r"Mean~$\pm$~std over seeds; \dag: single run. Vanilla and GCD$^\dagger$ decode the untrained "
                             r"model greedily and are deterministic, hence a single run.") + "}",
             r"\label{" + (label or f"tab:results_{domain}") + "}",
             r"\setlength{\tabcolsep}{2.8pt}", r"\renewcommand{\arraystretch}{1.08}",
             r"\begin{tabular}{ll *{4}{c} *{3}{c} c cc}", r"\hline",
             r"\multirow{2}{*}{\textbf{Model}} & \multirow{2}{*}{\textbf{Method}}"
             r" & \multicolumn{4}{c}{\textbf{Compliance}} & \multicolumn{3}{c}{\textbf{Semantic}}"
             r" & & \multicolumn{2}{c}{\textbf{BERTScore}} \\",
             r"\cline{3-6}\cline{7-9}\cline{11-12}",
             r" && \Ctok & \Cphr & \Ccfg & \Cbar & Slot-F1 & DA-F1 & Hall.\%$\downarrow$ & & $_\mathrm{D}$ & $_\mathrm{G}$ \\",
             r"\hline"]
    for model in models:
        sm = summary[summary.model == model].set_index("method")
        present = [m for m in methods if m in sm.index]
        multi = any(sm.loc[m, "n_seeds"] > 1 for m in present) if present else False
        # best per metric (on rounded display values)
        best: Dict[str, float] = {}
        for k in METRIC_KEYS:
            vals = [round(sm.loc[m, f"{k}_mean"], 1 if k == "halluc" else 3)
                    for m in present if m not in bold_exclude and pd.notna(sm.loc[m, f"{k}_mean"])]
            if vals:
                best[k] = max(vals) if HIGHER_BETTER[k] else min(vals)
        label = model_label(model) + ("" if multi else r"\dag")
        lines.append(rf"\multirow{{{len(methods)}}}{{*}}{{\rotatebox[origin=c]{{90}}{{{label}}}}}")
        for method in methods:
            cells = []
            for k in METRIC_KEYS:
                if method not in sm.index or pd.isna(sm.loc[method, f"{k}_mean"]):
                    cells.append("---")
                    continue
                mu, sd = sm.loc[method, f"{k}_mean"], sm.loc[method, f"{k}_std"]
                is_best = (method not in bold_exclude and k in best
                           and round(mu, 1 if k == "halluc" else 3) == best[k])
                if pd.notna(sd):
                    cells.append((r"\bms" if is_best else r"\ms") + f"{{{_fmt(mu, k)}}}{{{_fmt(sd, k)}}}")
                else:
                    cells.append(rf"\textbf{{{_fmt(mu, k)}}}" if is_best else _fmt(mu, k))
            row = cells[:7] + [""] + cells[7:]
            lines.append(f"  & {method_label(method)}\n    & " + " & ".join(row) + r" \\")
        lines.append(r"\hline")
    lines += [r"\end{tabular}", r"\end{table*}"]
    return "\n".join(lines)


def markdown_table(summary: pd.DataFrame, models: List[str], methods: List[str]) -> str:
    head = "| Model | Method | n | " + " | ".join(METRIC_KEYS) + " |"
    out = [head, "|" + "---|" * (len(METRIC_KEYS) + 3)]
    for model in models:
        for method in methods:
            r = summary[(summary.model == model) & (summary.method == method)]
            if r.empty:
                continue
            r = r.iloc[0]
            cells = []
            for k in METRIC_KEYS:
                mu, sd = r[f"{k}_mean"], r[f"{k}_std"]
                cells.append("—" if pd.isna(mu) else (_fmt(mu, k) + (f" ± {_fmt(sd, k)}" if pd.notna(sd) else "")))
            out.append(f"| {model} | {method} | {int(r.n_seeds)} | " + " | ".join(cells) + " |")
    return "\n".join(out)


def build_tables(runs: pd.DataFrame, domain: str, models: Optional[List[str]], methods: Optional[List[str]],
                 bold_exclude=None):
    from .results import MODEL_LABELS as ML
    from ..runner import ALL_METHODS
    models = models or order(runs.model, list(ML))
    # main table: main-grid methods only (ablation variants `<method>__<variant>` go to ablate-report)
    methods = methods or [m for m in order(runs.method, ALL_METHODS) if "__" not in m]
    summary = summarize(runs)
    return summary, latex_table(summary, domain, models, methods, bold_exclude), markdown_table(summary, models, methods)
