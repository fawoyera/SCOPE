"""Ablation report (``scope ablate-report``).

For every group of the spec (components, recipe, data, grpo, lambda, compute, inference):
  * mean ± std over seeds of every metric, per variant;
  * Δ vs the full model with an example-level paired permutation test (Holm-corrected
    *within the group*, i.e. one family per table) — arrows give direction, stars significance;
  * training cost (cumulative GPU-hours incl. any SFT start point), from compute.json.
Figures: one-at-a-time sensitivity curves for every ``sweep`` parameter, and a
compliance (C̄) vs. consistency (Slot-F1) Pareto plot over all variants.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .compute import STARTS_FROM, collect as collect_compute, training_table
from .results import method_label, model_label, HIGHER_BETTER, METRIC_KEYS, MODEL_LABELS, SCALE, collect_examples, collect_runs
from .stats import compare, stars

TABLE_METRICS = ["c_tok", "c_phr", "c_cfg", "c_bar", "slot_f1", "da_f1", "halluc"]


def variant_dir(spec: dict, v: str) -> str:
    vv = spec["variants"][v] or {}
    return vv["reuse"] if vv.get("reuse") else f"{spec['base_method']}__{v}"


def _fmt(mu, sd, k):
    d = 1 if k == "halluc" else 3
    if pd.isna(mu):
        return "---"
    return f"{mu:.{d}f}" + (rf"{{\tiny$\pm${sd:.{d}f}}}" if pd.notna(sd) else "")


def build(spec: dict, root: str, domain: str, models: Optional[List[str]] = None,
          groups: Optional[List[str]] = None, seed: int = 0):
    runs = collect_runs(root, domain)
    if runs.empty:
        raise SystemExit(f"No finished runs under {root}/{domain}")
    vmap = {variant_dir(spec, v): v for v in spec["variants"]}
    runs = runs[runs.method.isin(vmap)]
    if models:
        runs = runs[runs.model.isin(models)]
    ex = collect_examples(root, domain, models)
    if not ex.empty:
        ex = ex[ex.method.isin(vmap)]
    tr, _, _, _ = collect_compute(root, domain)
    cost = training_table(tr) if not tr.empty else pd.DataFrame()
    ref_dir = variant_dir(spec, spec.get("reference", "full"))

    rows, sig = [], []
    for g, gdef in spec["groups"].items():
        if groups and g not in groups:
            continue
        dirs = [variant_dir(spec, v) for v in gdef["variants"]]
        R = runs[runs.method.isin(dirs)]
        if R.empty:
            continue
        # contrast of each variant: the group's per-series reference (e.g. prompt × method) or full SCOPE
        refs = gdef.get("references") or {}
        ref_of = {}
        for v, d in zip(gdef["variants"], dirs):
            series = ((spec["variants"][v] or {}).get("sweep") or {}).get("series")
            ref_of[d] = variant_dir(spec, refs[series]) if series in refs else ref_dir
        parts = []
        for rd in dict.fromkeys(ref_of.values()):
            fam = [d for d, r in ref_of.items() if r == rd and d != rd]
            if not fam:
                continue
            keep = fam + [rd]
            ci = compare(R[R.method.isin(keep)], ex[ex.method.isin(keep)] if not ex.empty else ex,
                         reference=rd, baselines=fam, seed=seed)
            if not ci.empty:
                parts.append(ci)
        c = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
        if not c.empty:
            c["group"] = g
            sig.append(c)
        for model in sorted(R.model.unique()):
            for v, d in zip(gdef["variants"], dirs):
                df = R[(R.model == model) & (R.method == d)]
                if df.empty:
                    continue
                rec = {"group": g, "model": model, "variant": v, "dir": d,
                       "label": spec["variants"][v].get("label", v), "n_seeds": len(df)}
                for k in METRIC_KEYS:
                    x = df[k].dropna().astype(float) * SCALE[k]
                    rec[f"{k}_mean"] = x.mean() if len(x) else np.nan
                    rec[f"{k}_std"] = x.std(ddof=1) if len(x) > 1 else np.nan
                    if d != ref_of[d] and not c.empty:
                        s = c[(c.model == model) & (c.baseline == d) & (c.reference == ref_of[d]) & (c.metric == k)]
                        if len(s):
                            s = s.iloc[0]
                            # Δ reported as variant − its reference (sign flipped from compare's ref − base)
                            rec[f"{k}_delta"] = -s.get("delta_example", s["delta"])
                            rec[f"{k}_p_holm"] = s.get("perm_p_holm", np.nan)
                if not cost.empty:
                    cc = cost[(cost.model == model) & (cost.method == d)]
                    if not len(cc):     # decoding-only variant: inherits the cost of the checkpoint it decodes
                        vv = spec["variants"][v] or {}
                        src = (variant_dir(spec, vv["init_from"]) if vv.get("init_from") in spec["variants"]
                               else vv.get("init_from") or STARTS_FROM.get(vv.get("method", "")))
                        cc = cost[(cost.model == model) & (cost.method == src)] if src else cc
                    rec["gpu_hours_cumulative"] = cc.gpu_hours_cumulative_mean.iloc[0] if len(cc) else np.nan
                rec["is_reference"] = d == ref_of[d] or d in ref_of.values()
                rec["reference"] = ref_of[d]
                sw = (spec["variants"][v] or {}).get("sweep")
                if sw and (sw["param"] != "prompt" or g == "prompt" or "series" not in sw):
                    rec["sweep_param"], rec["sweep_value"] = sw["param"], sw["value"]
                    rec["sweep_series"] = sw.get("series")
                rows.append(rec)
    table = pd.DataFrame(rows)
    sig = pd.concat(sig, ignore_index=True) if sig else pd.DataFrame()
    return table, sig


def latex(table: pd.DataFrame, spec: dict, domain: str) -> str:
    if table.empty:
        return "% no ablation results"
    has_cost = "gpu_hours_cumulative" in table and table.gpu_hours_cumulative.notna().any()
    hdr = r"Variant & \Ctok & \Cphr & \Ccfg & \Cbar & Slot-F1 & DA-F1 & Hall.\%$\downarrow$" + (" & GPU-h" if has_cost else "") + r" \\"
    ncol = 8 + has_cost
    L = [r"\begin{table*}[ht]", r"\centering", r"\scriptsize",
         r"\caption{Ablations on " + domain.upper() + r" (test set; mean~$\pm$~std over seeds). Arrows: direction of "
         r"the change w.r.t.\ full SCOPE ($\uparrow$ better, $\downarrow$ worse); stars: example-level paired permutation "
         r"test, Holm-corrected within each block (* $p<.05$, ** $p<.01$, *** $p<.001$). GPU-h includes the SFT run a "
         r"variant starts from.}", r"\label{tab:ablations_" + domain + "}", r"\setlength{\tabcolsep}{3pt}",
         r"\begin{tabular}{l" + "c" * (ncol - 1) + "}", r"\hline"]
    for model in table.model.unique():
        L.append(rf"\multicolumn{{{ncol}}}{{l}}{{\textbf{{{model_label(model)}}}}} \\")
        for g, gdef in spec["groups"].items():
            T = table[(table.model == model) & (table.group == g)]
            if T.empty:
                continue
            L += [r"\hline", rf"\multicolumn{{{ncol}}}{{l}}{{\textit{{({chr(97 + list(spec['groups']).index(g))}) "
                             rf"{gdef['title']}}}}} \\", hdr]
            for _, r in T.iterrows():
                cells = []
                for k in TABLE_METRICS:
                    c = _fmt(r[f"{k}_mean"], r[f"{k}_std"], k)
                    d, p = r.get(f"{k}_delta", np.nan), r.get(f"{k}_p_holm", np.nan)
                    s = stars(p)
                    if pd.notna(d) and s:
                        better = (d > 0) == HIGHER_BETTER[k]
                        arrow = r"\uparrow" if better else r"\downarrow"
                        c += f"$^{{{arrow}{s}}}$"
                    cells.append(c)
                if has_cost:
                    cells.append("---" if pd.isna(r.get("gpu_hours_cumulative")) else f"{r['gpu_hours_cumulative']:.1f}")
                name = rf"\textbf{{{r['label']}}}" if r.get("is_reference") else r["label"]
                L.append(f"{name} & " + " & ".join(cells) + r" \\")
        L.append(r"\hline")
    L += [r"\end{tabular}", r"\end{table*}"]
    return "\n".join(L)


def figures(table: pd.DataFrame, spec: dict, out: Path, domain: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if table.empty:
        return
    full = table[table.variant == spec.get("reference", "full")].drop_duplicates("model")
    fv = spec.get("full_values", {})
    sw = table.dropna(subset=["sweep_param"]) if "sweep_param" in table else pd.DataFrame()
    ticks = spec.get("sweep_ticks", {})
    if not sw.empty:
        cat = sw[sw.sweep_param.isin(list(ticks))].drop_duplicates(["model", "variant"])
        sw = sw[~sw.sweep_param.isin(list(ticks))]
        _categorical_figures(cat, ticks, out, domain)
    for model in table.model.unique():
        if not sw.empty:
            params = list(dict.fromkeys(sw[sw.model == model].sweep_param))
            if params:
                fig, axes = plt.subplots(1, len(params), figsize=(3.4 * len(params), 3), squeeze=False)
                for ax, prm in zip(axes[0], params):
                    S = sw[(sw.model == model) & (sw.sweep_param == prm)].copy()
                    F = full[full.model == model]
                    if prm in fv and len(F):
                        S = pd.concat([S, F.assign(sweep_value=fv[prm])])
                    S = S.sort_values("sweep_value")
                    for k, col in (("c_bar", "#c51b7d"), ("slot_f1", "#1f78b4"), ("c_cfg", "#7570b3")):
                        ax.errorbar(S.sweep_value, S[f"{k}_mean"], yerr=S[f"{k}_std"].fillna(0), marker="o",
                                    capsize=2, color=col, label=k)
                    if prm in fv:
                        ax.axvline(fv[prm], color="grey", ls=":", lw=1)
                    ax.set(xscale="log" if prm.startswith("lambda") else "linear", xlabel=prm, ylim=(0, 1))
                    ax.grid(alpha=.3)
                axes[0][0].legend(fontsize=7)
                fig.suptitle(f"Sensitivity — {MODEL_LABELS.get(model, model)}")
                plt.tight_layout()
                plt.savefig(out / f"ablation_{domain}_{model}_sensitivity.pdf", bbox_inches="tight")
                plt.close(fig)
        T = table[table.model == model].drop_duplicates("variant")
        fig, ax = plt.subplots(figsize=(6.5, 5))
        for _, r in T.iterrows():
            ref = r.variant == spec.get("reference", "full")
            ax.errorbar(r.c_bar_mean, r.slot_f1_mean, xerr=r.c_bar_std if pd.notna(r.c_bar_std) else 0,
                        yerr=r.slot_f1_std if pd.notna(r.slot_f1_std) else 0, fmt="*" if ref else "o",
                        ms=12 if ref else 5, color="#c51b7d" if ref else "#555", capsize=2)
            ax.annotate(r.variant, (r.c_bar_mean, r.slot_f1_mean), fontsize=6, xytext=(3, 2), textcoords="offset points")
        ax.set(xlabel="test $\\bar C$ (compliance)", ylabel="test Slot-F1 (consistency)",
               title=f"Ablation variants — {MODEL_LABELS.get(model, model)}")
        ax.grid(alpha=.3)
        plt.tight_layout()
        plt.savefig(out / f"ablation_{domain}_{model}_pareto.pdf", bbox_inches="tight")
        plt.close(fig)


def run(spec: dict, root: str, domain: str, out_dir: str, models=None, groups=None) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    table, sig = build(spec, root, domain, models, groups)
    table.to_csv(out / f"ablation_{domain}.csv", index=False)
    if not sig.empty:
        sig.to_csv(out / f"ablation_{domain}_significance.csv", index=False)
    (out / f"ablation_{domain}.tex").write_text(latex(table, spec, domain))
    figures(table, spec, out, domain)
    show = ["group", "model", "variant", "n_seeds"] + [f"{k}_mean" for k in ("c_bar", "c_cfg", "slot_f1", "halluc")] \
        + [c for c in ("c_bar_delta", "c_bar_p_holm", "gpu_hours_cumulative") if c in table]
    with pd.option_context("display.width", 220, "display.max_rows", 300, "display.float_format", "{:.4g}".format):
        print(table[show].to_string(index=False))
    print(f"\nWrote {out}/ablation_{domain}.tex/.csv and figures")


def _categorical_figures(cat: pd.DataFrame, ticks: dict, out: Path, domain: str) -> None:
    """One panel per metric, x = ordered categorical level, one line per method (series)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from .results import method_label
    metrics = [("c_bar", r"$\bar C$"), ("c_cfg", r"$C_{cfg}$"), ("slot_f1", "Slot-F1"), ("halluc", "Hall. %")]
    for (model, prm), S in cat.groupby(["model", "sweep_param"]):
        fig, axes = plt.subplots(1, len(metrics), figsize=(3.3 * len(metrics), 3.1))
        for ax, (k, name) in zip(axes, metrics):
            for series, T in S.groupby("sweep_series"):
                T = T.sort_values("sweep_value")
                lab = method_label(series).replace(r"\textbf{", "").replace("}", "").replace("$^\\dagger$", "†")
                ax.errorbar(T.sweep_value, T[f"{k}_mean"], yerr=T[f"{k}_std"].fillna(0), marker="o", capsize=2,
                            label=lab, lw=2.2 if series == "scope" else 1.3)
            labels = ticks.get(prm, [])
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(labels, fontsize=8)
            ax.set(title=name, xlabel=f"{prm} explicitness →")
            if k != "halluc":
                ax.set_ylim(0, 1)
            ax.grid(alpha=.3)
        axes[0].legend(fontsize=7)
        fig.suptitle(f"{prm.capitalize()} ablation — {MODEL_LABELS.get(model, model)}")
        plt.tight_layout()
        plt.savefig(out / f"ablation_{domain}_{model}_{prm}.pdf", bbox_inches="tight")
        plt.close(fig)
