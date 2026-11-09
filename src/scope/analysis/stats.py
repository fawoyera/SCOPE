"""Significance testing: SCOPE (or any reference method) vs every baseline.

Two levels of evidence are reported for each (model, baseline, metric):

Seed level  (unit = training run; answers "is the gain robust to the training seed?")
  * mean ± std and a 95% t-interval for each method over seeds
  * Welch's t-test (unequal variances) between the two sets of seed means
  * paired t-test on seed-matched runs (same seed for both methods), when ≥2 pairs;
    for a deterministic baseline with a single run, a one-sample t-test of the
    reference's seed means against the baseline value
  With 3 seeds these tests have little power — report them, but do not rely on them alone.

Example level  (unit = test utterance; answers "is the gain consistent across inputs?")
  * per example, average the metric over each method's seeds, then take the paired
    difference d_i = ref_i − base_i on the shared test examples
  * 95% paired bootstrap CI of mean(d) (100,000 resamples)
  * two-sided paired sign-flip permutation test on mean(d) (100,000 permutations)
  * paired t-test on d (large-n parametric check)
  * Wilcoxon signed-rank test (zero differences split), and Cohen's d_z = mean(d)/sd(d)

Multiple comparisons: Holm–Bonferroni within each model across all (baseline × metric)
tests of the same family.  Stars in the LaTeX summary use the Holm-adjusted
permutation p-value: * p<0.05, ** p<0.01, *** p<0.001.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import warnings

import numpy as np
import pandas as pd
from scipy import stats as st

from .results import method_label, model_label, HIGHER_BETTER, METHOD_LABELS, METRIC_KEYS, MODEL_LABELS, SCALE

N_RESAMPLES = 100_000   # min attainable p ≈ 1e-5, so Holm over ~50 tests can still reach p<.001
CHUNK = 5_000


def t_interval(x: np.ndarray, conf: float = 0.95):
    x = np.asarray(x, float)
    n = len(x)
    if n < 2:
        return (float(x.mean()) if n else np.nan,) * 2
    h = st.t.ppf(0.5 + conf / 2, n - 1) * x.std(ddof=1) / np.sqrt(n)
    return float(x.mean() - h), float(x.mean() + h)


def paired_bootstrap_ci(d: np.ndarray, rng: np.random.Generator, conf: float = 0.95):
    means = np.concatenate([d[rng.integers(0, len(d), size=(min(CHUNK, N_RESAMPLES - i), len(d)))].mean(1)
                            for i in range(0, N_RESAMPLES, CHUNK)])
    return float(np.quantile(means, (1 - conf) / 2)), float(np.quantile(means, 0.5 + conf / 2))


def sign_flip_pvalue(d: np.ndarray, rng: np.random.Generator) -> float:
    if np.allclose(d, 0):
        return 1.0
    obs = abs(d.mean())
    hits = 0
    for i in range(0, N_RESAMPLES, CHUNK):
        signs = rng.choice([-1.0, 1.0], size=(min(CHUNK, N_RESAMPLES - i), len(d)))
        hits += int((np.abs(signs @ d / len(d)) >= obs - 1e-12).sum())
    return float((1 + hits) / (N_RESAMPLES + 1))


def holm(p: pd.Series) -> pd.Series:
    """Holm–Bonferroni step-down adjusted p-values (NaN-safe)."""
    p = p.astype(float)
    ok = p.notna()
    vals = p[ok].values
    order = np.argsort(vals)
    m = len(vals)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * vals[i]))
        adj[i] = running
    out = pd.Series(np.nan, index=p.index)
    out[ok] = adj
    return out


def method_summary(runs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model, method), df in runs.groupby(["model", "method"]):
        for k in METRIC_KEYS:
            v = df[k].dropna().astype(float).values * SCALE[k]
            if not len(v):
                continue
            lo, hi = t_interval(v)
            rows.append({"model": model, "method": method, "metric": k, "n_seeds": len(v),
                         "mean": v.mean(), "std": v.std(ddof=1) if len(v) > 1 else np.nan,
                         "sem": v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else np.nan,
                         "ci95_low": lo, "ci95_high": hi})
    return pd.DataFrame(rows)


def compare(runs: pd.DataFrame, examples: pd.DataFrame, reference: str = "scope",
            baselines: Optional[List[str]] = None, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    warnings.filterwarnings("ignore", category=RuntimeWarning, module="scipy")
    rows = []
    for model in sorted(runs.model.unique()):
        R = runs[(runs.model == model) & (runs.method == reference)]
        if R.empty:
            continue
        # default family: main-grid methods only; ablation variants are tested per group by ablate-report
        bases = baselines or sorted(m for m in runs[runs.model == model].method.unique()
                                    if m != reference and "__" not in m)
        for base in bases:
            B = runs[(runs.model == model) & (runs.method == base)]
            if B.empty:
                continue
            for k in METRIC_KEYS:
                r = R.set_index("seed")[k].dropna().astype(float) * SCALE[k]
                b = B.set_index("seed")[k].dropna().astype(float) * SCALE[k]
                if r.empty or b.empty:
                    continue
                rec = {"model": model, "reference": reference, "baseline": base, "metric": k,
                       "higher_is_better": HIGHER_BETTER[k], "ref_mean": r.mean(), "base_mean": b.mean(),
                       "delta": r.mean() - b.mean(), "n_ref_seeds": len(r), "n_base_seeds": len(b)}
                # ── seed level ────────────────────────────────────────────
                if len(r) >= 2 and len(b) >= 2:
                    rec["welch_t"], rec["welch_p"] = st.ttest_ind(r, b, equal_var=False)
                common = sorted(set(r.index) & set(b.index))
                if len(common) >= 2:
                    rec["paired_t"], rec["paired_t_p"] = st.ttest_rel(r[common], b[common])
                elif len(b) == 1 and len(r) >= 2:
                    rec["paired_t"], rec["paired_t_p"] = st.ttest_1samp(r, b.iloc[0])
                if len(r) >= 2:
                    d_seed = (r - b.mean()).values
                    rec["delta_ci95_seed_low"], rec["delta_ci95_seed_high"] = t_interval(d_seed) \
                        if len(b) == 1 else _welch_ci(r.values, b.values)
                # ── example level ────────────────────────────────────────
                if not examples.empty and k in examples.columns:
                    E = examples[(examples.model == model) & examples.method.isin([reference, base])]
                    piv = (E.groupby(["pair_id", "method"])[k].mean().unstack("method").dropna())
                    if reference in piv and base in piv and len(piv) > 1:
                        d = (piv[reference] - piv[base]).values * SCALE[k]
                        rec["n_examples"] = len(d)
                        rec["delta_example"] = d.mean()
                        rec["delta_ci95_low"], rec["delta_ci95_high"] = paired_bootstrap_ci(d, rng)
                        rec["perm_p"] = sign_flip_pvalue(d, rng)
                        rec["example_t_p"] = st.ttest_1samp(d, 0.0).pvalue if d.std() > 0 else 1.0
                        nz = d[d != 0]
                        rec["wilcoxon_p"] = (st.wilcoxon(d, zero_method="zsplit").pvalue if len(nz) else 1.0)
                        rec["cohens_dz"] = d.mean() / d.std(ddof=1) if d.std(ddof=1) > 0 else 0.0
                rows.append(rec)
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    for col in ("perm_p", "wilcoxon_p", "example_t_p", "welch_p", "paired_t_p"):
        if col in out:
            out[col + "_holm"] = out.groupby("model")[col].transform(holm)
    out["ref_better"] = np.where(out.higher_is_better, out.delta > 0, out.delta < 0)
    return out


def _welch_ci(a: np.ndarray, b: np.ndarray, conf: float = 0.95):
    va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    se = np.sqrt(va + vb)
    if se == 0:
        return (float(a.mean() - b.mean()),) * 2
    dof = (va + vb) ** 2 / (va ** 2 / (len(a) - 1) + vb ** 2 / (len(b) - 1))
    h = st.t.ppf(0.5 + conf / 2, dof) * se
    d = a.mean() - b.mean()
    return float(d - h), float(d + h)


def stars(p: float) -> str:
    if p is None or np.isnan(p):
        return ""
    return "***" if p < 1e-3 else "**" if p < 1e-2 else "*" if p < 0.05 else ""


def latex_significance(cmp: pd.DataFrame, reference: str = "scope", p_col: str = "perm_p_holm") -> str:
    """Δ(reference − baseline) per metric with stars from the Holm-adjusted p-value and the
    paired-bootstrap 95% CI underneath."""
    if cmp.empty:
        return "% no comparisons"
    lines = [r"\begin{table*}[ht]", r"\centering", r"\scriptsize",
             r"\caption{Paired comparison of " + method_label(reference).replace(r"\textbf", "") +
             r" against each baseline on the test set: $\Delta$ = reference $-$ baseline (Hall.\ in points), "
             r"example-level paired bootstrap 95\% CI in brackets. Stars: Holm-adjusted two-sided paired "
             r"permutation test, * $p<.05$, ** $p<.01$, *** $p<.001$.}",
             r"\label{tab:significance}", r"\setlength{\tabcolsep}{2.5pt}",
             r"\begin{tabular}{ll" + "c" * len(METRIC_KEYS) + "}", r"\hline",
             r"Model & vs. & \Ctok & \Cphr & \Ccfg & \Cbar & Slot-F1 & DA-F1 & Hall.\% & BS$_\mathrm{D}$ & BS$_\mathrm{G}$ \\",
             r"\hline"]
    for model in cmp.model.unique():
        sub = cmp[cmp.model == model]
        bases = list(dict.fromkeys(sub.baseline))
        for i, base in enumerate(bases):
            cells = []
            for k in METRIC_KEYS:
                r = sub[(sub.baseline == base) & (sub.metric == k)]
                if r.empty:
                    cells.append("---")
                    continue
                r = r.iloc[0]
                prec = 1 if k == "halluc" else 3
                delta = r.get("delta_example", r["delta"])
                s = stars(r.get(p_col, np.nan))
                ci = ""
                if pd.notna(r.get("delta_ci95_low", np.nan)):
                    ci = rf"\\[-1pt]{{\tiny[{r['delta_ci95_low']:+.{prec}f}, {r['delta_ci95_high']:+.{prec}f}]}}"
                sup = f"$^{{{s}}}$" if s else ""
                cells.append(rf"\makecell{{{delta:+.{prec}f}{sup}{ci}}}")
            name = (model_label(model) if i == 0 else "")
            lines.append(f"{name} & {method_label(base)} & " + " & ".join(cells) + r" \\")
        lines.append(r"\hline")
    lines += [r"\end{tabular}", r"\end{table*}", "% needs \\usepackage{makecell}"]
    return "\n".join(lines)
