"""Compute-cost and latency analysis (``scope compute``).

Inputs per run: ``compute.json`` (written by every experiment) and, if
``scope latency`` was run, ``latency.json``.

Outputs
  compute_<d>_training.csv / .tex   GPU-h (own and cumulative incl. the SFT run a method
                                    starts from), wall-clock, peak memory, est. PFLOPs,
                                    throughput — mean ± std over seeds
  compute_<d>_breakdown.csv         seconds and share per training phase (rollouts, rewards, …)
  compute_<d>_inference.csv / .tex  amortised test-time cost and batch-1 latency
                                    (mean, p50, p95, ms/token, TTFT, grammar-mask share)
  latency_<d>_paired.csv            per-utterance paired latency ratio vs a reference method:
                                    geometric-mean ratio with 95% bootstrap CI + Wilcoxon p
  compute_<d>_breakdown.pdf, latency_<d>_box.pdf
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
from scipy import stats as st

from ..config import iter_run_dirs
from .results import method_label, model_label, METHOD_LABELS, MODEL_LABELS, order

STARTS_FROM = {"dpo": "sft", "gcd_sft": "sft", "gcd_cfg_sft": "sft"}
PHASES = ["ce_tok_fwd_bwd", "rollout_generation", "reward_scoring", "policy_gradient_fwd_bwd", "reference_fwd",
          "gradnorm", "dpo_pair_sampling", "dpo_pair_scoring", "dpo_reference_logps", "dpo_fwd_bwd",
          "optimizer_step", "validation"]


def _load(root: Path, domain: str, name: str):
    for model, method, seed, f in iter_run_dirs(root, domain, name):
        yield model, method, seed, json.load(open(f))


def collect(root: str, domain: str):
    root = Path(root)
    tr, br, inf = [], [], []
    for model, method, seed, c in _load(root, domain, "compute.json"):
        hw = c.get("hardware", {})
        t = c.get("training")
        if t:
            tr.append({"model": model, "method": method, "seed": seed, "gpu": hw.get("gpu"),
                       "n_gpus": hw.get("n_gpus"), "params": c.get("params"),
                       "gpu_hours": t.get("gpu_hours", 0.0), "wall_hours": t["wall_seconds"] / 3600,
                       "train_loop_hours": t.get("train_loop_seconds", 0) / 3600,
                       "peak_mem_gb": t.get("peak_mem_gb_max_device", np.nan),
                       "pflops": t.get("flops_total", 0.0) / 1e15, "optimizer_steps": t.get("optimizer_steps"),
                       "samples_per_second": t.get("samples_per_second")})
            secs = t.get("seconds", {})
            tot = sum(secs.values()) or 1.0
            for ph, s in secs.items():
                br.append({"model": model, "method": method, "seed": seed, "phase": ph, "seconds": s,
                           "share": s / tot, "tokens": t.get("tokens", {}).get(ph, 0),
                           "pflops": t.get(f"flops_{ph}", 0.0) / 1e15})
        i = c.get("inference")
        if i:
            inf.append({"model": model, "method": method, "seed": seed, "decoding": i.get("decoding"),
                        "num_beams": i.get("num_beams"), "test_ms_per_example": i.get("ms_per_example_amortised"),
                        "test_ms_per_token": i.get("ms_per_token_amortised"),
                        "tokens_per_example": i.get("tokens_per_example"), "test_mask_fraction": i.get("mask_fraction"),
                        "test_eval_batch": i.get("batch_size"), "inference_tflops": i.get("flops_estimate", 0) / 1e12})
    lat = []
    for model, method, seed, l in _load(root, domain, "latency.json"):
        lat.append({"model": model, "method": method, "seed": seed, "n": l["n"],
                    **{f"lat_{k}": v for k, v in l["latency_ms"].items()},
                    "ms_per_token": l["ms_per_token"]["mean"], "ttft_p50": l.get("ttft_ms", {}).get("p50", np.nan),
                    "mask_fraction": l.get("mask_fraction", 0.0), "per_example": [x["latency_ms"] for x in l["per_example"]]})
    return pd.DataFrame(tr), pd.DataFrame(br), pd.DataFrame(inf), pd.DataFrame(lat)


def _ms(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    g = df.groupby(["model", "method"])
    out = g[cols].agg(["mean", "std"])
    out.columns = [f"{a}_{b}" for a, b in out.columns]
    out["n_seeds"] = g.size()
    return out.reset_index()


def training_table(tr: pd.DataFrame) -> pd.DataFrame:
    if tr.empty:
        return tr
    tr = tr.copy()
    # cumulative cost: a method that starts from SFT also pays for that SFT run (same seed)
    own = tr.set_index(["model", "method", "seed"])["gpu_hours"]
    wall = tr.set_index(["model", "method", "seed"])["wall_hours"]
    tr["gpu_hours_cumulative"] = [g + own.get((m, STARTS_FROM[me], s), 0.0) if me in STARTS_FROM else g
                                  for m, me, s, g in zip(tr.model, tr.method, tr.seed, tr.gpu_hours)]
    tr["wall_hours_cumulative"] = [w + wall.get((m, STARTS_FROM[me], s), 0.0) if me in STARTS_FROM else w
                                   for m, me, s, w in zip(tr.model, tr.method, tr.seed, tr.wall_hours)]
    return _ms(tr, ["gpu_hours", "gpu_hours_cumulative", "wall_hours", "wall_hours_cumulative", "peak_mem_gb",
                    "pflops", "samples_per_second", "optimizer_steps"])


def add_inference_only(train_summary: pd.DataFrame, inf: pd.DataFrame, tr: pd.DataFrame) -> pd.DataFrame:
    """GCD/GCD+SFT rows have no training of their own; carry their cumulative cost."""
    rows = []
    for (model, method), _ in inf.groupby(["model", "method"]):
        if ((train_summary.model == model) & (train_summary.method == method)).any() if not train_summary.empty else False:
            continue
        base = STARTS_FROM.get(method)
        src = tr[(tr.model == model) & (tr.method == base)] if base and not tr.empty else pd.DataFrame()
        rows.append({"model": model, "method": method, "gpu_hours_mean": 0.0,
                     "gpu_hours_cumulative_mean": src.gpu_hours.mean() if len(src) else 0.0,
                     "gpu_hours_cumulative_std": src.gpu_hours.std(ddof=1) if len(src) > 1 else np.nan,
                     "wall_hours_cumulative_mean": src.wall_hours.mean() if len(src) else 0.0,
                     "n_seeds": len(src) or 1})
    return pd.concat([train_summary, pd.DataFrame(rows)], ignore_index=True) if rows else train_summary


def paired_latency(lat: pd.DataFrame, reference: str = "scope", n_boot: int = 20000, seed: int = 0) -> pd.DataFrame:
    """Per-utterance latency ratio baseline/reference on the same utterances (first seed of each)."""
    rng = np.random.default_rng(seed)
    out = []
    for model in lat.model.unique() if not lat.empty else []:
        L = lat[lat.model == model].sort_values("seed").drop_duplicates("method")
        if reference not in set(L.method):
            continue
        ref = np.asarray(L[L.method == reference].per_example.iloc[0])
        for _, r in L[L.method != reference].iterrows():
            b = np.asarray(r.per_example)
            n = min(len(ref), len(b))
            if n < 2:
                continue
            lr = np.log(b[:n]) - np.log(ref[:n])
            boots = lr[rng.integers(0, n, (n_boot, n))].mean(1)
            out.append({"model": model, "method": r.method, "reference": reference, "n": n,
                        "geo_mean_ratio": float(np.exp(lr.mean())),
                        "ratio_ci95_low": float(np.exp(np.quantile(boots, .025))),
                        "ratio_ci95_high": float(np.exp(np.quantile(boots, .975))),
                        "median_ms_method": float(np.median(b[:n])), "median_ms_reference": float(np.median(ref[:n])),
                        "wilcoxon_p": float(st.wilcoxon(b[:n], ref[:n]).pvalue) if np.any(b[:n] != ref[:n]) else 1.0})
    return pd.DataFrame(out)


def _f(x, d=2):
    return "---" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{d}f}"


def _cell(mu, sd, d=2):
    if mu is None or (isinstance(mu, float) and np.isnan(mu)):
        return "---"
    return f"{mu:.{d}f}" if sd is None or (isinstance(sd, float) and np.isnan(sd)) else rf"{mu:.{d}f}{{\tiny$\pm${sd:.{d}f}}}"


def latex_tables(trs: pd.DataFrame, infs: pd.DataFrame, lat: pd.DataFrame, models, methods, domain: str) -> str:
    g = lambda df, m, me, c: (df[(df.model == m) & (df.method == me)][c].iloc[0]
                              if not df.empty and c in df and ((df.model == m) & (df.method == me)).any() else np.nan)
    L = [r"\begin{table*}[ht]", r"\centering", r"\scriptsize",
         r"\caption{Training and inference cost on " + domain.upper() + r". GPU-h: own training; "
         r"cum.: including the SFT run the method starts from. PFLOPs: dense-layer estimate "
         r"($6N$ per trained token, $2N$ per generated/scored token). Latency: batch size~1, same "
         r"test utterances for every method, after warm-up; Mask: share of decoding time spent computing grammar masks.}",
         r"\label{tab:compute_" + domain + "}", r"\setlength{\tabcolsep}{3pt}",
         r"\begin{tabular}{ll cccc c ccccc}", r"\hline",
         r"& & \multicolumn{4}{c}{\textbf{Training}} & & \multicolumn{5}{c}{\textbf{Inference (bs=1)}} \\",
         r"\cline{3-6}\cline{8-12}",
         r"Model & Method & GPU-h & GPU-h (cum.) & Peak GB & PFLOPs & & p50 ms & p95 ms & ms/tok & TTFT ms & Mask \% \\",
         r"\hline"]
    for m in models:
        first = True
        for me in methods:
            if not (((trs.model == m) & (trs.method == me)).any() if not trs.empty else False) and \
               not (((lat.model == m) & (lat.method == me)).any() if not lat.empty else False) and \
               not (((infs.model == m) & (infs.method == me)).any() if not infs.empty else False):
                continue
            cells = [_cell(g(trs, m, me, "gpu_hours_mean"), g(trs, m, me, "gpu_hours_std")),
                     _cell(g(trs, m, me, "gpu_hours_cumulative_mean"), g(trs, m, me, "gpu_hours_cumulative_std")),
                     _f(g(trs, m, me, "peak_mem_gb_mean"), 1), _f(g(trs, m, me, "pflops_mean"), 1), "",
                     _f(g(lat, m, me, "lat_p50"), 0), _f(g(lat, m, me, "lat_p95"), 0), _f(g(lat, m, me, "ms_per_token"), 1),
                     _f(g(lat, m, me, "ttft_p50"), 0),
                     _f(100 * g(lat, m, me, "mask_fraction"), 1)]
            L.append(f"{model_label(m) if first else ''} & {method_label(me)} & " + " & ".join(cells) + r" \\")
            first = False
        L.append(r"\hline")
    L += [r"\end{tabular}", r"\end{table*}"]
    return "\n".join(L)


def figures(br: pd.DataFrame, lat: pd.DataFrame, out: Path, domain: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if not br.empty:
        b = br.groupby(["model", "method", "phase"]).seconds.mean().reset_index()
        b["key"] = b.model + " / " + b.method
        piv = b.pivot_table(index="key", columns="phase", values="seconds", fill_value=0) / 3600
        piv = piv[[p for p in PHASES if p in piv.columns] + [c for c in piv.columns if c not in PHASES]]
        ax = piv.plot.barh(stacked=True, figsize=(10, 0.5 * len(piv) + 2), colormap="tab20")
        ax.set(xlabel="hours (mean over seeds)", ylabel="", title="Training time by phase")
        ax.legend(fontsize=7, bbox_to_anchor=(1.01, 1), loc="upper left")
        plt.tight_layout()
        plt.savefig(out / f"compute_{domain}_breakdown.pdf", bbox_inches="tight")
        plt.close()
    if not lat.empty:
        L = lat.sort_values("seed").drop_duplicates(["model", "method"])
        fig, ax = plt.subplots(figsize=(max(6, 0.8 * len(L)), 4))
        ax.boxplot([np.asarray(x) for x in L.per_example], showfliers=False)
        ax.set_xticks(range(1, len(L) + 1))
        ax.set_xticklabels([f"{a}\n{b}" for a, b in zip(L.model, L.method)], fontsize=7)
        ax.set(ylabel="latency per utterance (ms, bs=1)", yscale="log", title="Inference latency")
        plt.tight_layout()
        plt.savefig(out / f"latency_{domain}_box.pdf", bbox_inches="tight")
        plt.close()


def run(root: str, domain: str, out_dir: str, reference: str = "scope",
        models: Optional[List[str]] = None, methods: Optional[List[str]] = None) -> None:
    from ..runner import ALL_METHODS
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tr, br, inf, lat = collect(root, domain)
    if tr.empty and inf.empty:
        raise SystemExit(f"No compute.json under {root}/{domain}")
    trs = add_inference_only(training_table(tr), inf, tr)
    infs = _ms(inf, ["test_ms_per_example", "test_ms_per_token", "tokens_per_example", "test_mask_fraction",
                     "inference_tflops"]) if not inf.empty else inf
    trs.to_csv(out / f"compute_{domain}_training.csv", index=False)
    infs.to_csv(out / f"compute_{domain}_inference_amortised.csv", index=False)
    if not br.empty:
        br.groupby(["model", "method", "phase"])[["seconds", "share", "tokens", "pflops"]].mean().reset_index() \
          .to_csv(out / f"compute_{domain}_breakdown.csv", index=False)
    if not lat.empty:
        lat.drop(columns="per_example").to_csv(out / f"latency_{domain}.csv", index=False)
        paired_latency(lat, reference).to_csv(out / f"latency_{domain}_paired.csv", index=False)
    allm = pd.concat([d[["model", "method"]] for d in (tr, inf, lat) if not d.empty])
    models = models or order(allm.model, list(MODEL_LABELS))
    methods = methods or [m for m in order(allm.method, ALL_METHODS) if "__" not in m]
    (out / f"compute_{domain}.tex").write_text(latex_tables(trs, infs, lat, models, methods, domain))
    figures(br, lat, out, domain)
    with pd.option_context("display.width", 200, "display.float_format", "{:.3g}".format):
        cols = [c for c in ["model", "method", "n_seeds", "gpu_hours_mean", "gpu_hours_cumulative_mean",
                            "peak_mem_gb_mean", "pflops_mean", "samples_per_second_mean"] if c in trs]
        print("Training cost\n", trs[cols].to_string(index=False))
        if not infs.empty:
            print("\nTest-time (amortised)\n", infs[["model", "method", "test_ms_per_example_mean",
                                                    "test_ms_per_token_mean", "test_mask_fraction_mean"]].to_string(index=False))
        if not lat.empty:
            print("\nLatency bs=1\n", lat[["model", "method", "lat_mean", "lat_p50", "lat_p95", "ms_per_token",
                                          "mask_fraction"]].to_string(index=False))
    print(f"\nWrote {out}/compute_{domain}.tex and CSV/PDF files")
