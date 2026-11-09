"""Training diagnostics and cross-run figures (reads steps.csv / epochs.csv of each run).

Per run:     dashboard (CE, compliance, hallucination, λ trajectory, semantic, effective
             contributions λ·L), CE by loss phase.
Across runs: compliance–hallucination Pareto plot; seed-aggregated validation curves.
"""
from __future__ import annotations

from pathlib import Path
from typing import List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

C = {"ce": "#2166ac", "tok": "#1b9e77", "phr": "#d95f02", "cfg": "#7570b3", "bar": "#c51b7d",
     "slot": "#1f78b4", "da": "#33a02c", "hall": "#e31a1c", "bert": "#636363"}
PHASE_BG = {"P1:CE": "#e8f0fa", "P2:CE+tok": "#e7f5ee", "P3:all": "#fdf0e6"}

plt.rcParams.update({"figure.dpi": 150, "font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.3})


def _smooth(y, frac=50):
    y = np.asarray(y, float)
    w = max(1, len(y) // frac)
    return np.convolve(y, np.ones(w) / w, mode="valid"), w


def _col(df, name):
    return df[name] if name in df else pd.Series(np.zeros(len(df)))


def load_run(run_dir: Path) -> dict:
    d = {"dir": run_dir, "name": "/".join(run_dir.parts[-3:])}
    for k in ("steps", "epochs"):
        p = run_dir / f"{k}.csv"
        if p.exists() and p.stat().st_size:
            d[k] = pd.read_csv(p)
    return d


def dashboard(run: dict, out: Path) -> None:
    s, e = run.get("steps"), run.get("epochs")
    if s is None and e is None:
        return
    fig, ax = plt.subplots(2, 3, figsize=(17, 9))
    fig.suptitle(f"SCOPE training — {run['name']}")
    if s is not None:
        x = s.global_step.values
        y, w = _smooth(_col(s, "train_loss_ce"))
        ax[0, 0].plot(x, _col(s, "train_loss_ce"), alpha=.2, color=C["ce"])
        ax[0, 0].plot(x[w - 1:], y, color=C["ce"])
        if "phase" in s:
            for ph, grp in s.groupby("phase"):
                ax[0, 0].axvspan(grp.global_step.min(), grp.global_step.max(), color=PHASE_BG.get(ph, "#f5f5f5"),
                                 alpha=.6, zorder=0, label=ph)
            ax[0, 0].legend(fontsize=7)
        ax[0, 0].set(title="L_CE", xlabel="optimizer step")
        for k in ("ce", "tok", "phr", "cfg"):
            if f"lambda_{k}" in s:
                ax[1, 0].plot(x, s[f"lambda_{k}"], color=C[k], label=f"λ_{k}")
        ax[1, 0].set(title="λ (curriculum × GradNorm)", xlabel="optimizer step")
        ax[1, 0].legend(fontsize=7)
        for k in ("ce", "tok", "phr", "cfg"):
            if f"contrib_{k}" in s:
                yy, w2 = _smooth(s[f"contrib_{k}"])
                ax[1, 2].plot(x[w2 - 1:], yy, color=C[k], label=f"λ·L_{k}")
        ax[1, 2].set(title="Effective contributions", xlabel="optimizer step")
        ax[1, 2].legend(fontsize=7)
    if e is not None:
        ep = e.epoch.values
        for k, c in (("val_c_tok", "tok"), ("val_c_phr", "phr"), ("val_c_cfg", "cfg"), ("val_c_bar", "bar")):
            if k in e:
                ax[0, 1].plot(ep, e[k], "o-", color=C[c], label=k[4:], lw=2.5 if c == "bar" else 1.5)
        ax[0, 1].set(title="Validation compliance", xlabel="epoch", ylim=(0, 1))
        ax[0, 1].legend(fontsize=7)
        if "val_halluc_pct" in e:
            ax[0, 2].plot(ep, e.val_halluc_pct, "o-", color=C["hall"])
        ax[0, 2].set(title="Validation hallucination %", xlabel="epoch")
        for k, c in (("val_slot_f1", "slot"), ("val_da_f1", "da")):
            if k in e:
                ax[1, 1].plot(ep, e[k], "o-", color=C[c], label=k[4:])
        ax[1, 1].set(title="Validation semantic", xlabel="epoch", ylim=(0, 1))
        ax[1, 1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / f"dashboard_{run['name'].replace('/', '_')}.pdf", bbox_inches="tight")
    plt.close(fig)


def pareto(runs: List[dict], out: Path) -> None:
    pts = []
    for r in runs:
        e = r.get("epochs")
        if e is not None and "val_c_bar" in e:
            for _, row in e.iterrows():
                pts.append((row.val_c_bar, row.get("val_halluc_pct", 0.0), int(row.epoch), r["name"]))
    if not pts:
        return
    arr = np.array([[p[0], p[1]] for p in pts])
    eff = np.array([not np.any((arr[:, 0] >= a[0]) & (arr[:, 1] <= a[1]) & ((arr[:, 0] > a[0]) | (arr[:, 1] < a[1])))
                    for a in arr])
    fig, ax = plt.subplots(figsize=(8, 6))
    names = list(dict.fromkeys(p[3] for p in pts))
    cmap = plt.get_cmap("tab10")
    for i, n in enumerate(names):
        sel = [j for j, p in enumerate(pts) if p[3] == n]
        ax.scatter(arr[sel, 0], arr[sel, 1], color=cmap(i % 10), s=25, alpha=.6, label=n)
    front = arr[eff][np.argsort(arr[eff][:, 0])]
    ax.plot(front[:, 0], front[:, 1], "k--", lw=1, label="Pareto front")
    ax.set(xlabel="validation C̄ (↑)", ylabel="hallucination % (↓)", title="Compliance vs hallucination")
    ax.invert_yaxis()
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "pareto_cbar_vs_halluc.pdf", bbox_inches="tight")
    plt.close(fig)


def seed_curves(runs: List[dict], out: Path, metric: str = "val_c_bar") -> None:
    """Mean ± std across seeds of a validation metric, one line per (model, method)."""
    groups = {}
    for r in runs:
        e = r.get("epochs")
        if e is None or metric not in e:
            continue
        key = "/".join(r["dir"].parts[-3:-1])
        groups.setdefault(key, []).append(e.drop_duplicates("epoch", keep="last").set_index("epoch")[metric])
    if not groups:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    for key, series in groups.items():
        df = pd.concat(series, axis=1)
        m, sd = df.mean(1), df.std(1).fillna(0)
        ax.plot(df.index, m, "o-", label=f"{key} (n={df.shape[1]})")
        ax.fill_between(df.index, m - sd, m + sd, alpha=.2)
    ax.set(xlabel="epoch", ylabel=metric, title=f"{metric} across seeds (mean ± std)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / f"seeds_{metric}.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_all(paths: List[Path], out: Path) -> int:
    out.mkdir(parents=True, exist_ok=True)
    runs = [load_run(p) for p in paths]
    runs = [r for r in runs if "steps" in r or "epochs" in r]
    for r in runs:
        dashboard(r, out)
    pareto(runs, out)
    for m in ("val_c_bar", "val_c_cfg", "val_slot_f1"):
        seed_curves(runs, out, m)
    return len(runs)
