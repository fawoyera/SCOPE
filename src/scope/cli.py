"""Command-line interface.

    scope list                                   models / methods / domains available
    scope experiment --model M --method X --seed S [--set k=v ...]     one run
    scope run --models M1 M2 --methods all --seeds 42 123 456 [--launcher slurm]  full grid
    scope table  [--models ...] [--methods ...]  LaTeX/Markdown/CSV results table (mean ± std)
    scope stats  [--reference scope]             CIs, t-tests, paired bootstrap / permutation tests
    scope plot   [--runs-dir results/atc]        training dashboards, Pareto plot, seed curves
    scope ablate --models M --groups components recipe ...   ablation suite (configs/ablations/scope.yaml)
    scope ablate-report                          ablation tables (Δ vs full, Holm-corrected), sensitivity plots
    scope bertscore                              add BERTScore to finished runs without regenerating
    scope latency [--methods ...]                batch-1 latency benchmark of finished runs
    scope compute                                GPU-hours, FLOPs, phase breakdown, latency tables
    scope bert-mlm --domain atc                  build the domain-adapted BERTScore encoder (once)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _add_common(p):
    p.add_argument("--domain", default="atc")
    p.add_argument("--output-root", default="results")
    p.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE",
                   help="config override, e.g. --set train.lr=1e-5 --set logging.wandb.mode=offline")


def cmd_list(a):
    from .config import available
    for kind in ("models", "methods", "domains"):
        print(f"{kind:8s}: {', '.join(available(kind))}")


def cmd_experiment(a):
    from .config import load_config
    from .pipeline import run_experiment
    ov = [f"run.output_root={a.output_root}"] + a.overrides
    cfg = load_config(a.model, a.method, a.domain, a.seed, ov)
    if a.print_config:
        import yaml
        print(yaml.safe_dump(cfg.to_dict(), sort_keys=False))
        return 0
    run_experiment(cfg, force=a.force, discard=a.discard)
    return 0


def cmd_run(a):
    from .config import available
    from .runner import ALL_METHODS, plan, run_local, run_slurm
    methods = ALL_METHODS if a.methods == ["all"] else a.methods
    models = available("models") if a.models == ["all"] else a.models
    for m in models:
        if m not in available("models"):
            sys.exit(f"Unknown model '{m}'. Available: {available('models')}")
    for m in methods:
        if m not in available("methods"):
            sys.exit(f"Unknown method '{m}'. Available: {available('methods')}")
    jobs = plan(a.domain, models, methods, a.seeds, a.output_root, a.overrides, a.all_seeds_for_deterministic)
    print(f"Planned {len(jobs)} runs:")
    for j in jobs:
        print(f"  {j.name}" + (f"   (after {j.dep.name})" if j.dep else ""))
    if a.launcher == "slurm":
        return run_slurm(jobs, a.slurm_config, a.force, a.dry_run, a.discard)
    return run_local(jobs, a.force, a.dry_run, a.discard)


def cmd_table(a):
    from .analysis.results import collect_runs
    from .analysis.table import build_tables
    runs = collect_runs(a.output_root, a.domain)
    if runs.empty:
        sys.exit(f"No finished runs under {a.output_root}/{a.domain}")
    summary, tex, md = build_tables(runs, a.domain, a.models, a.methods, a.bold_exclude)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"results_{a.domain}.tex").write_text(tex)
    (out / f"results_{a.domain}.md").write_text(md)
    summary.to_csv(out / f"results_{a.domain}_summary.csv", index=False)
    runs.to_csv(out / f"results_{a.domain}_runs.csv", index=False)
    print(md)
    print(f"\nWrote {out}/results_{a.domain}.tex/.md/_summary.csv/_runs.csv")
    return 0


def cmd_stats(a):
    from .analysis.results import collect_examples, collect_runs
    from .analysis.stats import compare, latex_significance, method_summary
    runs = collect_runs(a.output_root, a.domain)
    if runs.empty:
        sys.exit(f"No finished runs under {a.output_root}/{a.domain}")
    if a.models:
        runs = runs[runs.model.isin(a.models)]
    ex = collect_examples(a.output_root, a.domain, a.models)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    ms = method_summary(runs)
    ms.to_csv(out / f"method_ci_{a.domain}.csv", index=False)
    cmp = compare(runs, ex, a.reference, a.baselines, seed=a.seed)
    cmp.to_csv(out / f"significance_{a.domain}.csv", index=False)
    (out / f"significance_{a.domain}.tex").write_text(latex_significance(cmp, a.reference))
    if not cmp.empty:
        import pandas as pd
        show = cmp[["model", "baseline", "metric", "delta", "delta_ci95_low", "delta_ci95_high",
                    "perm_p_holm", "wilcoxon_p_holm", "welch_p", "paired_t_p", "cohens_dz"]].copy()
        with pd.option_context("display.width", 200, "display.max_rows", 500, "display.float_format", "{:.4g}".format):
            print(show.to_string(index=False))
    print(f"\nWrote {out}/method_ci_{a.domain}.csv, significance_{a.domain}.csv/.tex")
    return 0


def cmd_plot(a):
    from .analysis.plots import plot_all
    root = Path(a.runs_dir or Path(a.output_root) / a.domain)
    import re
    paths = [root] if (root / "epochs.csv").exists() else sorted(
        p.parent for p in root.glob("**/epochs.csv") if re.fullmatch(r"s\d+", p.parent.name))
    n = plot_all(paths, Path(a.out))
    print(f"Plotted {n} run(s) → {a.out}")
    return 0


def _ablation_variants(spec, a):
    sel = list(a.variants or [])
    for g in a.groups or []:
        if g not in spec["groups"]:
            sys.exit(f"Unknown group {g!r}; available: {list(spec['groups'])}")
        sel += spec["groups"][g]["variants"]
    return list(dict.fromkeys(sel)) or None


def cmd_ablate(a):
    from .runner import load_ablation_spec, plan_ablation, run_local, run_slurm
    spec = load_ablation_spec(a.spec)
    jobs = plan_ablation(spec, a.domain, a.models, a.seeds, a.output_root, a.overrides, _ablation_variants(spec, a))
    print(f"Planned {len(jobs)} runs:")
    for j in jobs:
        print(f"  {j.name}" + (f"   (after {j.dep.name})" if j.dep else ""))
    if a.launcher == "slurm":
        return run_slurm(jobs, a.slurm_config, a.force, a.dry_run, a.discard)
    return run_local(jobs, a.force, a.dry_run, a.discard)


def cmd_ablate_report(a):
    from .analysis.ablation import run
    from .runner import load_ablation_spec
    run(load_ablation_spec(a.spec), a.output_root, a.domain, a.out, a.models, a.groups)
    return 0


def cmd_latency(a):
    from .config import run_dir_for
    from .evaluation.latency import benchmark_run
    from .pipeline import is_complete
    root = Path(a.output_root) / a.domain
    from .config import iter_run_dirs
    dirs = [f.parent for *_, f in iter_run_dirs(a.output_root, a.domain, "config.yaml")]
    dirs = [d for d in dirs if (not a.models or d.parts[-3] in a.models)
            and (not a.methods or d.parts[-2] in a.methods)
            and (not a.seeds or int(d.parts[-1][1:]) in a.seeds) and is_complete(d)]
    if a.one_seed:                                  # latency does not depend on the training seed
        seen, keep = set(), []
        for d in dirs:
            if (d.parts[-3], d.parts[-2]) not in seen:
                seen.add((d.parts[-3], d.parts[-2]))
                keep.append(d)
        dirs = keep
    for d in dirs:
        if (d / "latency.json").exists() and not a.force:
            print(f"✓ {d} (latency.json exists)")
            continue
        benchmark_run(d, a.n, a.warmup, not a.no_ttft)
    return 0


def cmd_bertscore(a):
    from .evaluation.rescore import rescore_all
    return rescore_all(a.output_root, a.domain, a.models, a.methods, a.domain_model, a.general_model,
                       a.device, a.force)


def cmd_compute(a):
    from .analysis.compute import run
    run(a.output_root, a.domain, a.out, a.reference, a.models, a.methods)
    return 0


def cmd_bert_mlm(a):
    from .metrics.bert_mlm import finetune_bert_mlm
    finetune_bert_mlm(a.domain, a.base, a.out or f"models/bert-{a.domain}-mlm", a.epochs, a.batch_size, a.lr)
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(prog="scope", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("list").set_defaults(fn=cmd_list)

    p = sub.add_parser("experiment", help="run one (model, method, seed)")
    _add_common(p)
    p.add_argument("--model", required=True)
    p.add_argument("--method", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--force", action="store_true",
                   help="re-run finished runs; the previous attempt is kept as s<seed>.<timestamp>")
    p.add_argument("--discard", action="store_true", help="with --force: delete the previous attempt instead")
    p.add_argument("--print-config", action="store_true")
    p.set_defaults(fn=cmd_experiment)

    p = sub.add_parser("run", help="run a grid of experiments")
    _add_common(p)
    p.add_argument("--models", nargs="+", required=True)
    p.add_argument("--methods", nargs="+", default=["all"])
    p.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 456])
    p.add_argument("--launcher", choices=["local", "slurm"], default="local")
    p.add_argument("--slurm-config", default="gilbreth")
    p.add_argument("--all-seeds-for-deterministic", action="store_true",
                   help="also repeat vanilla/gcd for every seed (they are deterministic)")
    p.add_argument("--force", action="store_true",
                   help="re-run finished runs; the previous attempt is kept as s<seed>.<timestamp>")
    p.add_argument("--discard", action="store_true", help="with --force: delete the previous attempt instead")
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(fn=cmd_run)

    p = sub.add_parser("table", help="results table (mean ± std over seeds)")
    _add_common(p)
    p.add_argument("--models", nargs="+")
    p.add_argument("--methods", nargs="+")
    p.add_argument("--bold-exclude", nargs="*", default=["vanilla"])
    p.add_argument("--out", default="tables")
    p.set_defaults(fn=cmd_table)

    p = sub.add_parser("stats", help="confidence intervals and significance tests")
    _add_common(p)
    p.add_argument("--reference", default="scope")
    p.add_argument("--baselines", nargs="+")
    p.add_argument("--models", nargs="+")
    p.add_argument("--seed", type=int, default=0, help="RNG seed for bootstrap/permutation")
    p.add_argument("--out", default="tables")
    p.set_defaults(fn=cmd_stats)

    p = sub.add_parser("plot", help="training diagnostics")
    _add_common(p)
    p.add_argument("--runs-dir", default="")
    p.add_argument("--out", default="plots")
    p.set_defaults(fn=cmd_plot)

    p = sub.add_parser("ablate", help="run an ablation suite (configs/ablations/<spec>.yaml)")
    _add_common(p)
    p.add_argument("--spec", default="scope")
    p.add_argument("--models", nargs="+", required=True)
    p.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 456])
    p.add_argument("--groups", nargs="+", help="components recipe data grpo lambda compute inference")
    p.add_argument("--variants", nargs="+")
    p.add_argument("--launcher", choices=["local", "slurm"], default="local")
    p.add_argument("--slurm-config", default="gilbreth")
    p.add_argument("--force", action="store_true",
                   help="re-run finished runs; the previous attempt is kept as s<seed>.<timestamp>")
    p.add_argument("--discard", action="store_true", help="with --force: delete the previous attempt instead")
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(fn=cmd_ablate)

    p = sub.add_parser("ablate-report", help="ablation tables, significance and sensitivity plots")
    _add_common(p)
    p.add_argument("--spec", default="scope")
    p.add_argument("--models", nargs="+")
    p.add_argument("--groups", nargs="+")
    p.add_argument("--out", default="tables")
    p.set_defaults(fn=cmd_ablate_report)

    p = sub.add_parser("latency", help="batch-1 latency benchmark of finished runs")
    _add_common(p)
    p.add_argument("--models", nargs="+")
    p.add_argument("--methods", nargs="+")
    p.add_argument("--seeds", nargs="+", type=int)
    p.add_argument("--n", type=int, default=200, help="test utterances to time")
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--no-ttft", action="store_true", help="skip time-to-first-token measurement")
    p.add_argument("--all-seeds", dest="one_seed", action="store_false",
                   help="benchmark every seed (default: first seed per model/method)")
    p.add_argument("--force", action="store_true",
                   help="re-run finished runs; the previous attempt is kept as s<seed>.<timestamp>")
    p.add_argument("--discard", action="store_true", help="with --force: delete the previous attempt instead")
    p.set_defaults(fn=cmd_latency)

    p = sub.add_parser("bertscore", help="add BERTScore to finished runs (no regeneration)")
    _add_common(p)
    p.add_argument("--models", nargs="+")
    p.add_argument("--methods", nargs="+")
    p.add_argument("--domain-model", default=None, help="override eval.bertscore.domain_model")
    p.add_argument("--general-model", default=None, help="override eval.bertscore.general_model")
    p.add_argument("--device", default=None, help="cpu | cuda")
    p.add_argument("--force", action="store_true", help="recompute even if present")
    p.set_defaults(fn=cmd_bertscore)

    p = sub.add_parser("compute", help="training-cost and latency tables")
    _add_common(p)
    p.add_argument("--reference", default="scope")
    p.add_argument("--models", nargs="+")
    p.add_argument("--methods", nargs="+")
    p.add_argument("--out", default="tables")
    p.set_defaults(fn=cmd_compute)

    p = sub.add_parser("bert-mlm", help="domain-adapt the BERTScore encoder")
    _add_common(p)
    p.add_argument("--base", default="bert-base-uncased")
    p.add_argument("--out", default="")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=2e-5)
    p.set_defaults(fn=cmd_bert_mlm)

    a = ap.parse_args(argv)
    return a.fn(a) or 0


if __name__ == "__main__":
    sys.exit(main())
