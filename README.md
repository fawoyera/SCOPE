# SCOPE — Structured-Constraint OPtimization for Grammar-informed gEneration

Code for training and evaluating language models that must produce **controlled natural
language** (ATC phraseology per ICAO Doc 4444 / FAA JO 7110.65, maritime IMO SMCP) with
compliance enforced at three levels: lexical (V), phraseological (P) and syntactic (G).

```
L_SCOPE = λ_CE·L_CE + λ_tok·L_tok + λ_phr·L_phr + λ_cfg·L_cfg
          masked CE   OOV mass     GRPO on C_phr   GRPO on C_cfg
```

One command runs the full experimental grid — any Hugging Face model × seeds × the methods

* **Vanilla** — untrained model, task prompt
* **SFT** — L_CE only · **DPO** — preference optimisation from SFT
* **GCD†** / **GCD†+SFT** — vocabulary + bigram constrained decoding
* **GCD** / **GCD+SFT** — grammar-constrained decoding of Geng et al. (EMNLP 2023) with the full CFG
* **SCOPE** (ours)

Further commands produce the results table (mean ± std over seeds), significance tests
(CIs, t-tests, paired bootstrap/permutation, Holm), an 8-group ablation study (including
prompt explicitness), and training-cost / inference-latency analysis.

---

## Install

```bash
conda create -n scope python=3.10 -y && conda activate scope
pip install -e ".[gpu,dev]"          # torch, transformers, lark, transformers-cfg (GCD), bert-score, wandb, ...
bash scripts/install_flash_attn.sh   # optional, used automatically when present
export HF_TOKEN=...                  # gated models (Llama, Gemma) — set in ~/.bashrc, never in a script
wandb login                          # or: --set logging.wandb.mode=offline  (then `wandb sync`)
```

Put the corpus at `data/atc_pairs_enriched.jsonl` (see `data/README.md`).

## Quick start

```bash
scope list                                   # models / methods / domains available

# 0. once per domain: ATC-adapted BERT for BERTScore_D
scope bert-mlm --domain atc

# 1. one run
scope experiment --model llama31_8b --method scope --seed 42

# 2. the paper grid (dependencies handled; finished runs are skipped; re-running resumes)
scope run --models llama31_8b qwen3_8b gpt2_large --methods all --seeds 42 123 456 --launcher slurm

# 3. results
scope table --domain atc                     # tables/results_atc.tex  (mean ± std, best in bold)
scope stats --domain atc --reference scope   # tables/significance_atc.{tex,csv}, method_ci_atc.csv
scope plot  --domain atc --out plots/atc     # training dashboards, Pareto, seed curves

# 4. ablations (8 groups, see below) — e.g. prompt explicitness, or everything
scope ablate --models llama31_8b --groups prompt --launcher slurm
scope ablate --models llama31_8b --launcher slurm
scope ablate-report                          # tables/ablation_atc.tex + sensitivity / prompt figures

# 5. compute and latency
scope latency --n 200 && scope compute      # tables/compute_atc.tex, phase breakdown, latency plots
```

Smoke test on a CPU in ~5 minutes (tiny random GPT-2, meaningless scores):

```bash
python scripts/make_debug_subset.py && python scripts/make_tiny_model.py && pytest -q
scope run --models tiny_debug --methods all --seeds 42 123 \
  --set data.path=data/debug_subset.jsonl --set eval.bertscore.enabled=false \
  --set logging.wandb.mode=offline --set train.epochs=3 --set train.val_max_pairs=8
```

## Methods

| method      | what runs                                                                     | config                    |
|-------------|-------------------------------------------------------------------------------|---------------------------|
| `vanilla`   | untrained model + task prompt, greedy decoding                                | `configs/methods/vanilla.yaml` |
| `sft`       | `L_CE` only — the SCOPE trainer with λ = (1,0,0,0)                            | `sft.yaml`                |
| `dpo`       | from the SFT checkpoint; chosen = target with C_cfg = 1, rejected = SFT sample with C_cfg < 1; β = 0.1 | `dpo.yaml` |
| `gcd`       | untrained model + vocabulary/bigram constrained decoding (GCD†)              | `gcd.yaml`                |
| `gcd_sft`   | SFT checkpoint + GCD†                                                         | `gcd_sft.yaml`            |
| `gcd_cfg`   | untrained model + **grammar-constrained decoding of Geng et al. (EMNLP 2023)** with the full role-gated G | `gcd_cfg.yaml` |
| `gcd_cfg_sft` | SFT checkpoint + GCD (Geng et al.)                                          | `gcd_cfg_sft.yaml`        |
| `scope`     | full objective, loss curriculum (CE → +tok → all), GradNorm                   | `scope.yaml`              |

`dpo`, `gcd_sft` and `gcd_cfg_sft` automatically depend on the `sft` run of the same model and
seed. `vanilla`, `gcd` and `gcd_cfg` are deterministic, so the grid runs them once per model
(`--all-seeds-for-deterministic` to repeat them). Ablations live in `configs/ablations/` (below).

### Grammar-constrained decoding (Geng et al., 2023)

`gcd_cfg*` reproduce the GCD recipe of *Grammar-Constrained Decoding for Structured NLP Tasks
without Finetuning*: at every step an incremental parser returns the tokens that keep the
output in the prefix language of the grammar and all other logits are set to −∞; EOS is
allowed only when the output is a complete sentence; the grammar is input-dependent (the
start symbol is chosen per example from the response role, as their per-datapoint grammars);
greedy or beam search (`--set gcd.num_beams=5`). The parsing backend is the same authors'
`transformers-cfg` (EBNF, incremental pushdown recogniser over a byte trie) instead of the GF
server of the original repo. `src/scope/decoding/cfg_gcd.py`:

* compiles `artefacts/atc/grammar.lark` (strict: catch-all rules removed) to EBNF via Lark's own
  grammar loader, rewriting Lark's left-recursive list rules as repetition and left-factoring
  alternatives (≈400× faster masks on the pilot grammar, identical language);
* uses a model-agnostic token→bytes table (works for Llama-3/3.1, Qwen, Gemma, Mistral, GPT-2
  under transformers 4.x and 5.x — transformers-cfg's own tokenizer dispatch does not);
* derives each row's allowed set from its generated prefix (cached), like `prefix_allowed_tokens_fn`,
  so it stays correct under beam search, where rows are re-ordered.

`tests/test_cfg_gcd.py` checks that the EBNF accepts exactly what Lark accepts on every
target and on corrupted targets, and that every finished constrained generation parses under G.
The open regex terminals of G (waypoints `/[A-Z]{3,10}/`, alphanumeric callsigns) let a
constrained model emit invented identifiers — GCD guarantees syntax, not lexical compliance.

## Models

Any Hugging Face causal LM works. Shipped: `llama31_8b`, `llama31_8b_instruct`, `qwen3_8b`,
`gpt2_large`, `gemma2_9b_it`, `gemma3_1b_it`, `mistral_7b_instruct`, `tiny_debug`.
To add one, drop a YAML file in `configs/models/`:

```yaml
# configs/models/phi4_mini.yaml
model:
  hf_id: microsoft/Phi-4-mini-instruct
  prompt_format: alpaca          # or `chat` to use the tokenizer's chat template
  max_length: 1024
method_overrides:                # optional per-method hyper-parameters for this model
  sft: {train: {lr: 1.0e-5}}
```

Decoder layers (for GradNorm), attention backend (flash-attn → SDPA fallback), multi-GPU
placement (`device_map=auto` pipeline over all visible GPUs) and padding are handled generically.

## Configuration

Layers are merged in this order (later wins), and unknown keys are errors:

```
configs/base.yaml → domains/<d>.yaml → methods/<m>.yaml → models/<k>.yaml → models/<k>.yaml:method_overrides[<m>] → --set
```

Any field can be overridden from the command line, e.g.
`--set train.lr=1e-5 --set objective.lambda_phr=0.3 --set eval.test_max_pairs=500`.
`scope experiment ... --print-config` shows the resolved config; every run also saves it as
`config.yaml` next to its results.

## Outputs

```
results/<domain>/<model>/<method>/s<seed>/
  config.yaml              resolved configuration
  run.log                  full console log
  steps.csv, epochs.csv    training curves (same values as TensorBoard / W&B)
  tensorboard/             tensorboard --logdir results/
  wandb/                   W&B files (run name <domain>-<model>-<method>-s<seed>, grouped by method)
  val_predictions_e*.jsonl per-epoch validation generations
  checkpoint-best/         weights selected on validation C̄ subject to Hall ≤ 10 %
  dpo_pairs.jsonl          (DPO) preference pairs with reference log-probs
  test_predictions.jsonl   one row per test example with all per-example metrics
  test_metrics.json        aggregates — the only input of `scope table`
```

Each (model, method, seed) has exactly one current run folder. Re-launching skips finished
runs and resumes crashed ones from their last step checkpoint. `--force` re-runs a finished
run and keeps the previous attempt as `s<seed>.<YYYYmmdd-HHMMSS>/` with its own W&B run
(add `--discard` to delete it instead). Tables, statistics, plots and latency read only the
current `s<seed>/` folders, never archived attempts.

## Ablation study (`scope ablate`, `scope ablate-report`)

`configs/ablations/scope.yaml` defines 51 variants in 8 groups, each tied to a claim:

| group | variants | question |
|---|---|---|
| components | −L_tok, −L_phr, −L_cfg, CE only (same recipe), CE+each single level, CE+L_phr+L_cfg | is every level necessary; is any single level sufficient? |
| recipe | no loss curriculum, no λ ramp, fixed λ (no GradNorm), no reward length penalty, binary R_cfg, +KL β=0.05, sum log-prob | do the gains come from the objective rather than heuristics? |
| data | unweighted, two-phase (A), gold+silver only (C) | robustness to how noisy targets are used |
| grpo | M=2, M=8, T=1.0, RL every step | sensitivity of the RL estimator |
| lambda | one-at-a-time ×0.2…×5 on each λ | stability / Pareto trade-off |
| compute | SFT for 6 epochs, CE-only under the SCOPE recipe | not explained by more training? |
| inference | SCOPE+GCD†, SCOPE+GCD, GCD+SFT with 3/5 beams | complementarity with decoding-time constraints |
| prompt | {minimal, task, rules, 3-shot} prompt × {Vanilla, GCD, SFT, SCOPE} | do the gains hinge on prompt engineering? |

```bash
scope ablate --models llama31_8b --seeds 42 123 456 --launcher slurm                # full suite
scope ablate --models qwen3_8b gpt2_large --groups components compute prompt --launcher slurm  # core claims, all models
scope ablate --models llama31_8b --groups prompt --launcher slurm                   # one group
scope ablate-report                     # tables/ablation_atc.tex: one block per group, Δ vs the group's
                                        # reference (full SCOPE; in `prompt`, the same method's `rules`
                                        # run) with arrows + Holm-corrected paired-permutation stars, GPU-h;
                                        # sensitivity curves, the prompt × method figure, a C̄-vs-Slot-F1 Pareto plot
```

**Prompt explicitness** (`data.prompt_style`, used for training *and* testing):
`minimal` — the incoming transmission only; `task` — task description, exchange context and roles;
`rules` — plus the explicit ICAO phraseology rules (the default and the paper prompt; byte-identical
to earlier versions); `fewshot` — plus `data.fewshot_k` fixed gold demonstrations of the same response
role, taken from the training split (parse under G, unique target, related to their request, distinct
exchange types). In this group each level is contrasted with the *same method's* `rules` run, and
`ablate-report` draws one line per method across the four levels (`ablation_<d>_<model>_prompt.pdf`).
Any run can use a level directly: `--set data.prompt_style=minimal`.

Cost: the full suite is ≈ 140 runs for one 8B model with 3 seeds (the `prompt` group alone is 18
training runs + 6 inference-only runs). Suggested order: `components`, `compute` and `prompt` on every
model; `recipe`, `inference`, `lambda`, `grpo` and `data` on the primary model.

The `full` variant reuses the main-grid `scope` runs (and `vanilla`/`gcd_cfg`/`sft` for the `rules`
level of the prompt group), so nothing is trained twice. Variants are stored as
`results/<d>/<model>/scope__<variant>/s<seed>/`; they are reported only by `scope ablate-report`
(the main `scope table`/`stats` keep to the main-grid methods, so ablations never enlarge the
main Holm family). Add a variant by appending a few lines (`set:` overrides, optional `method:`,
`init_from:`, `sweep:`, `deterministic:`).

## Compute and latency (`scope compute`, `scope latency`)

Every run writes `compute.json`: hardware, parameters, wall-clock and GPU-hours, peak GPU memory,
optimizer steps, samples/s, **time per training phase** (CE+L_tok forward/backward, rollout
generation, reward scoring, policy-gradient forward/backward, KL reference pass, GradNorm,
optimizer, validation; DPO: pair sampling, scoring, reference log-probs), tokens per phase and
a FLOP estimate (6N per trained token, 2N per generated/scored token), plus amortised test-time
cost (ms/utterance, ms/token, share of time computing grammar masks).

```bash
scope latency --n 200          # batch-1 benchmark of finished runs on the same 200 test utterances:
                               # mean/p50/p90/p95/p99 latency, ms/token, time-to-first-token, mask share
scope compute                  # tables/compute_atc.tex + CSVs: GPU-h (own and cumulative incl. SFT),
                               # peak memory, PFLOPs, latency; phase-breakdown and latency box plots;
                               # latency_atc_paired.csv: per-utterance latency ratio vs SCOPE with
                               # 95% bootstrap CI and Wilcoxon p
```

Run `scope latency` on an otherwise idle GPU of the same type for all methods.

## Monitoring (TensorBoard + W&B)

Both are on by default and receive the same scalars: `train/*` (loss components, rewards,
lr, grad norm) and `lambda/*`, `contrib/*` (λ·L) per optimizer step; `val/*` per epoch;
sample generations as a W&B table; final `test/*` in the run summary. In W&B, group by
`group` (`atc/<model>/<method>`) to see seeds aggregated. On nodes without internet use
`--set logging.wandb.mode=offline` and later `wandb sync results/**/wandb/offline-run-*`.
Re-running a pre-empted job resumes the same W&B run and the latest step checkpoint.

## Statistical analysis (`scope stats`)

For the reference method (default `scope`) against every baseline, per model and metric:

* **Seed level** — mean, std and 95 % t-interval per method over seeds; Welch's t-test; paired
  t-test on seed-matched runs (one-sample t-test against a deterministic baseline). With three
  seeds these have low power and are reported for completeness.
* **Example level** — per test utterance, metrics are averaged over each method's seeds and
  paired across methods: mean difference with a 95 % paired-bootstrap CI (100k resamples),
  two-sided sign-flip permutation test (100k), Wilcoxon signed-rank, paired t-test and
  Cohen's d_z.
* **Multiplicity** — Holm–Bonferroni within each model over all baseline × metric tests.

`significance_<domain>.tex` shows Δ with stars from the Holm-adjusted permutation p-value and
the bootstrap CI; the CSV has every statistic. All evaluated subsets use the same examples
(file order), which the paired tests require.

## Repository layout

```
configs/            base, domains/, methods/, models/, slurm/
artefacts/atc/      vocab.json (V), ngrams.json (P), grammar.lark (G)
src/scope/
  config.py         typed, layered configuration
  data/             prompts (4 explicitness levels, few-shot picker), splits, data curricula, datasets
  domain/           artefacts + parallel, timeout-safe role-gated CFG scorer (C_cfg)
  metrics/          C_tok, C_phr, Slot-F1, DA-F1, hallucination, BERTScore (+ BERT MLM)
  losses/           L_CE, L_tok, GRPO helpers, DPO, loss curriculum, GradNorm
  decoding/         gcd.py: GCD† (vocab/bigram) · cfg_gcd.py: GCD of Geng et al. (full CFG)
  training/         base loop (accumulation, selection, resume) · SCOPE/SFT · DPO
  evaluation/       batched generation + per-example scoring
  pipeline.py       one experiment end to end
  runner.py         grid planning, local / SLURM launchers
  tracking.py       TensorBoard + W&B + CSV
  analysis/         results table, statistics, ablation report, compute/latency, plots
  utils/profiling   CUDA-synchronised phase timers, token counters, FLOP estimates
  cli.py            `scope` command
scripts/            tiny debug model, flash-attn install, SLURM example
tests/              unit + gradient-path tests (CPU)
```

See `NOTES.md` for differences from the original scripts and paper/code consistency items.
