# callsat-audit

Preliminary research harness for **Topic 1: Auditability and Calibration of LLM-Based
Satisfaction Scoring on Synthetic Call Logs** (ECIS Track 10 framing, service measurement and
governance).

The object of study is a call-eval-style measurement chain:

```
transcript -> cited evidence spans -> typed facts -> deterministic score -> survey calibration
```

We generate paired synthetic call logs, shift the data-generating process along Mirage-style
dimensions, and ask whether the chain stays calibrated and auditable. This is a pilot: all data
are synthetic, and the LLM arm is a small local open model, not a frontier system.

## Current status (preliminary)

- **Baseline (source holdout):** QWK 0.718 vs latent satisfaction, ECE 0.050, calibration slope
  0.985 — but QWK only 0.502 and ECE 0.128 against the noisy survey the calibrator was fitted to.
- **Lexical shift (paraphrase):** QWK falls to 0.098, evidence-turn hit rate to 0.082, conformal
  coverage to 0.760. Isotonic calibration recovers QWK only to 0.107.
- **Disfluency:** QWK 0.661, ECE 0.071, large-error rate 0.280. The paired ΔQWK CI includes zero
  (+0.057 [−0.022, 0.149]), so this is directional, not established.
- **Emotional arc (concept shift):** QWK 0.697, ECE 0.065, but sentiment accuracy 0.620 and 38%
  of calls show a sentiment/resolution conflict.
- **Combined shift:** QWK 0.029, ECE 0.188, calibration slope 0.019, conformal coverage 0.750.
- **LLM pilot (Qwen2.5-3B, local MLX, 40 source calls + paraphrase twins):** source QWK
  0.669, paraphrase QWK **0.667** — it holds up under lexical shift while the symbolic chain
  falls to 0.113. But it is severely overconfident (mean confidence ≈ 0.87, confidence ECE
  ≈ 0.50) and citation precision is only 0.53–0.60.

The full decision log, including discarded changes, is in [`EXPERIMENT_LOG.md`](EXPERIMENT_LOG.md).
A LaTeX/TikZ work-in-progress report is in [`report/`](report/): `main.tex`, `refs.bib`, and the
compiled `callsat-audit-preliminary-report.pdf`.

## Notebooks

| Notebook | Purpose |
| --- | --- |
| `notebooks/00_protocol.ipynb` | RQ, hypotheses, pre-specified decision rules, design, limitations |
| `notebooks/01_baseline.ipynb` | Source replication: facts, calibration, reliability, evidence audit |
| `notebooks/02_shift_stress.ipynb` | Paired shifts, calibration transfer, break localisation, conformal coverage, audit signals |
| `notebooks/03_iterations.ipynb` | Keep/discard cards C1–C4 and final ledger |

## Reproduce

```bash
make setup          # uv sync + editable install
make data           # generate the frozen synthetic datasets
make notebooks      # build notebook sources
make run            # execute all notebooks headlessly
make test lint typecheck
```

Or manually:

```bash
export UV_CACHE_DIR="$PWD/.uv-cache"
uv sync && uv pip install -e .
uv run python scripts/build_notebooks.py
uv run python scripts/run_notebooks.py
uv run pytest
uv run ruff check src tests scripts
uvx ty check src
```

The LLM pilot is optional and slow:

```bash
export HF_HOME="$PWD/.hf-cache"
uv run --extra llm python scripts/run_llm_pilot.py --n 40
```

It writes per-call scores to `results/cache/direct_scores/` and metrics to `results/runs/`.

## Layout

| Path | Purpose |
| --- | --- |
| `src/callsat_audit/generate.py` | Synthetic DGP, evidence cues, paired shift transformations |
| `src/callsat_audit/extract.py` | Strict/broad offline extractor, LLM extraction prompt, local MLX backend |
| `src/callsat_audit/scoring.py` | Transparent fact-to-score weights |
| `src/callsat_audit/calibration.py` | Ordinal + temperature, isotonic, split conformal |
| `src/callsat_audit/audit.py` | Span validity, fact recovery, paired consistency |
| `src/callsat_audit/pipeline.py` | Extraction cache, calibration bundle, frames, metrics |
| `src/callsat_audit/llm_scorer.py` | Direct LLM scoring prompt and parser |
| `results/` | Figures, metric CSVs, cached extractions, generated data |
| `tests/` | Invariants for metrics, calibration, shifts, and cue spans |

## Context note

The shared conversation linked in the task brief was CloudFront-blocked, and is deliberately not
included here since it contains personal correspondence. The referenced
`call-eval` repository is now 404; its project guide is archived at
`docs/context/call_eval_project_guide.md`, and the study replicates its *architecture* rather
than its exact data.

## Limitations

Synthetic labels are the generator, not humans. The offline extractor is a lower-bound
instrument written against the templates; the broad-lexicon repair is circular by construction.
The LLM pilot is one 3B model on 40 source calls plus their paraphrase twins and cannot support
claims about frontier LLMs. See `EXPERIMENT_LOG.md` §5 for the full threat list.
