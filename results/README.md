# Results artefacts

All numbers in `README.md` and `EXPERIMENT_LOG.md` are generated from the frozen snapshot
described here. Regenerate everything with `make data && make notebooks && make run`.

## `data/`

JSONL call logs, one record per call.

| File | Contents |
| --- | --- |
| `source.jsonl` | 240 train + 80 dev + 100 holdout source calls |
| `disfluency.jsonl` | 100 paired transformations of the source holdout |
| `emotional_arc.jsonl` | 100 paired transformations |
| `paraphrase.jsonl` | 100 paired transformations |
| `combined.jsonl` | 100 paired transformations (paraphrase + arc + disfluency) |
| `persona_mix.jsonl` | 100 unpaired calls from a shifted persona distribution |

Seed: `20261020`. Paired files share `call_id` with the source holdout and preserve the latent
label and survey response. The generator is in `src/callsat_audit/generate.py`.

## `runs/`

| File | Produced by | Contents |
| --- | --- | --- |
| `01_source_metrics.csv` | `01_baseline.ipynb` | Source holdout metrics vs latent and survey |
| `02_latent_metrics.csv` | `02_shift_stress.ipynb` | Per-condition metrics + bootstrap QWK/ECE CIs |
| `02_survey_metrics.csv` | `02_shift_stress.ipynb` | Same metrics against the simulated survey |
| `02_paired_consistency.csv` | `02_shift_stress.ipynb` | Span/fact/score/label changes + paired ΔQWK CIs |
| `02_break_localisation.csv` | `02_shift_stress.ipynb` | Fact accuracy, gold-turn hit, quote support |
| `02_conformal_coverage.csv` | `02_shift_stress.ipynb` | 90% conformal coverage and set size |
| `02_audit_flags.csv` | `02_shift_stress.ipynb` | AUC of runtime flags for large errors |
| `03_ledger.csv` | `03_iterations.ipynb` | C0–C4 keep/discard ledger |
| `llm_pilot_direct_metrics.csv` | `scripts/run_llm_pilot.py` | Direct small-LLM metrics per condition |
| `llm_pilot_direct_<condition>.csv` | `scripts/run_llm_pilot.py` | Per-call direct LLM scores and citations |
| `llm_pilot.log` | `scripts/run_llm_pilot.py` | Raw pilot console log |

## `figures/`

| File | Contents |
| --- | --- |
| `01_source_reliability.png` | Reliability diagram vs latent and vs survey |
| `01_source_confusion.png` | Source holdout confusion matrix |
| `02_shift_metrics.png` | QWK and ECE by condition |

## `cache/`

| Path | Contents |
| --- | --- |
| `cache/extractions/offline-strict/` | Per-call lexicon extractions (JSONL) |
| `cache/extractions/offline-broad/` | Broad-lexicon variant used in C1 |
| `cache/direct_scores/` | Per-call direct LLM responses (JSONL) |

Caches are resumable and keyed by extractor/model name + condition. They are ignored by git;
delete them to force a clean rerun.

## Provenance caveats

- All data are synthetic; “latent satisfaction” is the generator's structural equation.
- The offline extractor is written against the templates and is not an LLM.
- The LLM pilot is one local 3B model on 40 source calls plus their paraphrase twins.
- The shared chat that motivated the topic was retrieved through a reader proxy because the
  original CloudFront URL returned 403; a copy is in `docs/context/shared_chat.md`.
