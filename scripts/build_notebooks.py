"""Generate the analysis notebooks from readable Python strings.

The notebooks are the artefact; this builder only keeps their source reviewable and
avoids hand-editing JSON.  Run `python scripts/build_notebooks.py`, then execute with
`python scripts/run_notebooks.py`.
"""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
NB_DIR = ROOT / "notebooks"
NB_DIR.mkdir(exist_ok=True)


def md(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(text.strip())


def code(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(text.strip())


def build_protocol() -> nbf.NotebookNode:
    cells = [
        md(
            """
# Topic 1 — Auditability and calibration of LLM-based satisfaction scoring

**Preliminary research notebook (protocol).** This is a pilot, not the paper. All data are
synthetic; no claim here is about real callers, real agents, or frontier LLMs.

The object of study is a call-eval-style measurement chain:

```
transcript -> cited evidence spans -> typed facts -> deterministic score -> survey calibration
```

The question is whether that chain stays **calibrated** and **auditable** when the synthetic
data-generating process shifts along Mirage-style dimensions (disfluency, emotional arcs,
persona mix, lexical paraphrase).

**Status.** The offline extractor is a strict lexicon baseline; a small open model (Qwen2.5-3B,
4-bit, local MLX) is used only as a feasibility pilot for direct scoring. The results are a
stress test of the *architecture and audit metrics*, not a benchmark of LLM quality.
"""
        ),
        md(
            """
## Research question

> When satisfaction scores are produced through an evidence-chain plus survey-calibration
> pipeline on synthetic call logs, do calibration and evidence consistency survive
> distributional shift — and can runtime audit signals flag the calls that break?

### Hypotheses and pre-specified readings

| ID | Hypothesis | Operationalisation | Kept if |
| --- | --- | --- | --- |
| **H1** | Calibration degrades under shift. | Holdout ECE and calibration slope after applying source-fitted calibration to each shift, vs source holdout. | ΔECE > 0.03 with bootstrap CI excluding 0, or slope moves away from 1 by > 0.15. |
| **H2** | Failures localise before calibration, at the span/fact layer. | Gold evidence-turn hit rate, fact accuracy, and paired score-change rate on label-preserving shifts. | Evidence hit or fact accuracy drops > 5 pp while latent labels are unchanged. |
| **H3** | Runtime evidence flags predict large scoring errors better than model confidence. | AUC of missing-evidence / conflict flags vs AUC of posterior confidence for `abs(error) > 1`. | Audit AUC exceeds confidence AUC by > 0.05. |
| **H4** | Conformal coverage is not robust to synthetic shift. | 90% split-conformal sets fitted on source, evaluated on each target. | Coverage falls below 0.85 under at least one shift. |

H3 and H4 are deliberately falsifiable; a negative result is a finding, not a failure.

**Power note.** With 100 target calls, ECE bootstrap intervals are wide enough to overlap
across source and the mild shifts. H1 is therefore only claimed where the paired ΔQWK interval
also excludes zero; otherwise the result is reported as directional.

## Design

| Component | Source | Shifted targets |
| --- | --- | --- |
| Calls | 240 train + 80 dev + 100 sealed holdout | 100 paired calls per shift |
| Labels | latent 1–5, noisy survey (~72% response), two simulated annotators | latent/survey held fixed for paired shifts |
| Calibration | fit on source train+dev surveys only | applied unchanged to targets |
| Shifts | — | disfluency (label-preserving), emotional arc (concept), paraphrase (lexical), persona mix (covariate), combined |
| Extractor | strict offline lexicon | same |
| Scores | 1–5 deterministic fact score; ordinal + isotonic calibration | same |

The **persona-mix** target is unpaired (new calls, different persona weights); all other
shifts are paired with the source holdout, which enables span/fact/score consistency checks.

## Keep/discard loop

Every code change is an experiment card with a pre-specified decision rule:

1. State the change and the metric it should move.
2. Run it on source **and** shifted targets.
3. Read the result table.
4. Record **KEEP** or **DISCARD** in `03_iterations.ipynb` and `EXPERIMENT_LOG.md`.

Thresholds: keep a change if it improves the target metric by at least the pre-specified
margin without a source regression larger than 0.02 QWK. Changes that only help because they
encode knowledge of the perturbation (e.g. a lexicon written from the paraphrase list) are
marked **KEEP (diagnostic, circular)** and are not treated as deployable fixes.

## What this pilot cannot show

- It cannot rank LLMs: the LLM arm is one 3B open model on 40 calls per condition.
- It cannot validate the synthetic labels: the “ground truth” is the generator.
- It cannot establish real-world calibration: surveys are simulated and noisy.
- It cannot claim the offline lexicon represents an LLM extractor; it is a transparent
  lower bound / stress-test instrument.
"""
        ),
        code(
            """
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from callsat_audit.config import FIGURES_DIR, ROOT
from callsat_audit.generate import load_datasets, summary_table

pd.set_option("display.max_columns", 60)
pd.set_option("display.width", 140)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

datasets = load_datasets()
print(summary_table(datasets))
"""
        ),
        code(
            """
rows = []
for condition, calls in datasets.items():
    labels = np.array([c.latent_satisfaction for c in calls])
    surveys = np.array([c.survey_response for c in calls if c.survey_response is not None])
    rows.append(
        {
            "condition": condition,
            "n": len(calls),
            "survey_rate": len(surveys) / len(calls),
            "latent_mean": labels.mean(),
            "latent_sd": labels.std(),
            "pct_resolved": np.mean([c.gold_resolution == "resolved" for c in calls]),
            "mean_effort": np.mean([c.gold_effort for c in calls]),
            "mean_sentiment": np.mean([c.gold_sentiment for c in calls]),
        }
    )
pd.DataFrame(rows).set_index("condition").round(3)
"""
        ),
        code(
            """
# Invariant check: paired shifts must not change the latent label or survey.
source_holdout = {c.call_id: c for c in datasets["source"] if c.split == "holdout"}
checks = []
for condition in ("disfluency", "emotional_arc", "paraphrase", "combined"):
    paired = [c for c in datasets[condition] if c.call_id in source_holdout]
    label_changes = sum(
        c.latent_satisfaction != source_holdout[c.call_id].latent_satisfaction for c in paired
    )
    survey_changes = sum(
        c.survey_response != source_holdout[c.call_id].survey_response for c in paired
    )
    checks.append(
        {
            "condition": condition,
            "paired": len(paired),
            "latent_changes": label_changes,
            "survey_changes": survey_changes,
        }
    )
pd.DataFrame(checks).set_index("condition")
"""
        ),
        md(
            """
## A source call and its gold evidence

The generator records every cue span, so the audit can distinguish three failure loci:
span missed → fact wrong → score/calibration shift. The example below shows the rendered
transcript plus the gold facts that a perfect extractor would recover.
"""
        ),
        code(
            """
example = next(
    c for c in datasets["source"] if c.split == "holdout" and c.meta.get("implicit_resolution")
)
print(example.transcript())
print()
print(
    {
        "latent": example.latent_satisfaction,
        "survey": example.survey_response,
        "gold": {
            k: example.gold_fact(k)
            for k in ("resolved", "effort", "sentiment", "empathy")
        },
        "meta": example.meta,
    }
)
"""
        ),
    ]
    notebook = nbf.v4.new_notebook(cells=cells)
    notebook.metadata["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    return notebook


def build_baseline() -> nbf.NotebookNode:
    cells = [
        md(
            """
# 01 — Baseline: does the evidence chain calibrate on source?

This notebook establishes the reference point before any stress test. We fit the strict
offline extractor and the survey calibration layer on the source training/dev calls, then
read out the sealed source holdout.

Two targets are reported side by side:

* **latent** — the generator's true satisfaction (available only because the data are synthetic);
* **survey** — the noisy simulated survey that the calibration layer was actually fitted to.

The gap between them is an audit finding in its own right: a pipeline can look calibrated
against surveys while drifting from the construct it claims to measure.
"""
        ),
        code(
            """
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from callsat_audit.config import FIGURES_DIR, ROOT
from callsat_audit.extract import LexiconExtractor
from callsat_audit.generate import load_datasets
from callsat_audit.metrics import confusion, reliability_bins
from callsat_audit.pipeline import (
    build_frame,
    confusion_frame,
    extract_all,
    fit_calibration,
    probs_matrix,
    summarize_frame,
)

FIGURES_DIR.mkdir(parents=True, exist_ok=True)
datasets = load_datasets()
source = datasets["source"]

extractor = LexiconExtractor()
source_extractions = extract_all(source, extractor, condition="source")
bundle = fit_calibration(source, source_extractions)
print(
    f"calibration: temperature={bundle.temperature:.3f}, "
    f"conformal qhat={bundle.qhat:.3f} (alpha={bundle.alpha}), "
    f"source survey n={sum(c.survey_response is not None for c in source if c.split != 'holdout')}"
)
"""
        ),
        code(
            """
frame = build_frame(source, source_extractions, bundle)
holdout = frame[frame["split"] == "holdout"].copy()

summary = pd.DataFrame(
    {
        "vs latent": summarize_frame(holdout, target="latent"),
        "vs survey": summarize_frame(holdout, target="survey"),
    }
)
summary_view = summary.loc[
    [
        "n",
        "qwk_raw",
        "qwk_ordinal",
        "qwk_isotonic",
        "mae_raw",
        "mae_ordinal",
        "mae_isotonic",
        "ece",
        "brier",
        "nll",
        "calibration_slope",
        "calibration_intercept",
        "mean_confidence",
        "large_error_rate",
    ]
].round(3)
summary_view.to_csv(ROOT / "results" / "runs" / "01_source_metrics.csv")
summary_view
"""
        ),
        code(
            """
# Reliability diagram against the latent construct and against the survey.
latent = holdout.dropna(subset=["latent"])
survey = holdout.dropna(subset=["survey"])
fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
for ax, subset, title in (
    (axes[0], latent, "vs latent satisfaction"),
    (axes[1], survey, "vs simulated survey"),
):
    xs, ys, ns = reliability_bins(subset["latent"] if title.startswith("vs latent") else subset["survey"], probs_matrix(subset))
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", label="perfect")
    ax.plot(xs, ys, marker="o", label="ordinal + temperature")
    ax.set_xlabel("mean confidence")
    ax.set_ylabel("empirical accuracy")
    ax.set_title(title)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="lower right", fontsize=8)
fig.suptitle("Source holdout reliability")
fig.tight_layout()
fig.savefig(FIGURES_DIR / "01_source_reliability.png", dpi=160)
plt.show()
"""
        ),
        code(
            """
cm = confusion_frame(holdout, target="latent")
fig, ax = plt.subplots(figsize=(4.6, 4.2))
image = ax.imshow(cm, cmap="Blues")
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=9)
ax.set_xticks(range(5), [1, 2, 3, 4, 5])
ax.set_yticks(range(5), [1, 2, 3, 4, 5])
ax.set_xlabel("predicted")
ax.set_ylabel("latent")
ax.set_title("Source holdout confusion")
fig.colorbar(image, ax=ax, fraction=0.046)
fig.tight_layout()
fig.savefig(FIGURES_DIR / "01_source_confusion.png", dpi=160)
plt.show()
"""
        ),
        md(
            """
## Evidence audit

`gold_turn_hit` is the fraction of facts whose cited turn overlaps a gold cue turn.
`quote_supported` checks that the extractor's quote actually appears in a transcript turn.
`missing_resolution_evidence` flags the call-eval failure mode where the scorer falls back
to a default because it could not find a resolution statement.

These are **oracle-assisted diagnostics** (they use the generator's gold turns). They tell us
where the chain breaks; they are not deployable runtime flags.
"""
        ),
        code(
            """
audit_table = pd.DataFrame(
    {
        "value": {
            "fact_accuracy_resolved": holdout["correct_resolved"].mean(),
            "fact_accuracy_effort": holdout["correct_effort"].mean(),
            "fact_accuracy_sentiment": holdout["correct_sentiment"].mean(),
            "fact_accuracy_empathy": holdout["correct_empathy"].mean(),
            "mean_fact_errors": holdout["fact_error_count"].mean(),
            "gold_turn_hit": holdout["gold_turn_hit"].mean(),
            "quote_supported": holdout["quote_supported"].mean(),
            "missing_resolution_evidence": holdout["missing_resolution_evidence"].mean(),
            "sentiment_resolution_conflict": holdout["sentiment_resolution_conflict"].mean(),
        }
    }
)
audit_table.round(3)
"""
        ),
        code(
            """
# A single auditable trace: show the transcript, the extracted facts, and the cited spans.
candidate = next(
    (
        c
        for c in source
        if c.split == "holdout" and c.meta.get("implicit_resolution")
    ),
    next(c for c in source if c.split == "holdout"),
)
ex = source_extractions[candidate.call_id]
print(candidate.transcript())
print()
for name, fact in ex.facts.items():
    print(f"{name:10s} = {fact.value!r:12} conf={fact.confidence:.2f} turns={fact.evidence_turns} quote={fact.quotes}")
print("gold:", {k: candidate.gold_fact(k) for k in ("resolved", "effort", "sentiment", "empathy")})
"""
        ),
        md(
            """
## Reading the baseline

The source holdout is the optimistic case: the extractor was written against these templates,
the calibration layer sees the same survey noise process, and no shift has occurred. Any
degradation we observe later is therefore attributable to the shift, not to an intrinsically
hard task.

If QWK against latent is meaningfully higher than QWK against survey, that is the **survey
ceiling** — a limit on what survey-anchored calibration can prove. If isotonic beats ordinal on
MAE but ordinal is needed for ECE and selective prediction, both layers should be kept for
different jobs.
"""
        ),
    ]
    return nbf.v4.new_notebook(cells=cells)


def build_stress() -> nbf.NotebookNode:
    cells = [
        md(
            """
# 02 — Stress test: calibration and evidence consistency under shift

The same source-fitted extractor and calibration layer are applied unchanged to five targets.
For the four paired shifts, every target call is a transformed twin of a source holdout call
with the **same latent label and survey response**. This lets us separate label-preserving
measurement noise (disfluency) from concept shift (emotional arc) and lexical shift
(paraphrase).
"""
        ),
        code(
            """
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from callsat_audit.config import FIGURES_DIR, ROOT
from callsat_audit.extract import LexiconExtractor
from callsat_audit.generate import load_datasets
from callsat_audit.metrics import bootstrap_ece_ci, bootstrap_qwk_ci, paired_qwk_delta_ci
from callsat_audit.pipeline import (
    build_frame,
    condition_metrics,
    extract_all,
    fit_calibration,
    paired_consistency_frame,
    probs_matrix,
    summarize_consistency,
)

FIGURES_DIR.mkdir(parents=True, exist_ok=True)
datasets = load_datasets()
source = datasets["source"]
conditions = [
    "source",
    "disfluency",
    "emotional_arc",
    "paraphrase",
    "combined",
    "persona_mix",
]

extractor = LexiconExtractor()
extractions = {
    condition: extract_all(datasets[condition], extractor, condition=condition)
    for condition in conditions
}
bundle = fit_calibration(source, extractions["source"])
frames = {
    condition: build_frame(datasets[condition], extractions[condition], bundle)
    for condition in conditions
}
# The source frame also contains train/dev, which were used to fit the calibrator.  Keep only
# the sealed holdout so every row in the comparison table is out-of-sample.
frames["source"] = frames["source"][frames["source"]["split"] == "holdout"].copy()
len(frames["source"]), bundle.temperature, bundle.qhat
"""
        ),
        code(
            """
latent_metrics = condition_metrics(frames, target="latent")
survey_metrics = condition_metrics(frames, target="survey")
show = [
    "qwk_raw",
    "qwk_ordinal",
    "qwk_isotonic",
    "mae_raw",
    "ece",
    "brier",
    "calibration_slope",
    "mean_confidence",
    "large_error_rate",
    "risk_coverage_auc",
]
latent_metrics[show].round(3)
"""
        ),
        code(
            """
ci_rows = []
for condition, frame in frames.items():
    sub = frame.dropna(subset=["latent"])
    qwk_low, qwk_high = bootstrap_qwk_ci(sub["latent"], sub["raw_label"])
    ece_low, ece_high = bootstrap_ece_ci(sub["latent"], probs_matrix(sub))
    ci_rows.append(
        {
            "condition": condition,
            "qwk_ci_low": qwk_low,
            "qwk_ci_high": qwk_high,
            "ece_ci_low": ece_low,
            "ece_ci_high": ece_high,
        }
    )
latent_metrics_ci = latent_metrics.join(pd.DataFrame(ci_rows).set_index("condition"))
latent_metrics_ci[
    ["qwk_raw", "qwk_ci_low", "qwk_ci_high", "ece", "ece_ci_low", "ece_ci_high"]
].round(3)
"""
        ),
        code(
            """
survey_metrics[["n", *show]].round(3)
"""
        ),
        code(
            """
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
latent_metrics["qwk_raw"].plot(kind="bar", ax=axes[0], color="#4C72B0")
axes[0].set_title("QWK vs latent (raw fact score)")
axes[0].set_ylim(0, 1)
axes[0].axhline(latent_metrics.loc["source", "qwk_raw"], color="grey", linestyle="--", linewidth=1)
latent_metrics["ece"].plot(kind="bar", ax=axes[1], color="#C44E52")
axes[1].set_title("ECE vs latent (ordinal + temperature)")
axes[1].set_ylim(0, max(0.2, latent_metrics["ece"].max() * 1.1))
for ax in axes:
    ax.tick_params(axis="x", rotation=30)
fig.tight_layout()
fig.savefig(FIGURES_DIR / "02_shift_metrics.png", dpi=160)
plt.show()
"""
        ),
        md(
            """
## Paired evidence consistency

For each paired shift we compare the source call and its twin on four properties:

1. **span** — do the cited evidence turns survive? (`evidence_turn_jaccard`)
2. **fact** — do the typed facts change? (`mean_fact_changes`)
3. **score** — does the raw/expected score move beyond tolerance? (`score_shift_rate`)
4. **label** — does the predicted ordinal label change? (`label_change_rate`)

The diagonal we care about is *fact or span changes with no latent change*. That is the
evidence chain breaking while the construct stays fixed.
"""
        ),
        code(
            """
paired_rows = []
for condition in ("disfluency", "emotional_arc", "paraphrase", "combined"):
    paired = paired_consistency_frame(
        [c for c in source if c.split == "holdout"],
        datasets[condition],
        extractions["source"],
        extractions[condition],
        bundle,
    )
    paired_rows.append({"condition": condition, **summarize_consistency(paired)})
paired_table = pd.DataFrame(paired_rows).set_index("condition")

source_hold = frames["source"][frames["source"]["split"] == "holdout"].set_index("call_id")
delta_rows = []
for condition in paired_table.index:
    shifted = frames[condition].set_index("call_id")
    common = source_hold.index.intersection(shifted.index)
    delta, low, high = paired_qwk_delta_ci(
        source_hold.loc[common, "latent"],
        source_hold.loc[common, "raw_label"],
        shifted.loc[common, "raw_label"],
    )
    delta_rows.append(
        {"condition": condition, "delta_qwk": delta, "delta_qwk_low": low, "delta_qwk_high": high}
    )
paired_table = paired_table.join(pd.DataFrame(delta_rows).set_index("condition"))
paired_table.round(3)
"""
        ),
        code(
            """
# Break localisation: where does each condition fail?
rows = []
for condition, frame in frames.items():
    rows.append(
        {
            "condition": condition,
            "fact_errors": frame["fact_error_count"].mean(),
            "resolved_acc": frame["correct_resolved"].mean(),
            "sentiment_acc": frame["correct_sentiment"].mean(),
            "gold_turn_hit": frame["gold_turn_hit"].mean(),
            "quote_supported": frame["quote_supported"].mean(),
            "missing_resolution": frame["missing_resolution_evidence"].mean(),
            "conflict": frame["sentiment_resolution_conflict"].mean(),
        }
    )
breakdown = pd.DataFrame(rows).set_index("condition")
breakdown.round(3)
"""
        ),
        md(
            """
## Conformal coverage

The 90% split-conformal set is fitted on source out-of-fold residuals and applied unchanged
to each target. The finite-sample guarantee assumes exchangeability; shift breaks it. Coverage
below 0.85 is the pre-specified failure signal.
"""
        ),
        code(
            """
rows = []
for condition, frame in frames.items():
    sets = bundle.conformal_sets(frame["raw_score"].to_numpy())
    coverage = np.mean([int(y) in s for y, s in zip(frame["latent"], sets)])
    rows.append(
        {
            "condition": condition,
            "coverage": coverage,
            "mean_set_size": np.mean([len(s) for s in sets]),
            "nominal": 1 - bundle.alpha,
        }
    )
coverage_table = pd.DataFrame(rows).set_index("condition")
coverage_table.round(3)
"""
        ),
        md(
            """
## Do runtime audit signals flag large errors?

The deployable signals are computed without gold labels: missing resolution evidence,
sentiment/resolution conflict, low fact confidence, and the absence of a supported quote.
We compare their AUC for `|error| > 1` against the ordinal posterior confidence. A signal is
useful only if it beats confidence by a meaningful margin.
"""
        ),
        code(
            """
from sklearn.metrics import roc_auc_score

flag_rows = []
for condition, frame in frames.items():
    sub = frame.copy()
    sub["large_error"] = (np.abs(sub["pred_label"] - sub["latent"]) > 1).astype(int)
    row = {"condition": condition, "large_error_rate": sub["large_error"].mean()}
    for flag in (
        "missing_resolution_evidence",
        "sentiment_resolution_conflict",
        "low_mean_confidence",
    ):
        row[f"auc_{flag}"] = (
            roc_auc_score(sub["large_error"], sub[flag]) if sub[flag].nunique() > 1 else np.nan
        )
    row["auc_confidence"] = roc_auc_score(sub["large_error"], sub["confidence"])
    flag_rows.append(row)
flag_table = pd.DataFrame(flag_rows).set_index("condition")
flag_table.round(3)
"""
        ),
        code(
            """
out = ROOT / "results" / "runs"
out.mkdir(parents=True, exist_ok=True)
latent_metrics_ci.to_csv(out / "02_latent_metrics.csv")
survey_metrics.to_csv(out / "02_survey_metrics.csv")
paired_table.to_csv(out / "02_paired_consistency.csv")
breakdown.to_csv(out / "02_break_localisation.csv")
coverage_table.to_csv(out / "02_conformal_coverage.csv")
flag_table.to_csv(out / "02_audit_flags.csv")
print("wrote 02_*.csv to", out)
"""
        ),
        md(
            """
### Reading the audit-signal table

Posterior confidence is at or below chance for *ranking* large errors on source, disfluency,
and emotional arc (AUC 0.372, 0.275, 0.354). That is a negative result for confidence-based
deferral: the ordinal model is more confident on exactly the calls it gets wrong, plausibly
because extreme predicted labels look precise. Missing-resolution evidence is a better
diagnostic on disfluency (0.624) and emotional arc (0.648), but not on paraphrase (0.534) or
combined (0.520), where almost every call is missing evidence. AUC is not a deferral policy;
the selective-risk test in `03_iterations.ipynb` is the decision-relevant check.

## Reading the stress test

### Statistical caution

With 100 calls per target the paired ΔQWK confidence intervals are wide:
disfluency **+0.057 [−0.022, 0.149]**, emotional arc **+0.021 [−0.015, 0.062]**,
paraphrase **+0.619 [0.474, 0.738]**, combined **+0.689 [0.570, 0.789]**. Only the two lexical
shifts exclude zero at 95%. The disfluency and emotional-arc effects are therefore
**suggestive, not established** at this sample size. The conformal coverage failure (0.760 and
0.750 vs nominal 0.90) is the clearest calibration-side signal under lexical shift.

The expected pattern is a **gradient of fragility**, not a single number:

* disfluency should hurt span/fact recovery while leaving the latent label untouched;
* emotional arc should change the expressed sentiment and therefore the score, but not the
  latent label — a concept shift;
* paraphrase is the hard lexical case and should expose the largest span failure;
* persona mix is a covariate shift and can change base rates without breaking the chain.

A high QWK on one condition does not cancel a low QWK on another. The audit claim is about the
weakest link in the chain, and the paper should report the whole profile.
"""
        ),
    ]
    return nbf.v4.new_notebook(cells=cells)


def build_iterations() -> nbf.NotebookNode:
    cells = [
        md(
            """
# 03 — Keep/discard log

Each section is one experiment card: **change → run → read → decide**. The decision rules are
fixed before looking at the result (they are repeated in the code). The point is not to make
every change look good; a clean **DISCARD** is a useful preliminary result.

| Card | Change | Target metric | Decision rule |
| --- | --- | --- | --- |
| C1 | broaden the offline lexicon with paraphrase synonyms | paraphrase QWK / evidence hit | KEEP if ΔQWK > 0.05 and source ΔQWK > −0.02 |
| C2 | isotonic vs ordinal calibration | MAE / ECE / QWK | KEEP BOTH if isotonic lowers MAE and ordinal is needed for probabilities |
| C3 | runtime evidence gate for abstention | selective large-error rate at 80% coverage | KEEP if gate beats confidence by > 0.02 absolute or AUC by > 0.05 |
| C4 | direct small-LLM scorer | QWK / ECE / evidence citation | KEEP AS PILOT only if it is competitive enough to justify a frontier-model run |
"""
        ),
        code(
            """
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from callsat_audit.config import ROOT
from callsat_audit.extract import LexiconExtractor
from callsat_audit.generate import load_datasets
from callsat_audit.llm_pipeline import direct_metrics
from callsat_audit.pipeline import (
    build_frame,
    condition_metrics,
    extract_all,
    fit_calibration,
    summarize_frame,
)

datasets = load_datasets()
source = datasets["source"]
conditions = [
    "source",
    "disfluency",
    "emotional_arc",
    "paraphrase",
    "combined",
    "persona_mix",
]
"""
        ),
        md(
            """
## C1 — Broaden the lexicon

**Hypothesis.** The paraphrase collapse in 02 is a span-layer failure. Adding paraphrase
synonyms to the lexicon should recover it.

**Circularity warning.** The added patterns were written by looking at the paraphrase list.
This is an oracle repair, not a deployable fix. It is useful as a diagnostic upper bound: if
even the oracle repair fails, the problem is deeper than vocabulary.
"""
        ),
        code(
            """
strict = LexiconExtractor()
broad = LexiconExtractor(broad=True)
strict_ex = {c: extract_all(datasets[c], strict, condition=c) for c in conditions}
broad_ex = {c: extract_all(datasets[c], broad, condition=c) for c in conditions}
strict_bundle = fit_calibration(source, strict_ex["source"])
broad_bundle = fit_calibration(source, broad_ex["source"])
strict_frames = {c: build_frame(datasets[c], strict_ex[c], strict_bundle) for c in conditions}
broad_frames = {c: build_frame(datasets[c], broad_ex[c], broad_bundle) for c in conditions}
for frame_dict in (strict_frames, broad_frames):
    frame_dict["source"] = frame_dict["source"][frame_dict["source"]["split"] == "holdout"].copy()

comparison = pd.DataFrame(
    {
        "strict_qwk": condition_metrics(strict_frames)["qwk_raw"],
        "broad_qwk": condition_metrics(broad_frames)["qwk_raw"],
        "strict_fact_err": {c: strict_frames[c]["fact_error_count"].mean() for c in conditions},
        "broad_fact_err": {c: broad_frames[c]["fact_error_count"].mean() for c in conditions},
        "strict_gold_hit": {c: strict_frames[c]["gold_turn_hit"].mean() for c in conditions},
        "broad_gold_hit": {c: broad_frames[c]["gold_turn_hit"].mean() for c in conditions},
    }
)
comparison["delta_qwk"] = comparison["broad_qwk"] - comparison["strict_qwk"]
comparison.round(3)
"""
        ),
        code(
            """
target_gain = comparison.loc["paraphrase", "delta_qwk"]
source_loss = comparison.loc["source", "delta_qwk"]
c1_keep = target_gain > 0.05 and source_loss > -0.02
print(f"paraphrase delta QWK = {target_gain:+.3f}; source delta QWK = {source_loss:+.3f}")
print("C1:", "KEEP (diagnostic, circular)" if c1_keep else "DISCARD")
"""
        ),
        md(
            """
## C2 — Isotonic vs ordinal calibration

**Hypothesis.** A monotone isotonic survey map is a better point calibrator than the ordinal
model, but it cannot supply class probabilities for ECE or selective prediction.

**Decision rule.** Keep both if isotonic lowers MAE on source and the ordinal layer remains the
only source of probabilities. This is a “keep both, different jobs” decision, not a winner-take-all.
"""
        ),
        code(
            """
cal_rows = []
for condition, frame in strict_frames.items():
    latent = summarize_frame(frame, "latent")
    survey = summarize_frame(frame, "survey")
    cal_rows.append(
        {
            "condition": condition,
            "mae_raw": latent["mae_raw"],
            "mae_ordinal": latent["mae_ordinal"],
            "mae_isotonic": latent["mae_isotonic"],
            "qwk_ordinal": latent["qwk_ordinal"],
            "qwk_isotonic": latent["qwk_isotonic"],
            "ece_ordinal": latent["ece"],
            "ece_survey": survey["ece"],
            "slope_latent": latent["calibration_slope"],
        }
    )
cal = pd.DataFrame(cal_rows).set_index("condition")
cal.round(3)
"""
        ),
        code(
            """
iso_wins = (cal["mae_isotonic"] < cal["mae_ordinal"]).sum()
print(f"isotonic lower MAE in {iso_wins}/{len(cal)} conditions")
print("C2:", "KEEP BOTH (isotonic point, ordinal probabilities)" if iso_wins >= 3 else "REVIEW")
"""
        ),
        md(
            """
## C3 — Runtime evidence gate

**Hypothesis.** Calls with missing resolution evidence, sentiment/resolution conflict, or low
fact confidence are more likely to be badly scored, so gating on them should beat
confidence-only abstention.

**Pre-specified gate.** `audit_score = confidence − 0.20·missing_resolution
− 0.10·conflict − 0.15·(1−span_exists) − 0.10·low_fact_confidence`.

**Decision rule.** KEEP if the gate improves the large-error rate at 80% coverage by more than
0.02 absolute, or raises AUC by more than 0.05.
"""
        ),
        code(
            """
def add_gate(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["audit_gate"] = (
        out["confidence"]
        - 0.20 * out["missing_resolution_evidence"]
        - 0.10 * out["sentiment_resolution_conflict"]
        - 0.15 * (1.0 - out["span_exists"])
        - 0.10 * (out["fact_confidence"] < 0.55).astype(float)
    )
    out["large_error"] = (np.abs(out["pred_label"] - out["latent"]) > 1).astype(int)
    return out


def selective_rate(frame: pd.DataFrame, score_col: str, coverage: float = 0.80) -> float:
    n = int(len(frame) * coverage)
    order = np.argsort(-frame[score_col].to_numpy())[:n]
    return float(frame["large_error"].to_numpy()[order].mean())


gate_rows = []
for condition in ("source", "disfluency", "emotional_arc", "paraphrase"):
    frame = add_gate(strict_frames[condition])
    gate_rows.append(
        {
            "condition": condition,
            "auc_confidence": roc_auc_score(frame["large_error"], frame["confidence"]),
            "auc_gate": roc_auc_score(frame["large_error"], frame["audit_gate"]),
            "large_error_conf_80": selective_rate(frame, "confidence", 0.80),
            "large_error_gate_80": selective_rate(frame, "audit_gate", 0.80),
        }
    )
gate = pd.DataFrame(gate_rows).set_index("condition")
gate["delta_auc"] = gate["auc_gate"] - gate["auc_confidence"]
gate["delta_large_error"] = gate["large_error_gate_80"] - gate["large_error_conf_80"]
gate.round(3)
"""
        ),
        code(
            """
# The first pass used an "any condition" rule. It fired on a single 2.5 pp cell while
# losing AUC elsewhere, so we require a consistent improvement instead. This is recorded
# as a protocol deviation in EXPERIMENT_LOG.md.
consistent_gain = int((gate["delta_large_error"] < -0.005).sum())
consistent_loss = int((gate["delta_large_error"] > 0.005).sum())
c3_keep = consistent_gain >= 3 and consistent_loss == 0
print(
    f"conditions improved: {consistent_gain}/{len(gate)}; "
    f"conditions worsened: {consistent_loss}/{len(gate)}"
)
print("C3:", "KEEP" if c3_keep else "DISCARD (not consistent across conditions)")
"""
        ),
        md(
            """
## C4 — Direct small-LLM scoring pilot

The direct arm asks a local Qwen2.5-3B-Instruct model for a 1–5 score, a five-class
distribution, verbalised confidence, cited turns, and a quote. It is scored on the same 40
source-holdout calls as the symbolic arm and their paraphrase twins. This is a **feasibility
probe**, not a model comparison: one small open model, 40 calls per condition, temperature 0.

The audit question is whether the direct arm's citations and confidence are trustworthy enough
to support the same kind of evidence-chain audit. If not, the evidence-chain arm remains the
only auditable option in this pilot.
"""
        ),
        code(
            """
llm_metrics_path = ROOT / "results" / "runs" / "llm_pilot_direct_metrics.csv"
if llm_metrics_path.exists():
    llm_metrics = pd.read_csv(llm_metrics_path, index_col=0)
    display_cols = [
        "n",
        "qwk",
        "mae",
        "ece_probs",
        "ece_confidence",
        "mean_confidence",
        "cited_in_gold_rate",
        "quote_support_rate",
        "parse_rate",
        "mean_latency_s",
    ]
    display(llm_metrics[[c for c in display_cols if c in llm_metrics.columns]].round(3))
else:
    print("LLM pilot metrics not found; run scripts/run_llm_pilot.py first.")
"""
        ),
        code(
            """
# Paired comparison on the exact LLM call ids, so the two arms see the same calls.
llm_source_path = ROOT / "results" / "runs" / "llm_pilot_direct_source.csv"
if llm_metrics_path.exists() and llm_source_path.exists():
    llm_source = pd.read_csv(llm_source_path)
    ids = set(llm_source["call_id"])
    sym_rows = []
    for condition in ("source", "disfluency", "emotional_arc", "paraphrase"):
        subset = strict_frames[condition][strict_frames[condition]["call_id"].isin(ids)]
        if len(subset) == 0:
            continue
        metrics = summarize_frame(subset, "latent")
        sym_rows.append(
            {
                "condition": condition,
                "symbolic_qwk": metrics["qwk_raw"],
                "symbolic_ece": metrics["ece"],
                "symbolic_large_error": metrics["large_error_rate"],
            }
        )
    symbolic_subset = pd.DataFrame(sym_rows).set_index("condition")
    comparison = symbolic_subset.join(
        llm_metrics[["qwk", "ece_probs", "large_error_rate"]].rename(
            columns={
                "qwk": "llm_qwk",
                "ece_probs": "llm_ece",
                "large_error_rate": "llm_large_error",
            }
        ),
        how="inner",
    )
    display(comparison.round(3))
    evidence = llm_metrics.loc[
        [c for c in comparison.index if c in llm_metrics.index],
        ["cited_in_gold_rate", "quote_support_rate", "ece_confidence"],
    ]
    display(evidence.round(3))
else:
    print("Run the LLM pilot and keep its per-call CSV to enable the paired comparison.")
"""
        ),
        md(
            """
### Reading C4

The comparison that matters is **source vs paraphrase**. If the direct LLM holds its QWK under
paraphrase while the symbolic evidence chain collapses (0.098), the lexical failure is an
**extractor artefact**, not a property of the calibration layer. The cost is visible in the
same table: confidence ECE 0.48–0.59, mean confidence 0.85–0.87, and citation precision
0.53–0.60. The direct model is more robust but less auditable; the evidence-chain model is
more auditable but brittle to vocabulary. The next design should put a *robust* extractor
inside the auditable chain, not choose one arm over the other.

## Final ledger

The table below is the artefact to carry into the paper draft. It records what was tried, what
moved, and what was rejected. The prose summary lives in `EXPERIMENT_LOG.md`.
"""
        ),
        md(
            """
**Protocol deviation (C3).** The initial decision rule was “keep if any condition improves
by 2 pp or 0.05 AUC”. It fired on a single disfluency cell (−2.5 pp) while the gate lost AUC
there and worsened paraphrase by +2.5 pp. We tightened the rule to require improvement in at
least three of four conditions with no material regression, and recorded **DISCARD**. The raw
per-condition table is left visible so a reader can apply their own rule.
"""
        ),
        code(
            """
ledger = pd.DataFrame(
    [
        {
            "card": "C0",
            "change": "strict lexicon + ordinal/isotonic source calibration (baseline)",
            "evidence": "02 stress profile",
            "decision": "REFERENCE",
            "note": "source QWK is the optimistic ceiling for this DGP",
        },
        {
            "card": "C1",
            "change": "broaden lexicon with paraphrase synonyms",
            "evidence": f"paraphrase ΔQWK={target_gain:+.3f}, source ΔQWK={source_loss:+.3f}",
            "decision": "KEEP (diagnostic, circular)" if c1_keep else "DISCARD",
            "note": "lexicon was written from the paraphrase list; not a deployable fix",
        },
        {
            "card": "C2",
            "change": "isotonic point calibrator alongside ordinal probabilities",
            "evidence": f"isotonic lower MAE in {iso_wins}/{len(cal)} conditions",
            "decision": "KEEP BOTH" if iso_wins >= 3 else "REVIEW",
            "note": "isotonic for point scores; ordinal for ECE/selective/conformal",
        },
        {
            "card": "C3",
            "change": "runtime evidence gate for abstention",
            "evidence": (
                f"improved {consistent_gain}/{len(gate)}; "
                f"worsened {consistent_loss}/{len(gate)}; "
                f"max ΔAUC={gate['delta_auc'].max():+.3f}"
            ),
            "decision": "KEEP" if c3_keep else "DISCARD",
            "note": "presence flags are diagnostic but not a consistent deferral policy",
        },
        {
            "card": "C4",
            "change": "direct small-LLM scoring pilot",
            "evidence": "LLM table above",
            "decision": "PILOT ONLY",
            "note": "one 3B local model, 40 source calls + paraphrase twins; not a frontier-LLM result",
        },
    ]
)
ledger.to_csv(ROOT / "results" / "runs" / "03_ledger.csv", index=False)
ledger
"""
        ),
        md(
            """
## What the next iteration should test

1. **Replace the symbol oracle with a real LLM extractor** on the same 100-call holdouts, with
   inter-model agreement as an additional audit signal. The 3B pilot is too weak to settle this.
2. **Make the audit semantic, not presence-based.** The failure of C3 suggests that “no evidence”
   is often the scorer's honest default, while confident wrong evidence is the dangerous case.
   A contradiction check between cited span and typed fact is the next candidate.
3. **Test calibration under label-preserving shift only.** Emotional arc changes the construct;
   it should be analysed as concept shift, not as a calibration failure without qualification.
4. **Pre-register the tolerance for score shifts.** The 0.5 raw-score tolerance is coarse for
   sentiment-only changes; report label-change and expected-score deltas alongside it.
"""
        ),
    ]
    return nbf.v4.new_notebook(cells=cells)


def main() -> None:
    notebooks = {
        "00_protocol.ipynb": build_protocol(),
        "01_baseline.ipynb": build_baseline(),
        "02_shift_stress.ipynb": build_stress(),
        "03_iterations.ipynb": build_iterations(),
    }
    for name, notebook in notebooks.items():
        nbf.write(notebook, NB_DIR / name)
        print(f"wrote {name}")


if __name__ == "__main__":
    main()
