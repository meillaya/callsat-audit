# Experiment log — Topic 1 preliminary study

**Status:** preliminary / pilot. Synthetic data only. No claim here is about real callers,
real agents, or frontier LLMs.
**Frozen dataset:** `results/data/*.jsonl` (seed `20261020`), 420 source calls and 100 calls per
shift condition.
**Pipeline snapshot:** `src/callsat_audit`, notebooks `00`–`03`.

> **Protocol note.** An earlier run of the stress notebook accidentally included the source
> train/dev calls (used to fit the calibrator) alongside the sealed holdout. That inflated the
> source QWK from 0.718 to 0.748 and the ECE from 0.050 to 0.082. The bug is fixed in
> `02_shift_stress.ipynb` and `03_iterations.ipynb`; all numbers below are holdout-only for the
> source and out-of-sample for every target.

## 1. Question and design

The object of study is a call-eval-style measurement chain:

```
transcript -> cited evidence spans -> typed facts -> deterministic score -> survey calibration
```

The pilot asks whether that chain stays calibrated and auditable when the synthetic
data-generating process shifts. Four paired shifts are generated from the same source holdout
calls (latent label and survey held fixed):

| Shift | What changes | Why it matters |
| --- | --- | --- |
| `disfluency` | fillers, repetitions, ASR noise inside cue spans | label-preserving measurement noise |
| `emotional_arc` | late frustration after resolution, or suppressed dissatisfaction | concept shift: expressed sentiment decouples from outcome |
| `paraphrase` | meaning-preserving synonym replacements | lexical shift; tests span robustness |
| `combined` | paraphrase + arc + disfluency | realistic worst case |
| `persona_mix` | different persona weights, new calls | covariate shift / base-rate change |

Calibration is fitted on source **train+dev surveys only** (n=227), with temperature and
conformal thresholds fitted out-of-fold. The 100-call source holdout and all shifted targets are
never used for fitting.

## 2. Headline results (holdout, target = latent satisfaction)

| Condition | QWK raw | ECE | Cal. slope | Large-error rate | Conformal 90% coverage |
| --- | ---: | ---: | ---: | ---: | ---: |
| source | 0.718 | 0.050 | 0.985 | 0.220 | 0.976 |
| disfluency | 0.661 | 0.071 | 1.040 | 0.280 | 0.980 |
| emotional_arc | 0.697 | 0.065 | 1.018 | 0.270 | 0.970 |
| paraphrase | **0.098** | **0.158** | 1.341 | 0.440 | **0.760** |
| combined | **0.029** | **0.188** | **0.019** | 0.590 | **0.750** |
| persona_mix | 0.782 | 0.063 | 0.979 | 0.270 | 0.980 |

Bootstrap 95% CIs (1,000 resamples, seed 0): source QWK **0.718 [0.605, 0.802]**, disfluency
**0.661 [0.543, 0.754]**, emotional arc **0.697 [0.582, 0.783]**, paraphrase **0.098
[0.025, 0.191]**, combined **0.029 [−0.014, 0.082]**, persona mix **0.782 [0.693, 0.848]**.
ECE CIs are wide and overlap across source and the mild shifts: source **0.050 [0.031, 0.164]**,
disfluency **0.071 [0.026, 0.164]**, emotional arc **0.065 [0.032, 0.168]**, paraphrase
**0.158 [0.088, 0.247]**, combined **0.188 [0.107, 0.259]**, persona mix **0.063 [0.029, 0.159]**.

Paired ΔQWK (source − target): disfluency **+0.057 [−0.022, 0.149]**, emotional arc **+0.021
[−0.015, 0.062]**, paraphrase **+0.619 [0.474, 0.738]**, combined **+0.689 [0.570, 0.789]**.
Only the two lexical shifts exclude zero.

Against the simulated survey (the target the calibrator actually sees), source QWK is **0.502**
and ECE **0.128**; against latent it is QWK **0.718** / ECE **0.050**. The calibration slope
against survey is **0.688**, against latent **0.985**. The large-error rate against survey is
**0.384** (vs 0.220 against latent).

### Evidence-chain localisation

| Condition | Fact errors / call | Resolution acc. | Sentiment acc. | Gold turn hit | Quote supported |
| --- | ---: | ---: | ---: | ---: | ---: |
| source | 0.51 | 0.790 | 0.940 | 0.848 | 1.000 |
| disfluency | 0.91 | 0.650 | 0.940 | 0.748 | 1.000 |
| emotional_arc | 0.83 | 0.790 | **0.620** | 0.867 | 1.000 |
| paraphrase | **2.61** | 0.190 | 0.360 | **0.082** | **0.260** |
| combined | **2.54** | 0.170 | 0.480 | 0.247 | 0.690 |
| persona_mix | 0.48 | 0.810 | 0.920 | 0.833 | 0.990 |

Paired consistency:

| Shift | Mean fact changes | Score shift > 0.5 | Label change | Evidence-turn Jaccard |
| --- | ---: | ---: | ---: | ---: |
| disfluency | 0.43 | 0.14 | 0.23 | 0.915 |
| emotional_arc | 0.48 | 0.00 | 0.17 | 0.970 |
| paraphrase | 2.15 | 0.75 | 0.76 | **0.065** |
| combined | 2.38 | 0.76 | 0.77 | 0.380 |

### Audit signals for large errors (`|error| > 1`)

| Condition | AUC confidence | AUC missing resolution | AUC conflict | AUC low fact confidence |
| --- | ---: | ---: | ---: | ---: |
| source | 0.372 | **0.589** | 0.467 | 0.522 |
| disfluency | 0.275 | **0.624** | 0.477 | 0.602 |
| emotional_arc | 0.354 | **0.648** | 0.426 | 0.525 |
| paraphrase | 0.478 | 0.534 | — | 0.494 |
| combined | 0.567 | 0.520 | — | 0.483 |
| persona_mix | 0.439 | 0.524 | 0.557 | 0.506 |

Posterior confidence is **anti-predictive** on source, disfluency, and emotional arc (AUC < 0.5).
Missing-resolution flags beat confidence on the source, disfluency, and emotional-arc targets,
but not on paraphrase or combined. AUC is a ranking diagnostic, not a deferral policy; see C3.

## 3. Keep/discard decisions

### C0 — Strict lexicon + source calibration — **REFERENCE**

Source holdout QWK 0.718, ECE 0.050, slope 0.985. This is the optimistic ceiling for this DGP,
not a result about real calls.

### C1 — Broaden the lexicon with paraphrase synonyms — **KEEP (diagnostic, circular)**

| Metric | Strict | Broad |
| --- | ---: | ---: |
| paraphrase QWK | 0.098 | **0.718** |
| combined QWK | 0.029 | 0.635 |
| source QWK | 0.718 | 0.718 |
| paraphrase fact errors | 2.61 | 0.51 |
| paraphrase gold-turn hit | 0.082 | 0.848 |

The repair was written by looking at the paraphrase list, so it is an **oracle upper bound**, not
a deployable fix. Its value is diagnostic: it shows the paraphrase collapse is a span-layer
failure, not an irreducible calibration problem.

### C2 — Isotonic point calibration alongside ordinal probabilities — **KEEP BOTH**

Isotonic has lower MAE in **4/6** conditions (source 0.624 vs 0.631; disfluency 0.651 vs 0.692;
paraphrase 0.946 vs 1.004; persona 0.630 vs 0.637). It is slightly worse on emotional arc
(0.668 vs 0.666) and combined (1.086 vs 1.082). The ordinal layer is retained because isotonic
cannot supply class probabilities for ECE, Brier, selective prediction, or conformal sets.

### C3 — Runtime evidence gate for abstention — **DISCARD**

Pre-specified gate:
`confidence − 0.20·missing_resolution − 0.10·conflict − 0.15·(1−span_exists) − 0.10·low_fact_confidence`.

At 80% coverage the gate changed the large-error rate by 0.0 pp (source), −2.5 pp (disfluency),
−1.2 pp (emotional arc), and **+2.5 pp** (paraphrase). AUC did not improve (max ΔAUC +0.015).
A first-pass “any condition” rule fired on the single disfluency cell; we tightened it to
require consistency across conditions and recorded the deviation here.

**Interpretation:** “no evidence” is often the scorer's honest default, not the dangerous case.
The dangerous case is *confident wrong evidence*; a presence flag cannot see it. A semantic
contradiction check (cited span vs typed fact) is the next candidate, not more presence gating.

### C4 — Direct small-LLM scoring pilot — **PILOT ONLY**

Local `Qwen2.5-3B-Instruct-4bit` via MLX, temperature 0, the same 40 source-holdout calls and
their paraphrase twins. This is one small open model, not a frontier LLM; it is a feasibility
probe for the audit instrument.

| Arm | Source QWK | Paraphrase QWK | Source large-error | Paraphrase large-error | Source ECE (probs) | Paraphrase ECE (probs) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| symbolic chain (same 40 calls) | 0.746 | **0.113** | 0.200 | 0.450 | 0.090 | 0.199 |
| direct 3B | 0.669 | **0.667** | 0.150 | 0.150 | 0.575 | 0.593 |

Direct-arm audit: mean verbalised confidence **0.874** (source) / **0.854** (paraphrase);
confidence ECE **0.498 / 0.478**; cited turns hit a gold cue turn on **97.5% / 92.5%** of calls;
quotes are supported **95% / 90%**; but citation precision is only **0.604 / 0.533**. Parse rate
is 1.000 / 0.975.

**Decision:** keep the direct arm as a pilot, do not treat its numbers as a model comparison.
The contrast is the finding: the direct LLM is **robust to lexical shift** (QWK stable at
0.667) while the symbolic evidence chain collapses (0.113). But the LLM is badly overconfident
and its citations are often real-but-wrong. This points to a robust extractor inside an
auditable chain, not to abandoning either arm.

## 4. Preliminary claims (for the paper draft, not final)

1. **Lexical shift is an extractor-level failure, not a calibration-layer failure.** The
   symbolic chain loses its evidence spans under paraphrase (gold-turn hit 0.848 → 0.082, fact
   errors 0.51 → 2.61), and paired ΔQWK is 0.619 [0.474, 0.738]. Post-hoc isotonic calibration
   recovers QWK only to 0.107. The direct 3B model, which never extracts a typed fact, holds QWK
   at 0.667. The fragile component is the span/fact interface.
2. **ECE is too blunt for this audit at n=100.** ECE point estimates rise in the expected
   direction, but the source interval [0.031, 0.164] overlaps every mild-shift interval and even
   the paraphrase interval [0.088, 0.247]. Discrimination (QWK), evidence recovery, and
   conformal coverage give sharper separation: coverage falls from 0.976 to 0.760 under
   paraphrase, and to 0.750 under combined. Report the profile, not a single ECE number.
3. **Survey anchoring is not construct validation.** On the same source holdout, QWK is 0.718
   against latent satisfaction but 0.502 against the noisy survey; ECE is 0.050 vs 0.128; the
   survey-calibrated slope is 0.688. The calibration layer is doing what it was asked to do
   (fit surveys), not what a service manager would assume it does (measure satisfaction).
4. **Evidence-presence flags are diagnostic, not a consistent deferral policy.** Missing-evidence
   AUC beats confidence on source (0.589 vs 0.372), disfluency (0.624 vs 0.275), and emotional
   arc (0.648 vs 0.354), but the gate improves selective risk in only two of four conditions and
   worsens paraphrase by 2.5 pp. The audit needs a contradiction/support check, not a presence
   check.
5. **Robustness and auditability trade off in this pilot.** The direct LLM is robust to
   paraphrase but poorly calibrated (confidence ECE ≈ 0.5) and weakly traceable (citation
   precision ≈ 0.53–0.60). The symbolic chain is traceable and calibrated on source but
   collapses under vocabulary change. The next design should put a robust (LLM) extractor inside
   the deterministic, calibrated, evidence-carrying chain.

## 5. Threats to validity

- **Small targets, wide intervals.** One hundred calls per shift gives QWK 95% CIs of roughly
  ±0.06–0.10. Only paraphrase and combined have paired ΔQWK CIs excluding zero. Treat
  disfluency and emotional-arc effects as directional, not established.
- **Synthetic ground truth is the generator.** “Latent satisfaction” is a structural equation,
  not a human judgment. The survey is a noisy draw from it. All calibration numbers inherit that.
- **The offline extractor is a lower bound, not an LLM.** It was written against the templates;
  the source fact accuracy is optimistic and the paraphrase collapse is partly a property of the
  lexicon. The broad-lexicon repair is circular by construction.
- **The LLM pilot is one 3B model on 40 calls per condition.** It cannot rank models or support
  a claim about GPT-class systems. It is a direct scorer, not the LLM-extractor arm; the
  evidence chain with an LLM extractor remains unrun.
- **The shifts are synthetic transformations.** They operationalise Mirage-style dimensions but
  do not reproduce real disfluency, code-switching, accents, or emotional expression.
- **The 0.5 raw-score tolerance is coarse** for sentiment-only changes; label-change and
  expected-score deltas are reported alongside it.
- **The conformal sets over-cover on source (0.976 vs nominal 0.90)** because the base model is
  refit on the full calibration set after the out-of-fold threshold. Coverage failures under
  shift are still informative, but the source baseline is conservative.

## 6. Next experiments

1. **LLM extractor inside the chain.** Run a frontier model through the typed-fact prompt on the
   same 100-call holdouts, then score deterministically. Test whether it keeps the direct arm's
   paraphrase robustness while retaining evidence spans. Compare with the symbolic and direct
   arms on QWK, ECE, coverage, and citation precision.
2. **Contradiction audit.** For each cited span, ask whether it entails the typed fact (NLI or a
   second LLM judge). Test whether contradiction rate predicts large errors better than
   presence flags. This directly addresses C3's failure.
3. **Frontier direct scorer with proper probability elicitation.** Compare verbalised histograms
   with token logprobs; fit temperature on a source dev split; test whether confidence ECE
   survives shift.
4. **Human-adjudicated subset (30–50 calls).** Test whether the survey ceiling is an artefact of
   the simulated annotators.
5. **Pre-register the tolerance and minimum detectable effect.** The current 100-call targets
   are underpowered for the mild shifts; the next run should size the target for a paired ΔQWK
   CI that excludes zero under disfluency, or report it as inconclusive.
