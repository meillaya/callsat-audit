# Preliminary report (LaTeX/TikZ)

Work-in-progress report for the Topic 1 pilot:

> Auditing Satisfaction, Not Just Accuracy: Calibration and Evidence Consistency of
> LLM-Based Satisfaction Scoring under Synthetic Call-Log Shift

The report is a **start**, not a submission-ready paper. It covers the motivation, related work,
study design (with a TikZ measurement-chain figure and a TikZ/pgfplots shift profile), the
preliminary results from the notebooks, a discussion, threats to validity, next steps, and the
references consulted.

## Files

| File | Purpose |
| --- | --- |
| `main.tex` | Report source, including all TikZ/pgfplots figures |
| `refs.bib` | BibTeX references |
| `main.pdf` | Compiled report |
| `callsat-audit-preliminary-report.pdf` | Same PDF under a descriptive name |

## Compile

Tectonic is used because it is self-contained and available in this environment:

```bash
cd report
TECTONIC_CACHE_DIR="$PWD/../.tectonic-cache" \
  tectonic -X compile main.tex --outdir . --keep-logs
```

The cache directory must be writable; the default `~/Library/Caches/Tectonic...` is blocked in
this workspace sandbox. On a normal TeX Live installation, `pdflatex + bibtex + pdflatex x2`
works as well.

## Reference provenance

Bibliographic metadata was checked on 20 October 2026 against ACL Anthology, Crossref, IEEE
Xplore, OpenAlex, and arXiv:

- Mirage: ACL Anthology / Crossref, DOI `10.18653/v1/2026.findings-acl.1261`.
- Conversational data survey: Crossref, DOI `10.1145/3795686`.
- Customer support conversation: Crossref, DOI `10.1609/aaai.v40i41.40825`.
- BellTeus: Crossref, DOI `10.1109/ICCC68654.2025.11438128`.
- Empathic communication: Crossref, DOI `10.1038/s42256-025-01169-6`.
- Interpretable satisfaction: ACL Anthology / Crossref, DOI `10.18653/v1/2026.sigdial-1.40`.
- CARE calibration cliff: Crossref, DOI `10.1109/ACCESS.2026.3721881`.
- pCSAT: arXiv `2411.12539`.
- Evidence-in-the-Loop: arXiv `2607.18039`.
- Qwen2.5: arXiv `2412.15115`.

The `call-eval` repository returned 404 on 20 October 2026. The report cites the archived
project guide (`https://jevlist.ai/projects/call-eval`) rather than the unavailable code.

## Relationship to the experiment log

`../EXPERIMENT_LOG.md` is the authoritative decision record. The report's tables are generated
from `../results/runs/*.csv`; if the notebooks are rerun, update the tables from the new CSVs.
The report deliberately keeps the LLM pilot scoped to source + paraphrase (40 calls each) and
labels it as a feasibility probe.
