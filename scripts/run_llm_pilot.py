"""Run the local open-model direct-scoring pilot and cache every response."""

from __future__ import annotations

import argparse
import json
import random

from callsat_audit.config import ROOT, SEED
from callsat_audit.generate import load_datasets
from callsat_audit.llm_pipeline import direct_frame, direct_metrics, score_all
from callsat_audit.llm_scorer import LocalMlxDirectScorer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="mlx-community/Qwen2.5-3B-Instruct-4bit")
    parser.add_argument("--n", type=int, default=40)
    parser.add_argument(
        "--conditions",
        default="source,disfluency,emotional_arc,paraphrase,persona_mix",
    )
    args = parser.parse_args()

    datasets = load_datasets()
    rng = random.Random(SEED)
    holdout_ids = [c.call_id for c in datasets["source"] if c.split == "holdout"]
    chosen = set(rng.sample(holdout_ids, min(args.n, len(holdout_ids))))
    pm_ids = set(
        rng.sample(
            [c.call_id for c in datasets["persona_mix"]], min(args.n, len(datasets["persona_mix"]))
        )
    )

    scorer = LocalMlxDirectScorer(args.model)
    frames = {}
    for condition in args.conditions.split(","):
        calls = datasets[condition]
        if condition == "persona_mix":
            subset = [c for c in calls if c.call_id in pm_ids]
        else:
            subset = [c for c in calls if c.call_id in chosen]
        print(f"[{condition}] scoring {len(subset)} calls", flush=True)
        scores = score_all(subset, scorer, condition)
        frame = direct_frame(subset, scores)
        frames[condition] = frame
        metrics = direct_metrics({condition: frame}, target="latent").iloc[0].to_dict()
        print(condition, json.dumps({k: round(v, 4) for k, v in metrics.items()}), flush=True)

    out_dir = ROOT / "results" / "runs"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = direct_metrics(frames, target="latent")
    summary.to_csv(out_dir / "llm_pilot_direct_metrics.csv")
    for condition, frame in frames.items():
        frame.to_csv(out_dir / f"llm_pilot_direct_{condition}.csv", index=False)
    print(summary.round(4).to_string(), flush=True)


if __name__ == "__main__":
    main()
