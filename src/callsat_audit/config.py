"""Shared constants and paths for the satisfaction-audit pilot."""

from __future__ import annotations

from pathlib import Path
from typing import Final, Literal

ROOT: Final = Path(__file__).resolve().parents[2]
DATA_DIR: Final = ROOT / "results" / "data"
RUNS_DIR: Final = ROOT / "results" / "runs"
FIGURES_DIR: Final = ROOT / "results" / "figures"
CACHE_DIR: Final = ROOT / "results" / "cache"

PERSONAS: Final = (
    "anxious_elderly",
    "busy_professional",
    "angry_returning",
    "calm_first_timer",
    "vulnerable_financial",
)

ISSUES: Final = (
    "billing_dispute",
    "refund_request",
    "technical_support",
    "account_access",
    "cancellation",
)

CONDITIONS: Final = (
    "source",
    "disfluency",
    "emotional_arc",
    "persona_mix",
    "paraphrase",
    "combined",
)

SPLITS: Final = ("train", "dev", "holdout")

LABELS: Final = (1, 2, 3, 4, 5)

Resolution = Literal["resolved", "partial", "unresolved"]
Split = Literal["train", "dev", "holdout"]
Condition = Literal[
    "source", "disfluency", "emotional_arc", "persona_mix", "paraphrase", "combined"
]

# Source-generating-process sizes.  The 100-call holdout is never used for fitting.
N_TRAIN: Final = 240
N_DEV: Final = 80
N_HOLDOUT: Final = 100
N_SHIFT: Final = 100

SEED: Final = 20261020

# Pre-registered decision thresholds for the keep/discard loop.
KEEP_ECE_IMPROVEMENT: Final = 0.03
KEEP_QWK_TOLERANCE: Final = 0.02
SCORE_SHIFT_TOLERANCE: Final = 0.5
LARGE_ERROR_THRESHOLD: Final = 1.0
