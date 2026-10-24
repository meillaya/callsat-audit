"""Synthetic call-log generator with known latent satisfaction and gold evidence cues.

The generator is intentionally *structural*: a small set of call facts (resolution,
caller effort, expressed sentiment, agent empathy) drives both the transcript cues and
the latent satisfaction.  That makes it possible to separate three failure loci:

1. span level   -- did the extractor find the cue?
2. fact level   -- did it turn the cue into the right typed fact?
3. calibration  -- did the fact-to-score mapping stay calibrated under shift?

Every call is synthetic.  No claim here is about real callers.
"""

from __future__ import annotations

import copy
import json
import random
import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from .config import (
    DATA_DIR,
    N_DEV,
    N_HOLDOUT,
    N_SHIFT,
    N_TRAIN,
    SEED,
    Condition,
    Resolution,
    Split,
)
from .schema import CallLog, Cue, CueKind, Speaker, Turn

CUTS = (-0.85, -0.15, 0.45, 1.05)
SENT_CUTS = (-1.20, -0.40, 0.40, 1.20)

PERSONA_WEIGHTS_SOURCE = {
    "anxious_elderly": 0.18,
    "busy_professional": 0.22,
    "angry_returning": 0.20,
    "calm_first_timer": 0.25,
    "vulnerable_financial": 0.15,
}
PERSONA_WEIGHTS_SHIFT = {
    "anxious_elderly": 0.05,
    "busy_professional": 0.10,
    "angry_returning": 0.35,
    "calm_first_timer": 0.05,
    "vulnerable_financial": 0.45,
}

PERSONA_LINES = {
    "anxious_elderly": "Sorry, I'm not very good with these online forms.",
    "busy_professional": "I've got about five minutes before my next meeting.",
    "angry_returning": "This is the third time I've called about it.",
    "calm_first_timer": "I'm not sure if this is the right department, but I hope you can help.",
    "vulnerable_financial": "I really can't afford another charge this month.",
}

ISSUE_LINES = {
    "billing_dispute": "There's a charge on my last bill that I don't recognize.",
    "refund_request": "I'm still waiting on a refund I asked for two weeks ago.",
    "technical_support": "My internet keeps dropping every evening around eight.",
    "account_access": "I'm locked out of my account and the reset link isn't working.",
    "cancellation": "I want to cancel my subscription, but I keep getting charged.",
}

ISSUE_DETAILS = {
    "billing_dispute": "It's a {amount} charge labelled '{label}'.",
    "refund_request": "It was {amount}, returned on the {day}.",
    "technical_support": "The router lights go red and then the connection drops.",
    "account_access": "I've tried the reset link three times and it just loops.",
    "cancellation": "I was told the last payment would be the final one.",
}

AMOUNTS = ("$45.00", "$59.99", "$89.00", "$120.50", "$24.99")
LABELS = ("service fee", "equipment rental", "late payment", "premium support")
DAYS = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday")

GREETINGS = (
    "Thanks for calling Meridian Support. This is {agent}. How can I help today?",
    "Meridian Customer Care, {agent} speaking. What can I do for you?",
    "Hello, you've reached Meridian. My name is {agent}. How can I help?",
)
AGENTS = ("Dana", "Priya", "Marcus", "Elena", "Sam", "Noor")

VERIFY = (
    "Can you confirm the account number for me?",
    "Could you verify the postcode on the account?",
    "Let me pull up the account. Can I take your name and date of birth?",
)
DIAGNOSIS = (
    "I can see the account now. Thanks for verifying that.",
    "Okay, I've found the record. Let me take a look.",
    "I can see what you mean. Give me a moment.",
)

EMPATHY_LINES = (
    "I completely understand how frustrating this must be.",
    "I'm sorry you've had to deal with that.",
    "I appreciate your patience while we sort this out.",
)

EFFORT_LINES = (
    "I'll need to put you on a brief hold while I check that.",
    "I'll have to transfer you to our billing team for this.",
    "You'll need to call back tomorrow with the reference number.",
)

RESOLUTION_LINES: dict[Resolution, tuple[str, ...]] = {
    "resolved": (
        "I've processed the refund for you today.",
        "I've reversed the charge and updated the account.",
        "I've applied the credit and sent you a confirmation email.",
    ),
    "partial": (
        "I've escalated this to the billing team; they'll follow up within three business days.",
        "I've submitted the request, but it still needs a supervisor's approval.",
        "I've opened a ticket and added a note to the account.",
    ),
    "unresolved": (
        "I'm sorry, but there's nothing more I can do on this account.",
        "You're not eligible for the refund under the current policy.",
        "I can't override that charge from my side.",
    ),
}

SENTIMENT_LINES = {
    2: "Thank you, that's a huge relief.",
    1: "Alright, that helps. Thanks for your help.",
    0: "Okay. I guess I'll wait and see.",
    -1: "I'm not really happy with that.",
    -2: "This is ridiculous. I've been passed around all day.",
}

LATE_FRUSTRATION = (
    "Honestly, I'm still annoyed it took three calls to sort out.",
    "I appreciate it, but this has taken far too long.",
)
SUPPRESSED = (
    "That's fine. Thank you for your help.",
    "Okay, thanks. I'll just wait and see, I suppose.",
)

IMPLICIT_RESOLUTION: dict[Resolution, tuple[str, ...]] = {
    "resolved": (
        "Okay, I've taken care of it on my end.",
        "You should see the change reflected shortly.",
    ),
    "partial": (
        "I've passed it along; someone will be in touch.",
        "It's with the right team now.",
    ),
    "unresolved": (
        "Unfortunately, that's not something we can do.",
        "That's outside what I'm able to offer.",
    ),
}

IMPLICIT_EMPATHY = (
    "I hear you.",
    "That does sound difficult.",
    "Okay, I've got you.",
)

IMPLICIT_SENTIMENT = {
    2: ("That's more like it, thank you.",),
    1: ("Okay, that's a load off.",),
    0: ("Hmm. Right.",),
    -1: ("I was hoping for a different answer.",),
    -2: ("I can't believe I have to deal with this.",),
}

IMPLICIT_EFFORT = (
    "I'll just need to check something with a colleague.",
    "Let me see if another team can look at this.",
)

DISTRACTORS_AGENT = (
    "I can't see the refund yet, but let me check the payment history.",
    "Thanks for verifying that so quickly.",
    "The previous agent left a note on the account.",
)
DISTRACTORS_CUSTOMER = (
    "I don't want to cancel the whole account, just fix the charge.",
    "I'm sorry, I didn't catch the last part.",
    "The previous agent said the same thing.",
)

CLOSINGS_AGENT = (
    "Is there anything else I can help with today?",
    "Anything else I can do for you?",
    "I'll send a summary to the email on file. Is there anything else?",
)
CLOSINGS_CUSTOMER = (
    "No, that's all. Thanks.",
    "That's everything. Thank you.",
    "No, I think we're done.",
)

# Paraphrases that keep the meaning but remove the surface cue the offline lexicon uses.
PARAPHRASES = {
    "I've processed the refund for you today.": "The refund has been issued to your card today.",
    "I've reversed the charge and updated the account.": "The charge has been backed out and your account is current.",
    "I've applied the credit and sent you a confirmation email.": "A credit is now on your account; you'll get an email receipt.",
    "I've escalated this to the billing team; they'll follow up within three business days.": "This is now with the billing team, who will be in touch within three working days.",
    "I've submitted the request, but it still needs a supervisor's approval.": "The request is in, though a supervisor still has to sign it off.",
    "I've opened a ticket and added a note to the account.": "There's a ticket on your record now with the details.",
    "I'm sorry, but there's nothing more I can do on this account.": "My hands are tied on this account, unfortunately.",
    "You're not eligible for the refund under the current policy.": "The current policy doesn't cover that refund, I'm afraid.",
    "I can't override that charge from my side.": "That charge isn't something I'm able to change from here.",
    "I'll need to put you on a brief hold while I check that.": "I'll ask you to hang on for a moment while I look into it.",
    "I'll have to transfer you to our billing team for this.": "I'll put you through to the billing team for this one.",
    "You'll need to call back tomorrow with the reference number.": "Please ring us again tomorrow with the reference number.",
    "I completely understand how frustrating this must be.": "I can see why this would be so annoying.",
    "I'm sorry you've had to deal with that.": "That sounds like it's been a real hassle.",
    "I appreciate your patience while we sort this out.": "Thanks for bearing with us while this gets sorted.",
    "Thank you, that's a huge relief.": "That's a weight off my mind, thank you.",
    "Alright, that helps. Thanks for your help.": "Okay, that's better. Cheers.",
    "Okay. I guess I'll wait and see.": "Right. I suppose I'll see what happens.",
    "I'm not really happy with that.": "That's not a great outcome from where I'm sitting.",
    "This is ridiculous. I've been passed around all day.": "This is beyond frustrating; I've been bounced around all day.",
    "Honestly, I'm still annoyed it took three calls to sort out.": "To be honest, needing three calls for this is pretty poor.",
    "I appreciate it, but this has taken far too long.": "Thanks, though this really shouldn't have taken so long.",
    "That's fine. Thank you for your help.": "No problem at all, thanks for looking into it.",
    "Okay, thanks. I'll just wait and see, I suppose.": "Alright then, I'll see what happens.",
}

ASR_MANGLES = {
    "refund": "refun",
    "account": "acount",
    "billing": "biling",
    "cancel": "kancel",
    "confirmation": "confirmaton",
    "supervisor": "superviser",
    "reference": "referance",
    "transfer": "tranfer",
}


@dataclass(slots=True)
class CallSpec:
    persona: str
    issue: str
    resolution: Resolution
    effort: int
    empathy: int
    sentiment: int
    agent: str
    prior_failures: int
    wait_minutes: int
    agent_style: str
    latent_z: float
    latent: int
    survey: int | None
    annotator_a: int
    annotator_b: int


def _clip_label(value: float) -> int:
    return max(1, min(5, round(value)))


def _ordinal_from_z(z: float, cuts: tuple[float, ...] = CUTS) -> int:
    return 1 + sum(z > c for c in cuts)


def _sample_persona(rng: random.Random, weights: dict[str, float]) -> str:
    return rng.choices(list(weights), weights=list(weights.values()), k=1)[0]


def _resolution_probs(persona: str, issue: str) -> tuple[float, float, float]:
    base = {
        "anxious_elderly": 0.58,
        "busy_professional": 0.62,
        "angry_returning": 0.45,
        "calm_first_timer": 0.72,
        "vulnerable_financial": 0.50,
    }[persona]
    base += {
        "billing_dispute": 0.00,
        "refund_request": 0.03,
        "technical_support": -0.05,
        "account_access": 0.06,
        "cancellation": -0.08,
    }[issue]
    resolved = min(0.82, max(0.30, base))
    partial = 0.22
    unresolved = max(0.05, 1.0 - resolved - partial)
    total = resolved + partial + unresolved
    return resolved / total, partial / total, unresolved / total


def sample_spec(rng: random.Random, weights: dict[str, float]) -> CallSpec:
    persona = _sample_persona(rng, weights)
    issue = rng.choice(list(ISSUE_LINES))
    p_res, p_part, p_unres = _resolution_probs(persona, issue)
    resolution = cast(
        Resolution,
        rng.choices(["resolved", "partial", "unresolved"], weights=[p_res, p_part, p_unres], k=1)[
            0
        ],
    )

    if resolution == "resolved":
        effort = rng.choices([0, 1, 2], weights=[0.65, 0.30, 0.05], k=1)[0]
    elif resolution == "partial":
        effort = rng.choices([1, 2, 3], weights=[0.40, 0.45, 0.15], k=1)[0]
    else:
        effort = rng.choices([2, 3], weights=[0.40, 0.60], k=1)[0]
    if persona == "busy_professional":
        effort = min(3, effort + 1)

    agent_style = rng.choices(
        ["empathetic", "transactional", "rushed"], weights=[0.55, 0.30, 0.15], k=1
    )[0]
    p_empathy = {"empathetic": 0.80, "transactional": 0.30, "rushed": 0.12}[agent_style]
    empathy = int(rng.random() < p_empathy)

    prior_failures = int(rng.random() < (0.85 if persona == "angry_returning" else 0.22))
    wait_minutes = rng.randint(2, 45)

    res_num = {"resolved": 1.0, "partial": 0.5, "unresolved": 0.0}[resolution]
    persona_shift = {
        "anxious_elderly": -0.10,
        "busy_professional": -0.05,
        "angry_returning": -0.18,
        "calm_first_timer": 0.12,
        "vulnerable_financial": -0.12,
    }[persona]
    z = (
        1.35 * (res_num - 0.55)
        + 0.38 * empathy
        + 0.22 * (1.0 - effort / 3.0)
        - 0.28 * prior_failures
        - 0.18 * (wait_minutes / 45.0)
        + persona_shift
        + rng.gauss(0.0, 0.42)
    )
    latent = _ordinal_from_z(z)

    p_respond = max(0.50, min(0.90, 0.72 + 0.04 * (latent - 3)))
    survey = _clip_label(latent + rng.gauss(0.0, 0.50)) if rng.random() < p_respond else None
    annotator_a = _clip_label(latent + rng.gauss(0.0, 0.55))
    annotator_b = _clip_label(latent + rng.gauss(0.0, 0.55))

    sent_z = 0.85 * (z - 0.25) + rng.gauss(0.0, 0.65)
    sentiment = sum(sent_z > c for c in SENT_CUTS) - 2  # map 0..4 -> -2..2

    return CallSpec(
        persona=persona,
        issue=issue,
        resolution=resolution,
        effort=effort,
        empathy=empathy,
        sentiment=sentiment,
        agent=rng.choice(AGENTS),
        prior_failures=prior_failures,
        wait_minutes=wait_minutes,
        agent_style=agent_style,
        latent_z=z,
        latent=latent,
        survey=survey,
        annotator_a=annotator_a,
        annotator_b=annotator_b,
    )


def _add_turn(
    turns: list[Turn],
    speaker: Speaker,
    text: str,
    cue: tuple[CueKind, str, int] | None = None,
) -> Turn:
    """Append a turn and optionally attach one gold cue (kind, text, polarity)."""
    turn = Turn(turn_id=len(turns) + 1, speaker=speaker, text=text)
    if cue is not None:
        kind, cue_text, polarity = cue
        turn.cues.append(
            Cue(
                cue_id=f"t{turn.turn_id}:{kind}",
                turn_id=turn.turn_id,
                kind=kind,
                polarity=polarity,
                text=cue_text,
            )
        )
    turns.append(turn)
    return turn


def _render_source(spec: CallSpec, rng: random.Random) -> list[Turn]:
    issue_line = ISSUE_LINES[spec.issue]
    detail = ISSUE_DETAILS[spec.issue].format(
        amount=rng.choice(AMOUNTS), label=rng.choice(LABELS), day=rng.choice(DAYS)
    )
    turns: list[Turn] = []

    _add_turn(turns, "agent", GREETINGS[rng.randrange(len(GREETINGS))].format(agent=spec.agent))
    _add_turn(turns, "customer", f"{issue_line} {PERSONA_LINES[spec.persona]}")
    _add_turn(turns, "agent", rng.choice(VERIFY))
    _add_turn(turns, "customer", detail)
    _add_turn(turns, "agent", rng.choice(DIAGNOSIS))

    if rng.random() < 0.55:
        _add_turn(turns, "agent", rng.choice(DISTRACTORS_AGENT))
    if rng.random() < 0.45:
        _add_turn(turns, "customer", rng.choice(DISTRACTORS_CUSTOMER))

    for i in range(spec.effort):
        line = EFFORT_LINES[min(i, len(EFFORT_LINES) - 1)]
        _add_turn(turns, "agent", line, cue=("effort", line, 1))

    resolution_line = rng.choice(RESOLUTION_LINES[spec.resolution])
    resolution_polarity = {"resolved": 1, "partial": 0, "unresolved": -1}[spec.resolution]

    if spec.empathy:
        empathy_line = rng.choice(EMPATHY_LINES)
        _add_turn(
            turns,
            "agent",
            f"{empathy_line} {resolution_line}",
            cue=("empathy", empathy_line, 1),
        )
        turns[-1].cues.append(
            Cue(
                cue_id=f"t{turns[-1].turn_id}:resolution",
                turn_id=turns[-1].turn_id,
                kind="resolution",
                polarity=resolution_polarity,
                text=resolution_line,
            )
        )
    else:
        _add_turn(
            turns,
            "agent",
            resolution_line,
            cue=("resolution", resolution_line, resolution_polarity),
        )

    sentiment_line = SENTIMENT_LINES[spec.sentiment]
    _add_turn(turns, "customer", sentiment_line, cue=("sentiment", sentiment_line, spec.sentiment))

    _add_turn(turns, "agent", rng.choice(CLOSINGS_AGENT))
    _add_turn(turns, "customer", rng.choice(CLOSINGS_CUSTOMER))
    return turns


def _apply_cue_dropout(turns: list[Turn], spec: CallSpec, rng: random.Random) -> dict[str, bool]:
    """Replace some explicit cues with implicit, non-lexical evidence.

    Gold facts are unchanged: the information is still in the transcript, but the
    offline extractor's surface patterns cannot see it.  This creates realistic
    evidence scarcity without making the label unfair.
    """
    flags: dict[str, bool] = {}

    resolution_cues = [c for t in turns for c in t.cues if c.kind == "resolution"]
    if resolution_cues and rng.random() < 0.22:
        cue = resolution_cues[0]
        replacement = rng.choice(IMPLICIT_RESOLUTION[spec.resolution])
        for turn in turns:
            if cue.text in turn.text:
                turn.text = turn.text.replace(cue.text, replacement)
                cue.text = replacement
                flags["implicit_resolution"] = True
                break

    empathy_cues = [c for t in turns for c in t.cues if c.kind == "empathy"]
    if empathy_cues and rng.random() < 0.18:
        cue = empathy_cues[0]
        replacement = rng.choice(IMPLICIT_EMPATHY)
        for turn in turns:
            if cue.text in turn.text:
                turn.text = turn.text.replace(cue.text, replacement)
                cue.text = replacement
                flags["implicit_empathy"] = True
                break

    sentiment_cues = [c for t in turns for c in t.cues if c.kind == "sentiment"]
    if sentiment_cues and rng.random() < 0.12:
        cue = sentiment_cues[0]
        replacement = rng.choice(IMPLICIT_SENTIMENT[spec.sentiment])
        for turn in turns:
            if cue.text in turn.text:
                turn.text = turn.text.replace(cue.text, replacement)
                cue.text = replacement
                flags["implicit_sentiment"] = True
                break

    effort_cues = [c for t in turns for c in t.cues if c.kind == "effort"]
    if effort_cues and rng.random() < 0.10:
        cue = effort_cues[0]
        replacement = rng.choice(IMPLICIT_EFFORT)
        for turn in turns:
            if cue.text in turn.text:
                turn.text = turn.text.replace(cue.text, replacement)
                cue.text = replacement
                flags["implicit_effort"] = True
                break

    return flags


def _gold_sentiment(turns: list[Turn]) -> int:
    values = [c.polarity for t in turns for c in t.cues if c.kind == "sentiment"]
    return max(-2, min(2, sum(values)))


def _new_call(
    call_id: str,
    pair_id: str,
    split: Split,
    condition: Condition,
    spec: CallSpec,
    turns: list[Turn],
    meta: dict[str, str | float | int | bool] | None = None,
) -> CallLog:
    return CallLog(
        call_id=call_id,
        pair_id=pair_id,
        condition=condition,
        split=split,
        persona=spec.persona,
        issue_type=spec.issue,
        turns=turns,
        latent_z=spec.latent_z,
        latent_satisfaction=spec.latent,
        survey_response=spec.survey,
        annotator_a=spec.annotator_a,
        annotator_b=spec.annotator_b,
        gold_resolution=spec.resolution,
        gold_effort=spec.effort,
        gold_sentiment=_gold_sentiment(turns),
        gold_empathy=spec.empathy,
        meta=meta or {},
    )


def _replace_cue_text(call: CallLog, old: str, new: str) -> None:
    """Replace a cue surface form in both the turn text and the cue record."""
    for turn in call.turns:
        if old in turn.text:
            turn.text = turn.text.replace(old, new)
            for cue in turn.cues:
                if cue.text == old:
                    cue.text = new


def _mangle_text(text: str, rng: random.Random) -> tuple[str, bool]:
    changed = False
    for word, mangled in ASR_MANGLES.items():
        if word in text and rng.random() < 0.65:
            text = text.replace(word, mangled, 1)
            changed = True
    return text, changed


def apply_disfluency(call: CallLog, rng: random.Random) -> CallLog:
    """Label-preserving shift: fillers, repetitions, and light ASR noise.

    The transformation deliberately lands *inside* cue phrases so that surface-form
    extraction breaks while the underlying facts stay the same.
    """
    shifted = copy.deepcopy(call)
    n_asr = 0
    n_disrupted = 0
    keywords = (
        "processed",
        "escalated",
        "submitted",
        "opened",
        "transfer",
        "hold",
        "call",
        "repeat",
        "understand",
        "appreciate",
        "sorry",
        "relief",
        "helps",
        "happy",
        "ridiculous",
        "annoyed",
        "fine",
    )
    for turn in shifted.turns:
        for cue in turn.cues:
            new_text = cue.text
            if rng.random() < 0.40:
                candidates = [w for w in keywords if w in new_text.lower()]
                if candidates:
                    word = rng.choice(candidates)
                    new_text = re.sub(
                        rf"\b{word}\b", f"{word}, um,", new_text, count=1, flags=re.IGNORECASE
                    )
                    n_disrupted += 1
            if rng.random() < 0.28:
                mangled, changed = _mangle_text(new_text, rng)
                if changed:
                    new_text = mangled
                    n_asr += 1
            if new_text != cue.text:
                turn.text = turn.text.replace(cue.text, new_text)
                cue.text = new_text
        if turn.speaker == "customer":
            if rng.random() < 0.65:
                turn.text = rng.choice(("Um, ", "Uh, ", "You know, ", "Like, ")) + turn.text
            if rng.random() < 0.25 and not turn.cues:
                words = turn.text.split()
                if len(words) > 3:
                    i = rng.randrange(1, len(words) - 1)
                    words.insert(i, words[i])
                    turn.text = " ".join(words)
            if rng.random() < 0.35 and not turn.cues:
                turn.text = turn.text.replace(". ", ", um, ", 1)
    shifted.condition = "disfluency"
    shifted.meta = {
        **shifted.meta,
        "n_asr": n_asr,
        "n_disrupted": n_disrupted,
        "shift": "disfluency",
    }
    return shifted


def apply_paraphrase(call: CallLog, rng: random.Random) -> CallLog:
    """Lexical shift: replace cue sentences with meaning-preserving paraphrases."""
    shifted = copy.deepcopy(call)
    n_paraphrased = 0
    for turn in shifted.turns:
        for cue in turn.cues:
            paraphrase = PARAPHRASES.get(cue.text)
            if paraphrase is not None and rng.random() < 0.90:
                turn.text = turn.text.replace(cue.text, paraphrase)
                cue.text = paraphrase
                n_paraphrased += 1
    shifted.condition = "paraphrase"
    shifted.meta = {**shifted.meta, "n_paraphrased": n_paraphrased, "shift": "paraphrase"}
    return shifted


def apply_emotional_arc(call: CallLog, rng: random.Random) -> CallLog:
    """Concept shift: late frustration after resolution or suppressed dissatisfaction.

    Latent satisfaction and the simulated survey are held fixed.  Only the expressed
    sentiment changes, which is exactly the kind of spurious cue drift that breaks a
    calibration layer fitted on the source distribution.
    """
    shifted = copy.deepcopy(call)
    sentiment_cues = [c for t in shifted.turns for c in t.cues if c.kind == "sentiment"]
    if not sentiment_cues:
        return shifted
    cue = sentiment_cues[0]
    flip = False
    new_text = cue.text
    new_polarity = cue.polarity
    if call.gold_resolution == "resolved" and rng.random() < 0.65:
        new_text = rng.choice(LATE_FRUSTRATION)
        new_polarity = rng.choice((-2, -1))
        flip = True
    elif call.gold_resolution == "unresolved" and rng.random() < 0.65:
        new_text = rng.choice(SUPPRESSED)
        new_polarity = 1
        flip = True
    elif call.gold_resolution == "partial" and rng.random() < 0.50:
        new_text = rng.choice(LATE_FRUSTRATION)
        new_polarity = rng.choice((-2, -1))
        flip = True
    if flip:
        for turn in shifted.turns:
            if cue.text in turn.text:
                turn.text = turn.text.replace(cue.text, new_text)
                cue.text = new_text
                cue.polarity = new_polarity
                break
    shifted.condition = "emotional_arc"
    shifted.gold_sentiment = _gold_sentiment(shifted.turns)
    shifted.meta = {
        **shifted.meta,
        "arc_flip": flip,
        "arc_polarity": new_polarity,
        "shift": "emotional_arc",
    }
    return shifted


def apply_combined(call: CallLog, rng: random.Random) -> CallLog:
    shifted = apply_paraphrase(call, rng)
    shifted = apply_emotional_arc(shifted, rng)
    shifted = apply_disfluency(shifted, rng)
    shifted.condition = "combined"
    shifted.meta = {**shifted.meta, "shift": "combined"}
    return shifted


def generate_source(
    n_train: int = N_TRAIN,
    n_dev: int = N_DEV,
    n_holdout: int = N_HOLDOUT,
    seed: int = SEED,
) -> list[CallLog]:
    calls: list[CallLog] = []
    total = n_train + n_dev + n_holdout
    for i in range(total):
        split = "train" if i < n_train else ("dev" if i < n_train + n_dev else "holdout")
        rng = random.Random(seed + 10_000 + i)
        spec = sample_spec(rng, PERSONA_WEIGHTS_SOURCE)
        turns = _render_source(spec, rng)
        dropout_flags = _apply_cue_dropout(turns, spec, rng)
        call_id = f"c{i:04d}"
        calls.append(
            _new_call(
                call_id=call_id,
                pair_id=call_id,
                split=split,
                condition="source",
                spec=spec,
                turns=turns,
                meta={
                    "shift": "none",
                    "agent_style": spec.agent_style,
                    **dropout_flags,
                },
            )
        )
    return calls


def generate_persona_mix(n: int = N_SHIFT, seed: int = SEED, start: int = 9000) -> list[CallLog]:
    calls: list[CallLog] = []
    for i in range(n):
        rng = random.Random(seed + 20_000 + i)
        spec = sample_spec(rng, PERSONA_WEIGHTS_SHIFT)
        turns = _render_source(spec, rng)
        dropout_flags = _apply_cue_dropout(turns, spec, rng)
        call_id = f"pm{i:04d}"
        calls.append(
            _new_call(
                call_id=call_id,
                pair_id=call_id,
                split="holdout",
                condition="persona_mix",
                spec=spec,
                turns=turns,
                meta={
                    "shift": "persona_mix",
                    "agent_style": spec.agent_style,
                    **dropout_flags,
                },
            )
        )
    return calls


def apply_shift(call: CallLog, condition: str, seed: int = SEED) -> CallLog:
    rng = random.Random(seed + 30_000 + int(call.call_id[1:]) * 17)
    if condition == "disfluency":
        return apply_disfluency(call, rng)
    if condition == "emotional_arc":
        return apply_emotional_arc(call, rng)
    if condition == "paraphrase":
        return apply_paraphrase(call, rng)
    if condition == "combined":
        return apply_combined(call, rng)
    raise ValueError(f"unknown shift condition: {condition}")


def build_datasets(
    data_dir: Path = DATA_DIR, seed: int = SEED, force: bool = False
) -> dict[str, list[CallLog]]:
    """Generate and cache source + paired shifted datasets as JSONL."""
    data_dir.mkdir(parents=True, exist_ok=True)
    source_path = data_dir / "source.jsonl"
    required = {"source", "disfluency", "emotional_arc", "paraphrase", "combined", "persona_mix"}
    if source_path.exists() and not force:
        datasets = {p.stem: _load_jsonl(p) for p in sorted(data_dir.glob("*.jsonl"))}
        if required <= set(datasets):
            return datasets

    source = generate_source(seed=seed)
    holdout = [c for c in source if c.split == "holdout"]
    datasets: dict[str, list[CallLog]] = {"source": source}
    for condition in ("disfluency", "emotional_arc", "paraphrase", "combined"):
        datasets[condition] = [apply_shift(c, condition, seed) for c in holdout]
    datasets["persona_mix"] = generate_persona_mix(seed=seed)

    for name, calls in datasets.items():
        _save_jsonl(data_dir / f"{name}.jsonl", calls)
    return datasets


def _save_jsonl(path: Path, calls: Iterable[CallLog]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for call in calls:
            handle.write(json.dumps(call.to_dict(), ensure_ascii=False) + "\n")


def _load_jsonl(path: Path) -> list[CallLog]:
    with path.open(encoding="utf-8") as handle:
        return [CallLog.from_dict(json.loads(line)) for line in handle if line.strip()]


def load_datasets(data_dir: Path = DATA_DIR) -> dict[str, list[CallLog]]:
    required = {"source", "disfluency", "emotional_arc", "paraphrase", "combined", "persona_mix"}
    paths = sorted(data_dir.glob("*.jsonl"))
    if not paths or not required <= {p.stem for p in paths}:
        return build_datasets(data_dir)
    return {p.stem: _load_jsonl(p) for p in paths}


def summary_table(datasets: dict[str, list[CallLog]]) -> str:
    rows = []
    for name, calls in datasets.items():
        labels = [c.latent_satisfaction for c in calls]
        surveys = [c.survey_response for c in calls if c.survey_response is not None]
        rows.append(
            f"{name:14s} n={len(calls):3d} survey={len(surveys):3d} "
            f"latent_mean={sum(labels) / len(labels):.2f}"
        )
    return "\n".join(rows)


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()
