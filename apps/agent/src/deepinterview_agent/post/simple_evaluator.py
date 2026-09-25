"""Unified 2-call holistic scoring (replaces the fragmented N-call pipeline).

Old pipeline: N per-question evaluate calls + 1 language call + 1 narrative
call + N model-answer calls (~16 calls for 7 questions), each seeing one
answer in isolation.

New pipeline (2 calls, both holistic):
  Call 1 (scores+language): full transcript + all rubrics -> per-question
    competency scores with quoted evidence + spoken-language report.
  Call 2 (narrative+models): scores + transcripts -> strengths, weaknesses,
    next steps, summary paragraph, and one model answer per ANSWERED question.

Numeric fields (level, overall, weak, coverage) are derived deterministically
in Python — never trusted to the model — preserving the Prep Coach loop
contract (competency maps to plan target_competency).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict

from ..shared_models import (
    CompetencyScore,
    LanguageReport,
    ModelAnswer,
    ScoreCard,
)
from .evaluator import level_for_score
from .report import _coverage_pct, _overall_score, _weak_competencies

if TYPE_CHECKING:
    from ..core.deps import Deps
    from ..shared_models import InterviewContext

# Per-answer cap so a very long interview stays prompt-sized. 7 answers x 2000
# chars ≈ 14k chars ≈ 3.5k tokens — comfortable for Flash + local models.
_TRANSCRIPT_CAP = 2000


class _RawCompetency(BaseModel):
    model_config = ConfigDict(extra="forbid")
    competency: str
    score: float
    evidence: str


class _Pass1Response(BaseModel):
    """LLM output for call 1: raw scores + language report."""

    model_config = ConfigDict(extra="forbid")
    competency_scores: list[_RawCompetency]
    language_report: LanguageReport


class _Pass2Response(BaseModel):
    """LLM output for call 2: narrative + model answers."""

    model_config = ConfigDict(extra="forbid")
    strengths: list[str]
    weaknesses: list[str]
    next_steps: list[str]
    summary: str
    model_answers: list[ModelAnswer]


def _clamp(value: float) -> float:
    return max(0.0, min(5.0, float(value)))


def _thinking(deps: Deps, tier: str) -> int | None:
    try:
        settings = deps.settings
    except Exception:
        return None
    if tier == "mid":
        return getattr(settings, "gemini_thinking_budget_mid", 512)
    return getattr(settings, "gemini_thinking_budget_low", 0)


def _answered(ctx: InterviewContext) -> list[tuple[str, str, str, str]]:
    """(question_id, target_competency, rubric_block, transcript) per ANSWERED q.

    Order follows the plan. Unanswered questions are excluded — a competency
    we never probed must not read as weak.
    """
    by_id = {a.question_id: a for a in ctx.answers}
    out: list[tuple[str, str, str, str]] = []
    for q in ctx.plan.questions:
        a = by_id.get(q.id)
        text = (a.transcript or "").strip() if a else ""
        if not text:
            continue
        rubric_lines = "\n".join(
            f"- {r.criterion} (w {r.weight:.2f}): {r.description}" for r in q.rubric
        ) or "- (no rubric)"
        q_en = q.text.get("en") or next(iter(q.text.values()), "")
        block = (
            f"QID: {q.id} | section={q.section} difficulty={q.difficulty}\n"
            f"TARGET COMPETENCY: {q.target_competency}\n"
            f"QUESTION: {q_en}\nRUBRIC:\n{rubric_lines}"
        )
        out.append((q.id, q.target_competency, block, text[:_TRANSCRIPT_CAP]))
    return out


def _pass1_prompts(ctx: InterviewContext) -> tuple[str, str]:
    answered = _answered(ctx)
    parts: list[str] = []
    for qid, comp, block, transcript in answered:
        parts.append(f"{block}\nCANDIDATE ANSWER:\n{transcript}")
    joined = "\n\n---\n\n".join(parts) if parts else "(no answers)"
    system = (
        "You are a rigorous, fair interview assessor. Score EACH answered "
        "question below against its rubric on a 0-5 scale "
        "(0=no relevant content, 3=solid, 5=exceptional). Weigh rubric criteria "
        "by weight. For each, return competency (copy TARGET COMPETENCY exactly), "
        "score, and evidence citing a short concrete quote from the answer. "
        "Return the entries in the SAME ORDER as given. Also assess spoken "
        "delivery across all answers: fluency_score and clarity_score 0-5, "
        "approximate filler_word_count, code-switching and pronunciation notes, "
        "and an encouraging summary. Respond ONLY with the requested schema."
    )
    user = (
        f"ROLE: {ctx.job.title} ({ctx.job.seniority}) at {ctx.job.company_name}\n"
        f"CANDIDATE: {ctx.candidate.headline}\n\n"
        f"ANSWERED QUESTIONS ({len(answered)}):\n\n{joined}"
    )
    return system, user


def _pass2_prompts(
    ctx: InterviewContext,
    comp_scores: list[CompetencyScore],
    lang_report: LanguageReport,
    overall: float,
) -> tuple[str, str]:
    answered = _answered(ctx)
    comp_lines = "\n".join(
        f"- {cs.competency}: {cs.score:.1f}/5 ({cs.level}) — {cs.evidence}"
        for cs in comp_scores
    ) or "- (none)"
    # Give the writer the transcripts so model answers can improve on them.
    answer_blocks = "\n\n".join(
        f"QID {qid} ({comp}):\n{transcript}" for qid, comp, _, transcript in answered
    )
    system = (
        "You are an expert interview coach writing the candidate's feedback "
        "report. Given per-competency scores, produce concrete strengths, "
        "weaknesses tied to evidence, 3 prioritized study drills as next_steps, "
        "and a warm honest summary paragraph. Also write ONE concise model "
        "answer (under ~180 words, STAR for behavioral) per answered question "
        "below, tailored to the candidate's background — return them with the "
        "matching question_id. Respond ONLY with the requested schema."
    )
    user = (
        f"ROLE: {ctx.job.title} ({ctx.job.seniority}) at {ctx.job.company_name}\n"
        f"CANDIDATE: {ctx.candidate.headline} "
        f"({ctx.candidate.years_experience}y, {ctx.candidate.seniority})\n"
        f"SKILLS: {', '.join(ctx.candidate.skills)}\n"
        f"OVERALL: {overall:.2f}/5\n\n"
        f"PER-COMPETENCY SCORES:\n{comp_lines}\n\n"
        f"LANGUAGE: fluency {lang_report.fluency_score:.1f}/5, "
        f"clarity {lang_report.clarity_score:.1f}/5, "
        f"fillers ~{lang_report.filler_word_count}. {lang_report.summary}\n\n"
        f"CANDIDATE ANSWERS:\n{answer_blocks}"
    )
    return system, user


def _merge_by_competency(scores: list[CompetencyScore]) -> list[CompetencyScore]:
    """De-dup competencies by averaging; preserve first-seen order."""
    order: list[str] = []
    buckets: dict[str, list[CompetencyScore]] = {}
    for cs in scores:
        if cs.competency not in buckets:
            buckets[cs.competency] = []
            order.append(cs.competency)
        buckets[cs.competency].append(cs)
    merged: list[CompetencyScore] = []
    for comp in order:
        group = buckets[comp]
        if len(group) == 1:
            merged.append(group[0])
            continue
        avg = _clamp(sum(s.score for s in group) / len(group))
        evidence = " ".join(s.evidence for s in group if s.evidence).strip()
        merged.append(
            CompetencyScore(
                competency=comp,
                score=avg,
                evidence=evidence or "Averaged across multiple questions.",
                level=level_for_score(avg),
            )
        )
    return merged


async def score_pass1(
    ctx: InterviewContext, deps: Deps
) -> tuple[list[CompetencyScore], LanguageReport]:
    """Call 1: holistic competency scores + language report.

    Forces ``competency`` to each question's ``target_competency`` (loop
    contract) and derives ``level`` from the number. Length mismatches
    (e.g. MockLLM returning 1 entry) are stretched by cycling templates in
    plan order, so every answered question is scored.
    """
    answered = _answered(ctx)
    if not answered:
        raise RuntimeError("simple_evaluator: no answered questions to score")
    system, user = _pass1_prompts(ctx)
    raw = await deps.llm.complete_json(
        system=system, user=user, schema=_Pass1Response, thinking_budget=_thinking(deps, "mid")
    )
    templates = list(raw.competency_scores) or [
        _RawCompetency(competency="General", score=2.5, evidence="No evidence returned.")
    ]
    per_question: list[CompetencyScore] = []
    for i, (qid, target, _block, _t) in enumerate(answered):
        t = templates[i % len(templates)]
        score = _clamp(t.score)
        evidence = (t.evidence or "").strip() or "Scored against the question rubric."
        per_question.append(
            CompetencyScore(
                competency=target,
                score=score,
                evidence=evidence,
                level=level_for_score(score),
            )
        )
    lang = raw.language_report.model_copy(
        update={
            "fluency_score": _clamp(raw.language_report.fluency_score),
            "clarity_score": _clamp(raw.language_report.clarity_score),
            "filler_word_count": max(0, int(raw.language_report.filler_word_count)),
        }
    )
    return _merge_by_competency(per_question), lang


async def build_scorecard(
    ctx: InterviewContext,
    comp_scores: list[CompetencyScore],
    lang_report: LanguageReport,
    deps: Deps,
) -> ScoreCard:
    """Call 2: narrative + model answers, assembled deterministically."""
    overall = _overall_score(comp_scores)
    weak = _weak_competencies(comp_scores)
    system, user = _pass2_prompts(ctx, comp_scores, lang_report, overall)
    narrative = await deps.llm.complete_json(
        system=system, user=user, schema=_Pass2Response, thinking_budget=_thinking(deps, "low")
    )
    # Realign model answers to answered question ids in plan order. The model
    # may return them out of order or with wrong ids — align by id when
    # possible, else cycle templates by position (MockLLM returns a single
    # element; cycling keeps full coverage instead of dropping questions).
    answered_ids = [qid for qid, _, _, _ in _answered(ctx)]
    by_id = {m.question_id: m for m in narrative.model_answers}
    aligned: list[ModelAnswer] = []
    templates = list(narrative.model_answers)
    for i, qid in enumerate(answered_ids):
        hit = by_id.get(qid)
        if hit is not None:
            aligned.append(hit)
        elif templates:
            fallback = templates[i % len(templates)]
            aligned.append(ModelAnswer(question_id=qid, answer=fallback.answer))
        # Empty list: skip (no hallucinated filler).
    return ScoreCard(
        overall_score=overall,
        competency_scores=comp_scores,
        strengths=list(narrative.strengths),
        weaknesses=list(narrative.weaknesses),
        weak_competencies=weak,
        model_answers=aligned,
        next_steps=list(narrative.next_steps),
        language_report=lang_report,
        summary=narrative.summary,
        coverage_pct=_coverage_pct(ctx),
    )
