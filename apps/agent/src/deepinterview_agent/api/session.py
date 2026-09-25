"""Session routes: the session view + the worker's live-result write-back.

``GET /api/session/{id}`` returns a :class:`SessionView` (status, completed prep
steps, input-quality warnings, and the :class:`InterviewContext` once ready).
Unknown ids → 404.

``POST /api/session/from-context`` clones a cached ``InterviewContext`` into a
fresh ready session (0 LLM calls) for instant repeat practice.

``POST /api/session/{id}/new-questions`` keeps the saved candidate/job/company/
gap analysis and makes ONE planner LLM call for a fresh question set that avoids
the previous session's questions (same count + section order, so the live loop
contract holds). Personal-use fast path for repeat practice.

``POST /api/session/{id}/live-result`` is the INTERNAL write path the voice
worker uses at shutdown. The worker runs in a separate process, so with no
Supabase configured its own in-memory repo is invisible to the API — answers
would be lost and never scored. Writing through this endpoint lands the result
in the store the API actually reads. (Like every route, it is session-id
capability-guarded: ids are unguessable uuid4.)
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from ..core.deps import build_deps
from ..core.logging import get_logger
from ..prep.prompts import question_refresh_prompts
from ..shared_models import InterviewContext, PrepRequest, PrepResponse, QuestionPlan
from .auth import require_internal_secret
from .views import PROGRESS_STEPS, SessionView

router = APIRouter()

log = get_logger(__name__)


class LiveResultRequest(BaseModel):
    """Worker→API write-back: the post-interview context + verbatim transcript.

    API-internal (both ends are Python), so it lives here — NOT in
    ``shared_models`` — keeping the TS↔Pydantic parity registry untouched.
    """

    model_config = ConfigDict(extra="forbid")
    context: InterviewContext
    transcript: list[dict] = Field(default_factory=list)
    # Optional terminal hint ("no_answers" when the interview captured nothing).
    status: str | None = None


_ALLOWED_LIVE_STATUSES = {"no_answers", "error"}

# Once a session reaches one of these it is done; a late/replayed live-result
# write must not be able to overwrite a scored interview's history.
_TERMINAL_STATUSES = {"complete", "no_answers", "error", "rejected"}


class FromContextRequest(BaseModel):
    """Instant reuse: a previously cached InterviewContext + optional owner."""

    model_config = ConfigDict(extra="forbid")
    context: InterviewContext
    user_id: str | None = None


@router.post("/api/session/from-context", response_model=PrepResponse)
async def from_context(req: FromContextRequest) -> PrepResponse:
    """Clone a cached context into a fresh ``ready`` session (0 LLM calls).

    Validates the context, mints a new session id, resets the live cursor
    (``cursor=0``, ``answers=[]``, ``scorecard=None``), persists it, marks all
    prep steps complete, and returns the new id. The cached plan/candidate/job
    are reused verbatim.
    """
    if not req.context.plan.questions:
        raise HTTPException(status_code=422, detail="Cached context has no questions")
    deps = build_deps()
    prep_req = PrepRequest(
        cv_url="cached",
        jd_text=req.context.job.raw_text or req.context.job.title,
        company=req.context.job.company_name,
        language_mode=req.context.plan.language_mode,
        user_id=req.user_id,
    )
    session_id = await deps.repo.create_session(prep_req)
    fresh = req.context.model_copy(
        update={
            "session_id": session_id,
            "cursor": 0,
            "answers": [],
            "scorecard": None,
        }
    )
    await deps.repo.save_context(session_id, fresh)
    for step in PROGRESS_STEPS:
        try:
            await deps.repo.mark_progress(session_id, step)
        except Exception:
            pass
    await deps.repo.update_status(session_id, "ready")
    return PrepResponse(session_id=session_id)


@router.get("/api/session/{session_id}", response_model=SessionView)
async def get_session(session_id: str) -> SessionView:
    view = await build_deps().repo.get_session_view(session_id)
    if view is None:
        raise HTTPException(status_code=404, detail="Unknown session_id")
    return view


def _question_key(text: str) -> str:
    """Normalized question text for duplicate comparison."""
    return " ".join(text.lower().split())


def _overlap_fraction(previous: set[str], candidate: list[str]) -> float:
    """Fraction of candidate questions already seen (0.0 = all fresh)."""
    if not candidate:
        return 1.0
    seen = sum(1 for t in candidate if _question_key(t) in previous)
    return seen / len(candidate)


@router.post("/api/session/{session_id}/new-questions", response_model=PrepResponse)
async def new_questions(session_id: str) -> PrepResponse:
    """Fresh question set for a finished prep, reusing its saved analysis.

    Loads the session's ``InterviewContext`` (candidate/job/company/gap stay
    untouched) and makes a single planner LLM call — via ``deps.llm``, i.e. the
    analytic ``gemini_model`` (full Flash), never the live ``flash-lite`` tier —
    with the previous questions as an avoid-list. The new plan mirrors the old
    one's count + section order so the live loop and handoff personas keep
    working. One retry on heavy overlap; a total LLM failure surfaces as 502
    (no silent static fallback — that would return the SAME questions).
    """
    deps = build_deps()
    ctx = await deps.repo.load_context(session_id)
    if ctx is None:
        raise HTTPException(status_code=404, detail="Unknown session_id")
    old_qs = ctx.plan.questions
    if not old_qs:
        raise HTTPException(status_code=422, detail="Saved context has no questions")

    previous_texts = [
        (q.text.get("en") or next(iter(q.text.values()), "")) for q in old_qs
    ]
    previous_keys = {_question_key(t) for t in previous_texts if t.strip()}
    sections = [q.section for q in old_qs]
    thinking = getattr(deps.settings, "gemini_thinking_budget_high", 1024)

    plan: QuestionPlan | None = None
    for firm in (False, True):
        system, user = question_refresh_prompts(
            candidate=ctx.candidate,
            job=ctx.job,
            company=ctx.company,
            gap=ctx.gap,
            language_mode=ctx.plan.language_mode,
            previous_texts=previous_texts,
            sections=sections,
            firm=firm,
        )
        try:
            attempt = await deps.llm.complete_json(
                system=system,
                user=user,
                schema=QuestionPlan,
                thinking_budget=thinking,
            )
        except Exception as exc:  # noqa: BLE001 - surfacing as 502 below
            log.warning("new-questions: planner call failed (%s)", exc)
            attempt = None
        if attempt is None or not attempt.questions:
            continue
        attempt = attempt.model_copy(update={"language_mode": ctx.plan.language_mode})
        if len(attempt.questions) != len(old_qs) or [
            q.section for q in attempt.questions
        ] != sections:
            # Wrong shape for the live loop — retry once, then fail honestly.
            log.warning("new-questions: plan shape mismatch, retrying")
            continue
        fresh_texts = [
            (q.text.get("en") or next(iter(q.text.values()), "")) for q in attempt.questions
        ]
        if _overlap_fraction(previous_keys, fresh_texts) > 0.5 and not firm:
            log.info("new-questions: too much overlap, retrying with firmer prompt")
            continue
        plan = attempt
        break

    if plan is None:
        raise HTTPException(
            status_code=502,
            detail="Could not generate fresh questions — try again in a moment.",
        )

    prep_req = PrepRequest(
        cv_url="cached",
        jd_text=ctx.job.raw_text or ctx.job.title,
        company=ctx.job.company_name,
        language_mode=ctx.plan.language_mode,
    )
    new_id = await deps.repo.create_session(prep_req)
    fresh = ctx.model_copy(
        update={
            "session_id": new_id,
            "plan": plan,
            "cursor": 0,
            "answers": [],
            "scorecard": None,
        }
    )
    await deps.repo.save_context(new_id, fresh)
    for step in PROGRESS_STEPS:
        try:
            await deps.repo.mark_progress(new_id, step)
        except Exception:
            pass
    await deps.repo.update_status(new_id, "ready")
    return PrepResponse(session_id=new_id)


@router.post(
    "/api/session/{session_id}/live-result",
    dependencies=[Depends(require_internal_secret)],
)
async def post_live_result(session_id: str, req: LiveResultRequest) -> dict:
    deps = build_deps()
    view = await deps.repo.get_session_view(session_id)
    if view is None:
        raise HTTPException(status_code=404, detail="Unknown session_id")
    # Refuse to rewrite a session that has already reached a terminal state:
    # the transcript/answers are final once scored, and a replayed or forged
    # write must not be able to mutate them.
    if view.status in _TERMINAL_STATUSES:
        raise HTTPException(
            status_code=409, detail=f"Session already {view.status}"
        )
    if req.transcript:
        await deps.repo.save_transcript(session_id, req.transcript)
    await deps.repo.save_context(session_id, req.context)
    if req.status in _ALLOWED_LIVE_STATUSES:
        await deps.repo.update_status(session_id, req.status)
    return {"ok": True}
