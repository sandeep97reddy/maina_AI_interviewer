"""Session routes: the session view + the worker's live-result write-back.

``GET /api/session/{id}`` returns a :class:`SessionView` (status, completed prep
steps, input-quality warnings, and the :class:`InterviewContext` once ready).
Unknown ids → 404.

``POST /api/session/from-context`` clones a cached ``InterviewContext`` into a
fresh ready session (0 LLM calls) for instant repeat practice.

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
from ..shared_models import InterviewContext, PrepRequest, PrepResponse
from .auth import require_internal_secret
from .views import PROGRESS_STEPS, SessionView

router = APIRouter()


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
