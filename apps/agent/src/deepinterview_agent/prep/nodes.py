"""Async node functions for the WP-6 prep graph.

Every node takes ``(state, deps)`` and returns a partial ``PrepState`` dict with
only the key(s) it computes; LangGraph merges those into the running state. Deps
are injected into the graph via :func:`functools.partial` (see ``graph.py``), so
each compiled node presents the ``(state)`` signature LangGraph expects.

``fetch_cv`` is deliberately best-effort and offline-tolerant: it delegates to
:func:`deepinterview_agent.prep.cv_extract.extract_cv_text`, which parses an
uploaded PDF/DOCX (``data:`` URL or fetched ``http(s)`` URL) into real text and,
on any failure, falls back to the URL string itself as the document text. This
keeps the whole pipeline — and the existing ``POST /api/prep`` test that points
at an unreachable example.com — green without network access.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

from ..core.adapters.mock import build_mock
from ..core.logging import get_logger
from ..core.tracing import traced
from ..shared_models import (
    CandidateProfile,
    Citation,
    CompanyIntel,
    GapAnalysis,
    JobSpec,
    LanguageMode,
    PlannedQuestion,
    QuestionPlan,
    RubricItem,
    Seniority,
)
from .cv_extract import extract_cv_text
from .prompts import (
    company_research_prompts,
    cv_analysis_prompts,
    gap_matching_prompts,
    jd_analysis_prompts,
    language_name,
    question_planner_prompts,
)
from .state import PrepState

if TYPE_CHECKING:
    from ..core.deps import Deps

log = get_logger(__name__)


async def _mark(state: PrepState, deps: Deps, step: str) -> None:
    """Record that ``step`` finished, if running against a known session.

    Best-effort: a missing/closed session must never crash prep — the progress
    signal is for the UI only.
    """
    session_id = state.get("session_id")
    if not session_id:
        return
    try:
        await deps.repo.mark_progress(session_id, step)
    except Exception as exc:  # noqa: BLE001 - progress is advisory only
        log.warning("mark_progress(%s) failed (%s)", step, exc)


async def _warn(state: PrepState, deps: Deps, warnings: list[str]) -> None:
    """Attach input-quality/fallback warnings to the session (best-effort)."""
    session_id = state.get("session_id")
    if not session_id or not warnings:
        return
    try:
        await deps.repo.add_warnings(session_id, warnings)
    except Exception as exc:  # noqa: BLE001 - warnings are advisory only
        log.warning("add_warnings failed (%s)", exc)


@traced("prep.fetch_cv")
async def fetch_cv(state: PrepState, deps: Deps) -> PrepState:
    """Best-effort parse of the CV document into text; fall back to the URL string.

    Delegates to :func:`extract_cv_text`, which converts an uploaded PDF/DOCX
    (``data:`` URL or fetched ``http(s)`` URL) into plain text via markitdown
    (Gemini fallback for scanned/image PDFs). Any returned warnings are attached
    to the session via ``_warn``.

    Idempotent: if ``cv_text`` was already resolved (the caller pre-fetched it so
    it could validate inputs), this is a no-op — we never parse the CV twice. We
    check key *presence*, not truthiness: an unreadable document resolves to ``""``
    and must NOT trigger a re-parse (which would re-warn and re-bill Gemini).
    """
    if "cv_text" in state:
        return {}
    req = state["req"]
    try:
        cv_text, warnings = await extract_cv_text(req.cv_url, deps)
    except Exception as exc:  # noqa: BLE001 - best-effort: any failure -> raw fallback
        log.warning("fetch_cv: extraction failed, using cv_url as text (%s)", exc)
        return {"cv_text": req.cv_url}
    if warnings:
        await _warn(state, deps, warnings)
    return {"cv_text": cv_text}


def _detect_seniority(text: str) -> Seniority:
    low = (text or "").lower()
    if any(w in low for w in ("staff", "principal", "director", "architect")):
        return "staff"
    if any(w in low for w in ("senior", "sr.", "lead")):
        return "senior"
    if any(w in low for w in ("junior", "entry", "jr.", "associate")):
        return "junior"
    if any(w in low for w in ("intern", "internship", "trainee", "student")):
        return "intern"
    return "mid"


def _fallback_candidate(cv_text: str) -> CandidateProfile:
    """Extract a clean, non-mock CandidateProfile from CV text when LLM fails."""
    lines = [line.strip() for line in (cv_text or "").splitlines() if line.strip()]
    name = "Candidate"
    for line in lines[:5]:
        clean = re.sub(r"[^a-zA-Z\s\.\-']", "", line).strip()
        if (
            2 <= len(clean.split()) <= 4
            and len(clean) <= 40
            and not any(
                k in clean.lower()
                for k in ("resume", "curriculum", "vitae", "profile", "summary", "contact", "experience")
            )
        ):
            name = clean
            break

    seniority = _detect_seniority(cv_text)
    headline = f"{seniority.title()} Professional"
    for line in lines[1:6]:
        if 5 <= len(line) <= 60 and any(
            k in line.lower()
            for k in ("engineer", "developer", "manager", "designer", "lead", "specialist", "scientist", "architect")
        ):
            headline = line
            break

    summary = (
        f"{name} is an experienced professional with a background in software and technology."
    )
    if lines:
        summary = f"{name}: " + " ".join(lines[:4])[:200]

    return CandidateProfile(
        name=name,
        headline=headline,
        summary_120w=summary,
        years_experience=5 if seniority in ("senior", "staff") else 3,
        seniority=seniority,
        skills=["Problem Solving", "Communication", "Technical Architecture"],
        projects=[],
        achievements=[],
        education=[],
        spoken_languages=["English"],
        links=[],
    )


def _fallback_job(req: Any) -> JobSpec:
    """Extract a clean, non-mock JobSpec from request and JD text when LLM fails."""
    company_name = req.company.strip() if getattr(req, "company", None) and req.company.strip() else "Target Company"
    jd_lines = [line.strip() for line in (getattr(req, "jd_text", "") or "").splitlines() if line.strip()]

    title = "Software Engineer"
    for line in jd_lines[:5]:
        if 4 <= len(line) <= 60 and any(
            k in line.lower()
            for k in ("engineer", "developer", "architect", "lead", "manager", "analyst", "specialist")
        ):
            title = line
            break

    seniority = _detect_seniority(getattr(req, "jd_text", ""))

    return JobSpec(
        title=title,
        company_name=company_name,
        location="Remote",
        seniority=seniority,
        must_have=["Technical proficiency", "System design", "Communication"],
        nice_to_have=["Domain expertise"],
        responsibilities=["Develop and deliver high quality solutions"],
        tech_stack=["Modern stack"],
        raw_text=getattr(req, "jd_text", "") or f"{title} at {company_name}",
    )


def _fallback_plan(
    candidate: CandidateProfile,
    job: JobSpec,
    company: CompanyIntel,
    language_mode: LanguageMode,
) -> QuestionPlan:
    """Generate meaningful, tailored interview questions when planner LLM fails."""
    company_name = (
        company.name
        if company and company.name and company.name.lower() != "mock"
        else (job.company_name if job and job.company_name.lower() != "mock" else "our team")
    )
    cand_name = candidate.name if candidate and candidate.name.lower() != "mock" else "Candidate"
    job_title = job.title if job and job.title.lower() != "mock" else "Software Engineer"
    primary = language_mode.primary

    questions = [
        PlannedQuestion(
            id="q_intro",
            section="intro",
            text={
                "en": f"Welcome, {cand_name}. Could you briefly introduce yourself and tell me what excites you about the {job_title} role at {company_name}?",
                primary: f"Welcome, {cand_name}. Could you briefly introduce yourself and tell me what excites you about the {job_title} role at {company_name}?",
            },
            difficulty=2,
            rubric=[
                RubricItem(criterion="Clarity and communication", weight=0.5, description="Clear, structured overview"),
                RubricItem(criterion="Role alignment", weight=0.5, description="Articulates interest in role and company"),
            ],
            followups=[
                "Which recent project best highlights your strengths for this role?",
            ],
            target_competency="Communication",
        ),
        PlannedQuestion(
            id="q_technical",
            section="technical",
            text={
                "en": f"In your work relevant to this {job_title} position, could you walk me through an architecture or technical challenge you tackled and how you decided on the final solution?",
                primary: f"In your work relevant to this {job_title} position, could you walk me through an architecture or technical challenge you tackled and how you decided on the final solution?",
            },
            difficulty=3,
            rubric=[
                RubricItem(criterion="Technical depth", weight=0.6, description="Explains technical tradeoffs and depth"),
                RubricItem(criterion="Problem solving", weight=0.4, description="Demonstrates structured debugging or design"),
            ],
            followups=[
                "What trade-offs or constraints did you face, and what would you do differently in retrospect?",
            ],
            target_competency="Technical Execution",
        ),
        PlannedQuestion(
            id="q_behavioral",
            section="behavioral",
            text={
                "en": f"At {company_name}, cross-functional alignment is key. Tell me about a time you had a technical disagreement with a colleague or stakeholder and how you reached a resolution.",
                primary: f"At {company_name}, cross-functional alignment is key. Tell me about a time you had a technical disagreement with a colleague or stakeholder and how you reached a resolution.",
            },
            difficulty=3,
            rubric=[
                RubricItem(criterion="Collaboration", weight=0.5, description="Shows empathy and constructive consensus"),
                RubricItem(criterion="Ownership", weight=0.5, description="Demonstrates focus on team outcome"),
            ],
            followups=[
                "How did the outcome impact the team's delivery and relationship?",
            ],
            target_competency="Collaboration",
        ),
        PlannedQuestion(
            id="q_wrap",
            section="wrap",
            text={
                "en": f"Thank you, {cand_name}. To wrap up our conversation today, what questions do you have for me about {company_name} or the {job_title} team?",
                primary: f"Thank you, {cand_name}. To wrap up our conversation today, what questions do you have for me about {company_name} or the {job_title} team?",
            },
            difficulty=1,
            rubric=[
                RubricItem(criterion="Engagement", weight=1.0, description="Thoughtful questions about company and role"),
            ],
            followups=[],
            target_competency="Culture Fit",
        ),
    ]

    return QuestionPlan(
        sections_order=["intro", "technical", "behavioral", "wrap"],
        questions=questions,
        time_budget_min=30,
        language_mode=language_mode,
    )


@traced("prep.cv_analysis")
async def cv_analysis(state: PrepState, deps: Deps) -> PrepState:
    """Extract a ``CandidateProfile`` from the fetched CV text."""
    system, user = cv_analysis_prompts(state["cv_text"])
    try:
        candidate = await deps.llm.complete_json(
            system=system, user=user, schema=CandidateProfile
        )
    except Exception as exc:  # noqa: BLE001 - resilient: degrade, don't crash prep
        log.warning("cv_analysis failed, using fallback profile (%s)", exc)
        candidate = _fallback_candidate(state.get("cv_text", ""))
        await _warn(state, deps, ["Could not analyze the CV with LLM; extracted profile directly."])
    await _mark(state, deps, "cv_analysis")
    return {"candidate": candidate}


@traced("prep.jd_analysis")
async def jd_analysis(state: PrepState, deps: Deps) -> PrepState:
    """Extract a ``JobSpec`` from the job description text."""
    req = state["req"]
    system, user = jd_analysis_prompts(req.jd_text, req.company)
    try:
        job = await deps.llm.complete_json(system=system, user=user, schema=JobSpec)
    except Exception as exc:  # noqa: BLE001 - resilient: degrade, don't crash prep
        log.warning("jd_analysis failed, using fallback job spec (%s)", exc)
        job = _fallback_job(req)
        await _warn(
            state, deps, ["Could not analyze the job description with LLM; extracted job spec directly."]
        )
    await _mark(state, deps, "jd_analysis")
    return {"job": job}


def _empty_company_intel(name: str) -> CompanyIntel:
    """A valid, empty ``CompanyIntel`` for a junk/unknown company (no fabrication)."""
    return CompanyIntel(
        name=name or "Unknown",
        summary="",
        industry=None,
        tech_stack=[],
        values=[],
        interview_process=[],
        recent_news=[],
        citations=[],
    )


@traced("prep.company_research")
async def company_research(state: PrepState, deps: Deps) -> PrepState:
    """Search the web for company interview intel and synthesize ``CompanyIntel``.

    If the company name was flagged junk (``company_ok`` is False) this skips all
    search + LLM work and returns an empty-but-valid intel, so we never fabricate
    company knowledge from a meaningless name or waste calls on it.
    """
    req = state["req"]
    company = req.company

    if not state.get("company_ok", True):
        log.info("company_research: skipping for junk company %r", company)
        await _mark(state, deps, "company_research")
        return {"company": _empty_company_intel(company)}

    primary = req.language_mode.primary

    queries: list[tuple[str, str]] = [(f"{company} interview process", "en")]
    if primary != "en":
        localized = f"{company} interview process {language_name(primary)}"
        queries.append((localized, primary))

    results = []
    for query, lang in queries:
        try:
            results.extend(await deps.search.search(query, lang=lang, max_results=4))
        except Exception as exc:  # noqa: BLE001 - search is best-effort
            log.warning("company_research: search failed for %r (%s)", query, exc)

    snippets = "\n".join(f"- {r.title}: {r.snippet}" for r in results) or "(no results)"
    system, user = company_research_prompts(company, snippets)
    try:
        intel = await deps.llm.complete_json(
            system=system, user=user, schema=CompanyIntel
        )
    except Exception as exc:  # noqa: BLE001 - resilient: degrade, don't crash prep
        log.warning("company_research failed, using minimal intel (%s)", exc)
        intel = _empty_company_intel(company)
        await _warn(
            state, deps, ["Could not research the company; proceeding without intel."]
        )

    citations = [
        Citation(title=r.title, url=r.url, snippet=r.snippet) for r in results
    ]
    intel = intel.model_copy(update={"citations": citations})
    await _mark(state, deps, "company_research")
    return {"company": intel}


@traced("prep.gap_matching")
async def gap_matching(state: PrepState, deps: Deps) -> PrepState:
    """Compare candidate vs job into a ``GapAnalysis`` (join of cv + jd)."""
    system, user = gap_matching_prompts(state["candidate"], state["job"])
    try:
        gap = await deps.llm.complete_json(
            system=system, user=user, schema=GapAnalysis
        )
    except Exception as exc:  # noqa: BLE001 - resilient: degrade, don't crash prep
        log.warning("gap_matching failed, using minimal analysis (%s)", exc)
        gap = build_mock(GapAnalysis)
        await _warn(state, deps, ["Could not compute the gap analysis; used a minimal one."])
    await _mark(state, deps, "gap_matching")
    return {"gap": gap}


# Body sections a pack contributes to the planner prompt, and the total budget.
# Bounded so a growing library can never blow up prep prompt size (golden rule:
# keep prompts compact).
_HINT_SECTIONS = frozenset({"round structure", "question bank", "signals", "pitfalls"})
_HINT_CHAR_BUDGET = 1500


def _extract_hint_sections(body_md: str) -> str:
    """Pull the planner-relevant ``##`` sections out of a pack body, in order."""
    sections: list[tuple[str, list[str]]] = []
    current: list[str] | None = None
    for line in body_md.splitlines():
        if line.startswith("## "):
            title = line[3:].strip()
            current = [] if title.lower() in _HINT_SECTIONS else None
            if current is not None:
                sections.append((title, current))
        elif current is not None:
            current.append(line)
    parts = [f"{title}:\n" + "\n".join(lines).strip() for title, lines in sections if lines]
    return "\n".join(p for p in parts if not p.endswith(":\n"))


def _skill_library_hint(
    company: str, role: str, level: str, skills_dir: str | None = None
) -> str:
    """Playbook context for the planner (WP-10 retrieval): provenance header
    plus the pack's question bank / signals / pitfalls, capped at
    ``_HINT_CHAR_BUDGET`` characters.

    Best-effort and additive: any failure (missing/un-parseable library, import
    error) returns an empty string so prep never depends on the skill store.
    """
    try:
        from ..skilllib import effective_confidence, find_relevant

        skills = find_relevant(skills_dir, company=company, role=role, level=level)
        if not skills:
            return ""
        blocks: list[str] = []
        for skill in skills:
            fm = skill.frontmatter
            header = (
                f"### {fm.company} · {fm.role} · {fm.level} "
                f"[{fm.status}; confidence {effective_confidence(fm):.2f} "
                f"from {fm.source_runs} run(s); verified {fm.last_verified}]"
            )
            extract = _extract_hint_sections(skill.body_md)
            blocks.append(f"{header}\n{extract}" if extract else header)
        hint = "\n\n".join(blocks)
        if len(hint) > _HINT_CHAR_BUDGET:
            hint = hint[:_HINT_CHAR_BUDGET].rsplit("\n", 1)[0] + "\n[truncated]"
        return hint
    except Exception:  # noqa: BLE001 - retrieval is strictly best-effort
        return ""


@traced("prep.question_planner")
async def question_planner(state: PrepState, deps: Deps) -> PrepState:
    """Keystone: synthesize the full ``QuestionPlan`` from all upstream state."""
    req = state["req"]
    system, user = question_planner_prompts(
        candidate=state["candidate"],
        job=state["job"],
        company=state["company"],
        gap=state["gap"],
        language_mode=req.language_mode,
    )
    # WP-10: inject any matching distilled playbook as extra planner context.
    hint = _skill_library_hint(
        company=state["company"].name,
        role=state["job"].title,
        level=state["job"].seniority,
    )
    if hint:
        user = (
            f"{user}\n\n"
            "PLAYBOOK REFERENCE (from the interview skill library — adapt to THIS "
            "candidate and JD; prefer reusing its strongest questions over "
            "inventing near-duplicates, and never copy questions that don't fit "
            "the gap analysis):\n"
            f"{hint}"
        )
    try:
        plan = await deps.llm.complete_json(
            system=system, user=user, schema=QuestionPlan
        )
    except Exception as exc:  # noqa: BLE001 - keystone must still emit a valid plan
        log.warning("question_planner failed, using tailored fallback plan (%s)", exc)
        plan = _fallback_plan(
            candidate=state["candidate"],
            job=state["job"],
            company=state["company"],
            language_mode=req.language_mode,
        )
        await _warn(
            state,
            deps,
            ["Could not tailor the question plan with LLM; used a structured fallback plan."],
        )
    # Pin the language mode to the request so the live loop routes voice correctly,
    # regardless of what the model echoed back.
    plan = plan.model_copy(update={"language_mode": req.language_mode})
    await _mark(state, deps, "question_planner")
    return {"plan": plan}
