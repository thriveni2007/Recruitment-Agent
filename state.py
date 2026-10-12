"""
LangGraph state definition for the Recruitment Agent.

Defines the TypedDict-based state that flows through the graph nodes,
carrying job descriptions, parsed profiles, scorecards, trajectory logs,
and guardrail metadata.
"""

from __future__ import annotations

from typing import TypedDict, List, Dict, Any, Optional, Literal

from .models import (
    CandidateProfile,
    ScoreCard,
    InterviewSlot,
    TrajectoryStep,
    GuardrailStatus,
)


class RecruitmentState(TypedDict):
    """
    The complete state carried through the LangGraph execution.

    Attributes:
        jd: Raw job description text.
        candidates: Raw resume texts keyed by candidate name.
        parsed_profiles: Structured profiles keyed by candidate name.
        scorecards: Completed scorecards keyed by candidate name.
        verified_scores: Scorecards that have passed fairness validation.
        revision_count: Number of revision cycles performed.
        shortlist: Ranked list of candidate score entries.
        trajectory: Ordered list of logged reasoning steps.
        guardrail_status: Status of all active guardrails.
        pending_action: Action awaiting human approval, if any.
        status: Overall pipeline status.
        error: Error message if the pipeline failed.
        interview_slots: Available interview slots per candidate.
        interview_confirmations: Confirmed interview bookings.
        candidate_order: Processing order of candidates.
        current_candidate: Name of the candidate currently being processed.
        final_report: Generated final summary report.
    """

    jd: str
    candidates: Dict[str, str]
    parsed_profiles: Dict[str, CandidateProfile]
    scorecards: Dict[str, ScoreCard]
    verified_scores: Dict[str, ScoreCard]
    revision_count: int
    shortlist: List[Dict[str, Any]]
    trajectory: List[TrajectoryStep]
    guardrail_status: GuardrailStatus
    pending_action: Optional[Dict[str, Any]]
    status: Literal["running", "completed", "error", "awaiting_approval"]
    error: Optional[str]
    interview_slots: Dict[str, List[InterviewSlot]]
    interview_confirmations: Dict[str, Dict[str, Any]]
    candidate_order: List[str]
    current_candidate: Optional[str]
    final_report: Optional[str]


def create_initial_state(
    jd: str,
    candidates: Dict[str, str],
    candidate_order: Optional[List[str]] = None,
) -> RecruitmentState:
    """
    Create a fresh initial state for a recruitment run.

    Args:
        jd: Job description text.
        candidates: Dict mapping candidate names to resume texts.
        candidate_order: Optional explicit processing order.
            Defaults to the dict insertion order.

    Returns:
        A populated RecruitmentState ready for graph execution.
    """
    if candidate_order is None:
        candidate_order = list(candidates.keys())

    return RecruitmentState(
        jd=jd,
        candidates=candidates,
        parsed_profiles={},
        scorecards={},
        verified_scores={},
        revision_count=0,
        shortlist=[],
        trajectory=[],
        guardrail_status=GuardrailStatus(),
        pending_action=None,
        status="running",
        error=None,
        interview_slots={},
        interview_confirmations={},
        candidate_order=candidate_order,
        current_candidate=None,
        final_report=None,
    )