"""
Conditional routing logic for the LangGraph recruitment agent.

The planner node dynamically determines the next action based on
the current state. Edges are NOT hardcoded as a linear pipeline —
the graph uses conditional edges with LLM-based planning.

Routing decisions:
- If profile missing → Parse Resume
- If score missing → Score Candidate
- If shortlisted → Check Availability
- If approval pending → Wait (Human Approval)
- If approved → Schedule Interview
- Else → Finish
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, Literal, Optional

from .models import CandidateProfile, ScoreCard
from .state import RecruitmentState

logger = logging.getLogger(__name__)


RouteName = Literal[
    "parse_resume",
    "score_candidate",
    "check_availability",
    "human_approval",
    "schedule_interview",
    "finish",
    "continue",
]


def determine_next_action(state: RecruitmentState) -> RouteName:
    """
    Determine the next action based on the current state.

    This function implements the planner logic with conditional edges.
    Each check looks at what's missing in the state and routes accordingly.

    Args:
        state: The current recruitment state.

    Returns:
        The name of the next node to execute.
    """
    # Check guardrails first
    guardrails = state.get("guardrail_status")
    if guardrails and guardrails.injection_blocked:
        logger.info("Injection was blocked — continuing processing")

    # If the pipeline is complete or errored, finish
    status = state.get("status", "running")
    if status in ("completed", "error"):
        logger.info("Status is '%s' — routing to finish", status)
        return "finish"

    # If there's a pending action awaiting human approval
    pending = state.get("pending_action")
    if pending is not None:
        logger.info("Action pending human approval — routing to human_approval")
        return "human_approval"

    # Check for candidates that haven't been parsed yet
    candidates: Dict[str, str] = state.get("candidates", {})
    parsed_profiles: Dict[str, CandidateProfile] = state.get("parsed_profiles", {})

    for candidate_name in candidates:
        if candidate_name not in parsed_profiles:
            logger.info("Resume not parsed for %s — routing to parse_resume", candidate_name)
            state["current_candidate"] = candidate_name
            return "parse_resume"

    # Check for parsed candidates that haven't been scored
    scorecards: Dict[str, ScoreCard] = state.get("scorecards", {})

    for candidate_name in parsed_profiles:
        if candidate_name not in scorecards:
            logger.info("Profile not scored for %s — routing to score_candidate", candidate_name)
            state["current_candidate"] = candidate_name
            return "score_candidate"

    # All candidates parsed and scored — check shortlist
    shortlist: list = state.get("shortlist", [])
    interview_slots: dict = state.get("interview_slots", {})
    interview_confirmations: dict = state.get("interview_confirmations", {})

    # Get shortlisted candidates
    shortlisted_names = [entry["candidate"] for entry in shortlist if entry["recommendation"] == "INTERVIEW"]

    if not shortlisted_names:
        # No candidates were shortlisted for interview — all done
        logger.info("No candidates shortlisted — routing to finish")
        return "finish"

    # If there are shortlisted candidates with no availability check
    for name in shortlisted_names:
        if name not in interview_slots:
            logger.info("Availability not checked for %s — routing to check_availability", name)
            state["current_candidate"] = name
            return "check_availability"

    # If availability checked but not yet proposed
    for name in shortlisted_names:
        if name in interview_slots and name not in interview_confirmations:
            logger.info("Interview not yet proposed for %s — routing to human_approval", name)
            state["current_candidate"] = name
            return "human_approval"

    # If some interviews are confirmed but others remain
    confirmed_count = sum(
        1 for conf in interview_confirmations.values()
        if isinstance(conf, dict) and conf.get("status") == "confirmed"
    )
    if confirmed_count < len(shortlisted_names):
        logger.info("Not all shortlisted candidates have confirmed interviews — continuing")
        return "continue"

    # All done
    logger.info("All candidates processed — routing to finish")
    return "finish"


def build_plan_summary(state: RecruitmentState) -> str:
    """
    Build a text summary of the current state for the planner LLM.

    Args:
        state: The current recruitment state.

    Returns:
        A string summarising the state.
    """
    candidates = list(state.get("candidates", {}).keys())
    parsed = list(state.get("parsed_profiles", {}).keys())
    scored = list(state.get("scorecards", {}).keys())
    shortlist = state.get("shortlist", [])
    interview_slots = state.get("interview_slots", {})

    lines = [
        f"Candidates: {', '.join(candidates)}",
        f"Parsed: {', '.join(parsed) if parsed else 'none'}",
        f"Scored: {', '.join(scored) if scored else 'none'}",
        f"Shortlisted: {len([e for e in shortlist if e['recommendation'] == 'INTERVIEW'])}",
        f"Availability checked for: {', '.join(interview_slots.keys()) if interview_slots else 'none'}",
        f"Revision count: {state.get('revision_count', 0)}",
        f"Status: {state.get('status', 'running')}",
        f"Steps in trajectory: {len(state.get('trajectory', []))}",
    ]
    return "\n".join(lines)