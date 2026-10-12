"""
LangGraph nodes for the TechVest Recruitment Agent.

Each node is a callable function that takes the state dict and returns
an updated state dict. The graph orchestrates execution order via
conditional routing.

Nodes:
- planner: Decides the next action dynamically.
- resume_parser: Parses a candidate's resume into a structured profile.
- scorer: Scores a candidate against the rubric.
- availability_checker: Checks interview availability for shortlisted candidates.
- decision_maker: Proposes interviews and handles approval logic.
- human_approval: Processes human approval/rejection of interview proposals.
- interview_scheduler: Confirms and schedules interviews.
- finish_node: Generates the final report and marks completion.
"""

from __future__ import annotations

import json
import logging
from copy import deepcopy
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from .models import (
    CandidateProfile,
    ScoreCard,
    InterviewSlot,
    GuardrailStatus,
    generate_interview_confirmation_id,
)
from .state import RecruitmentState
from .llm_client import LLMClient
from .tools import (
    parse_resume_tool,
    score_candidate_tool,
    check_availability_tool,
    propose_interview_tool,
)
from .trajectory import log_trajectory_step
from .guardrails import (
    check_prompt_injection,
    check_recursion_limit,
    check_iteration_cap,
    validate_fairness,
    update_guardrail_status,
)
from .prompts import DEFAULT_RUBRIC
from .routing import build_plan_summary, determine_next_action

logger = logging.getLogger(__name__)


# ── Planner Node ───────────────────────────────────────────────────────────

def planner_node(state: RecruitmentState) -> Dict[str, Any]:
    """
    Planner node that decides the next action dynamically.

    Uses the routing logic to determine what needs to happen next
    based on the current state.

    Args:
        state: The current recruitment state.

    Returns:
        Updates to the state.
    """
    logger.info("─" * 50)
    logger.info("PLANNER: Evaluating state and determining next action")

    # Check guardrails
    if check_recursion_limit(state):
        return {
            "status": "error",
            "error": "Recursion limit exceeded — max 50 steps",
        }

    if check_iteration_cap(state):
        return {
            "status": "error",
            "error": f"Iteration cap exceeded — max {ITERATION_CAP} revisions",
        }

    from .guardrails import ITERATION_CAP
    if check_iteration_cap(state):
        return {
            "status": "error",
            "error": f"Iteration cap exceeded — max {ITERATION_CAP} revisions",
        }

    # Use LLM-based planning if available
    llm_client = state.get("_llm_client")
    if llm_client:
        try:
            summary = build_plan_summary(state)
            plan = llm_client.plan_next_action(summary)
            action = plan.get("action", "finish")
            candidate = plan.get("candidate")
            reasoning = plan.get("reasoning", "")

            logger.info("LLM Planner: action=%s candidate=%s reason=%s", action, candidate, reasoning)

            updates: Dict[str, Any] = {
                "revision_count": state.get("revision_count", 0) + 1,
            }

            if candidate:
                updates["current_candidate"] = candidate

            log_trajectory_step(
                state,
                thought=f"Planner decision: {action} — {reasoning}",
                tool="planner",
                arguments={"plan": plan},
                observation=f"Next action: {action}, candidate: {candidate}",
                decision=action,
            )

            return updates
        except Exception as exc:
            logger.warning("LLM planning failed, using rule-based fallback: %s", exc)

    # Rule-based fallback
    action = determine_next_action(state)
    logger.info("Rule-based planner: action=%s", action)

    state["revision_count"] = state.get("revision_count", 0) + 1

    log_trajectory_step(
        state,
        thought=f"Rule-based planning: next action is {action}",
        tool="planner",
        arguments={},
        observation=f"Determined next action: {action}",
        decision=action,
    )

    return {"revision_count": state["revision_count"]}


# ── Resume Parser Node ────────────────────────────────────────────────────

def resume_parser_node(state: RecruitmentState) -> Dict[str, Any]:
    """
    Parse the current candidate's resume into a structured profile.

    Also runs injection detection on the resume text.

    Args:
        state: The current recruitment state.

    Returns:
        Updates to the state with the parsed profile.
    """
    candidate_name: Optional[str] = state.get("current_candidate")
    if not candidate_name:
        # Find first unparsed candidate
        for name in state.get("candidates", {}):
            if name not in state.get("parsed_profiles", {}):
                candidate_name = name
                state["current_candidate"] = name
                break

    if not candidate_name:
        logger.warning("No candidates to parse")
        return {"error": "No candidates to parse"}

    logger.info("PARSER: Parsing resume for %s", candidate_name)

    resume_text: str = state.get("candidates", {}).get(candidate_name, "")
    if not resume_text:
        return {"error": f"No resume text found for {candidate_name}"}

    # ── Guardrail 4: Injection Detection ──
    llm_client: Optional[LLMClient] = state.get("_llm_client")
    injection_detected = check_prompt_injection(resume_text, llm_client)

    guardrail_status: GuardrailStatus = state.get("guardrail_status", GuardrailStatus())
    if injection_detected:
        logger.warning("INJECTION DETECTED in resume: %s", candidate_name)
        guardrail_status = update_guardrail_status(
            guardrail_status,
            injection_detected=True,
            injection_blocked=True,
        )
        observation = (
            f"INJECTION DETECTED in '{candidate_name}'s resume. "
            f"The manipulative instructions have been blocked. "
            f"Parsing only factual content."
        )
    else:
        observation = "No injection detected."

    # Parse the resume
    result = parse_resume_tool(resume_text, llm_client)

    if not result["success"]:
        return {"error": result["error"]}

    profile: CandidateProfile = result["profile"]
    parsed_profiles: Dict[str, CandidateProfile] = dict(state.get("parsed_profiles", {}))
    parsed_profiles[candidate_name] = profile

    obs_detail = (
        f"{observation} Profile parsed: {profile.name}, "
        f"{len(profile.skills)} skills, "
        f"{profile.years_experience} years exp, "
        f"{len(profile.projects)} projects"
    )

    log_trajectory_step(
        state,
        thought=f"Parse resume for {candidate_name}",
        tool="parse_resume",
        arguments={"candidate": candidate_name, "resume_length": len(resume_text)},
        observation=obs_detail,
        decision="Continue to next action",
    )

    return {
        "parsed_profiles": parsed_profiles,
        "guardrail_status": guardrail_status,
    }


# ── Scorer Node ────────────────────────────────────────────────────────────

def scorer_node(state: RecruitmentState) -> Dict[str, Any]:
    """
    Score the current candidate's profile against the rubric.

    Args:
        state: The current recruitment state.

    Returns:
        Updates to the state with the scorecard.
    """
    candidate_name: Optional[str] = state.get("current_candidate")
    if not candidate_name:
        # Find first parsed but unscored candidate
        for name in state.get("parsed_profiles", {}):
            if name not in state.get("scorecards", {}):
                candidate_name = name
                state["current_candidate"] = name
                break

    if not candidate_name:
        logger.warning("No candidates to score")
        return {"error": "No candidates to score"}

    logger.info("SCORER: Scoring candidate %s", candidate_name)

    profile: Optional[CandidateProfile] = state.get("parsed_profiles", {}).get(candidate_name)
    if not profile:
        return {"error": f"No parsed profile for {candidate_name}"}

    llm_client: Optional[LLMClient] = state.get("_llm_client")
    jd: str = state.get("jd", "")

    result = score_candidate_tool(
        profile=profile,
        llm_client=llm_client,
        rubric=DEFAULT_RUBRIC,
        jd=jd,
    )

    if not result["success"]:
        return {"error": result["error"]}

    scorecard: ScoreCard = result["scorecard"]
    scorecards: Dict[str, ScoreCard] = dict(state.get("scorecards", {}))
    scorecards[candidate_name] = scorecard

    # Build evidence summary for trajectory
    evidence_lines = []
    for cs in scorecard.criterion_scores:
        evidence_lines.append(
            f"  {cs.criterion}: {cs.score}/5 (w:{cs.weight}) — {cs.evidence[:80]}"
        )
    obs = (
        f"Scorecard for {candidate_name}:\n"
        + "\n".join(evidence_lines)
        + f"\nTotal: {scorecard.total_score:.2f}/5.0 → {scorecard.recommendation}"
    )

    # Add to shortlist
    shortlist: List[Dict[str, Any]] = list(state.get("shortlist", []))
    shortlist.append({
        "candidate": candidate_name,
        "scorecard": scorecard.model_dump(),
        "total_score": scorecard.total_score,
        "recommendation": scorecard.recommendation,
        "justification": scorecard.justification,
    })

    log_trajectory_step(
        state,
        thought=f"Score {candidate_name} against rubric",
        tool="score_candidate",
        arguments={"candidate": candidate_name, "criteria_count": len(scorecard.criterion_scores)},
        observation=obs,
        decision=f"Recommendation: {scorecard.recommendation}",
    )

    return {
        "scorecards": scorecards,
        "shortlist": shortlist,
    }


# ── Availability Checker Node ──────────────────────────────────────────────

def availability_checker_node(state: RecruitmentState) -> Dict[str, Any]:
    """
    Check interview availability for shortlisted candidates.

    Args:
        state: The current recruitment state.

    Returns:
        Updates to the state with interview slots.
    """
    candidate_name: Optional[str] = state.get("current_candidate")
    if not candidate_name:
        # Find first shortlisted candidate without availability check
        shortlist = state.get("shortlist", [])
        interview_slots = state.get("interview_slots", {})
        for entry in shortlist:
            if entry["recommendation"] == "INTERVIEW" and entry["candidate"] not in interview_slots:
                candidate_name = entry["candidate"]
                state["current_candidate"] = candidate_name
                break

    if not candidate_name:
        logger.warning("No candidates for availability check")
        return {"error": "No candidates for availability check"}

    logger.info("AVAILABILITY: Checking slots for %s", candidate_name)

    llm_client: Optional[LLMClient] = state.get("_llm_client")

    result = check_availability_tool(candidate_name, llm_client)

    if not result["success"]:
        return {"error": result["error"]}

    slots: List[InterviewSlot] = result["slots"]
    interview_slots: Dict[str, List[InterviewSlot]] = dict(state.get("interview_slots", {}))
    interview_slots[candidate_name] = slots

    slot_str = ", ".join([f"{s.date} {s.time}" for s in slots])

    log_trajectory_step(
        state,
        thought=f"Check interview availability for {candidate_name}",
        tool="check_availability",
        arguments={"candidate": candidate_name},
        observation=f"Available slots for {candidate_name}: {slot_str}",
        decision="Proceed to propose interview",
    )

    return {"interview_slots": interview_slots}


# ── Decision Maker Node ────────────────────────────────────────────────────

def decision_maker_node(state: RecruitmentState) -> Dict[str, Any]:
    """
    Propose interviews for shortlisted candidates with available slots.

    This node creates interview proposals that require human approval.
    It never executes automatically — it sets pending_action and
    switches status to awaiting_approval.

    Args:
        state: The current recruitment state.

    Returns:
        Updates to the state with the interview proposal.
    """
    candidate_name: Optional[str] = state.get("current_candidate")

    if not candidate_name:
        # Find first shortlisted with slots but no proposal
        shortlist = state.get("shortlist", [])
        interview_slots = state.get("interview_slots", {})
        interview_confirmations = state.get("interview_confirmations", {})
        for entry in shortlist:
            if entry["recommendation"] == "INTERVIEW":
                name = entry["candidate"]
                if name in interview_slots and name not in interview_confirmations:
                    candidate_name = name
                    state["current_candidate"] = name
                    break

    if not candidate_name:
        logger.warning("No candidates for interview proposal")
        return {"error": "No candidates for interview proposal"}

    slots = state.get("interview_slots", {}).get(candidate_name, [])
    if not slots:
        return {"error": f"No interview slots for {candidate_name}"}

    logger.info("DECISION: Proposing interview for %s", candidate_name)

    # Propose the first available slot
    proposed_slot = slots[0]
    proposal = propose_interview_tool(candidate_name, proposed_slot)

    if not proposal["success"]:
        return {"error": proposal.get("error", "Interview proposal failed")}

    confirmation_id = proposal["confirmation_id"]
    pending_action = {
        "type": "interview_proposal",
        "candidate": candidate_name,
        "slot": proposed_slot.model_dump(),
        "confirmation_id": confirmation_id,
        "status": "pending_approval",
        "message": proposal["message"],
    }

    interview_confirmations: Dict[str, Dict[str, Any]] = dict(
        state.get("interview_confirmations", {})
    )
    interview_confirmations[candidate_name] = {
        "candidate": candidate_name,
        "slot": proposed_slot.model_dump(),
        "status": "pending_approval",
        "confirmation_id": confirmation_id,
    }

    log_trajectory_step(
        state,
        thought=f"Propose interview for {candidate_name}",
        tool="propose_interview",
        arguments={
            "candidate": candidate_name,
            "slot": proposed_slot.model_dump(),
        },
        observation=proposal["message"],
        decision="Awaiting human approval",
    )

    return {
        "pending_action": pending_action,
        "interview_confirmations": interview_confirmations,
        "status": "awaiting_approval",
    }


# ── Human Approval Node ────────────────────────────────────────────────────

def human_approval_node(state: RecruitmentState) -> Dict[str, Any]:
    """
    Human approval node.

    This node checks the status of the pending action.
    If it has been approved (via the UI), it schedules the interview.
    If rejected, it notes the rejection and moves on.

    The actual approval decision is made in the Streamlit UI via
    approve_interview() which modifies the state.

    Args:
        state: The current recruitment state.

    Returns:
        Updates to the state.
    """
    pending_action = state.get("pending_action")
    if not pending_action:
        logger.info("No pending action — continuing")
        return {"status": "running"}

    candidate_name = pending_action.get("candidate", "unknown")
    logger.info("APPROVAL: Checking approval status for %s", candidate_name)

    interview_confirmations = state.get("interview_confirmations", {})
    candidate_conf = interview_confirmations.get(candidate_name, {})

    if isinstance(candidate_conf, dict) and candidate_conf.get("status") == "confirmed":
        logger.info("Interview confirmed for %s — scheduling", candidate_name)
        return {
            "pending_action": None,
            "status": "running",
            "current_candidate": candidate_name,
        }

    if isinstance(candidate_conf, dict) and candidate_conf.get("status") == "rejected":
        logger.info("Interview rejected for %s — moving on", candidate_name)
        return {
            "pending_action": None,
            "status": "running",
        }

    # Still pending — stay in awaiting_approval
    logger.info("Interview still pending approval for %s", candidate_name)
    return {}


# ── Interview Scheduler Node ───────────────────────────────────────────────

def interview_scheduler_node(state: RecruitmentState) -> Dict[str, Any]:
    """
    Schedule a confirmed interview.

    This node runs after human approval has been granted.
    It marks the interview as scheduled in the state.

    Args:
        state: The current recruitment state.

    Returns:
        Updates to the state with the scheduled interview.
    """
    candidate_name: Optional[str] = state.get("current_candidate")

    if not candidate_name:
        # Find first confirmed candidate
        interview_confirmations = state.get("interview_confirmations", {})
        for name, conf in interview_confirmations.items():
            if isinstance(conf, dict) and conf.get("status") == "confirmed":
                candidate_name = name
                state["current_candidate"] = name
                break

    if not candidate_name:
        logger.warning("No candidate to schedule interview for")
        return {"error": "No confirmed interview to schedule"}

    logger.info("SCHEDULER: Scheduling interview for %s", candidate_name)

    interview_confirmations = state.get("interview_confirmations", {})
    candidate_conf = interview_confirmations.get(candidate_name, {})

    if isinstance(candidate_conf, dict):
        candidate_conf["status"] = "scheduled"
        interview_confirmations[candidate_name] = candidate_conf

    log_trajectory_step(
        state,
        thought=f"Schedule interview for {candidate_name}",
        tool="schedule_interview",
        arguments={"candidate": candidate_name},
        observation=f"Interview scheduled for {candidate_name}",
        decision="Interview confirmed and scheduled",
    )

    return {"interview_confirmations": interview_confirmations}


# ── Finish Node ────────────────────────────────────────────────────────────

def finish_node(state: RecruitmentState) -> Dict[str, Any]:
    """
    Generate the final report and mark the pipeline as complete.

    Runs fairness validation and produces a comprehensive report.

    Args:
        state: The current recruitment state.

    Returns:
        Updates to the state with the final report.
    """
    logger.info("FINISH: Generating final report")

    # ── Guardrail 5: Fairness Validation ──
    llm_client: Optional[LLMClient] = state.get("_llm_client")
    fairness_result = validate_fairness(state, llm_client)

    guardrail_status: GuardrailStatus = state.get("guardrail_status", GuardrailStatus())
    guardrail_status = update_guardrail_status(
        guardrail_status,
        fairness_result=fairness_result,
    )

    if fairness_result.get("bias_detected", False):
        logger.warning("Fairness check detected potential bias")
        log_trajectory_step(
            state,
            thought="Run fairness validation",
            tool="fairness_check",
            arguments={"fairness_result": fairness_result},
            observation=f"Fairness result: {fairness_result.get('note', '')}",
            decision="Fairness check completed with warnings",
        )
    else:
        log_trajectory_step(
            state,
            thought="Run fairness validation",
            tool="fairness_check",
            arguments={},
            observation="Fairness check passed — no bias detected",
            decision="Fairness check completed",
        )

    # Build final report
    shortlist = state.get("shortlist", [])
    sorted_shortlist = sorted(
        shortlist,
        key=lambda e: e.get("total_score", 0),
        reverse=True,
    )
    interview_confirmations = state.get("interview_confirmations", {})
    trajectory = state.get("trajectory", [])

    lines: List[str] = []
    lines.append("=" * 60)
    lines.append("TECHVEST RECRUITMENT AGENT — FINAL REPORT")
    lines.append("=" * 60)
    lines.append(f"Generated: {datetime.now(timezone.utc).isoformat()}")
    lines.append(f"Status: {'✅ Completed' if state.get('status') != 'error' else '❌ Error'}")
    lines.append("")

    lines.append("─" * 60)
    lines.append("RANKED CANDIDATES")
    lines.append("─" * 60)
    for i, entry in enumerate(sorted_shortlist):
        lines.append(f"\nRank #{i + 1}: {entry['candidate']}")
        lines.append(f"  Score: {entry['total_score']:.2f}/5.0")
        lines.append(f"  Recommendation: {entry['recommendation']}")
        lines.append(f"  Justification: {entry.get('justification', 'N/A')}")

        scorecard_data = entry.get("scorecard", {})
        criterion_scores = scorecard_data.get("criterion_scores", [])
        for cs in criterion_scores:
            lines.append(f"    • {cs['criterion']}: {cs['score']}/5 (w:{cs['weight']})")
            lines.append(f"      Evidence: {cs.get('evidence', 'N/A')}")

    lines.append("\n" + "─" * 60)
    lines.append("INTERVIEW STATUS")
    lines.append("─" * 60)
    for entry in sorted_shortlist:
        name = entry["candidate"]
        conf = interview_confirmations.get(name, {})
        if isinstance(conf, dict):
            status = conf.get("status", "not_proposed")
            slot = conf.get("slot", {})
            lines.append(f"  {name}: {status}")
            if slot:
                lines.append(f"    Slot: {slot.get('date', '')} at {slot.get('time', '')}")
        else:
            lines.append(f"  {name}: not proposed")

    lines.append("\n" + "─" * 60)
    lines.append("GUARDRAIL STATUS")
    lines.append("─" * 60)
    lines.append(f"  Human-in-the-loop:     {'✅ Active' if guardrail_status.human_approval_required else '❌ Inactive'}")
    lines.append(f"  Recursion Limit:       {'✅ ' + str(guardrail_status.recursion_limit) + ' max steps'}")
    lines.append(f"  Iteration Cap:         {'✅ ' + str(guardrail_status.iteration_cap) + ' max iterations'}")
    lines.append(f"  Injection Defence:     {'🚨 Triggered & Blocked' if guardrail_status.injection_detected else '✅ No threats'}")
    lines.append(f"  Fairness Validation:   {'✅ Passed' if guardrail_status.fairness_validated else '⚠️ Not run'}")
    lines.append(f"  Audit Log:             {'✅ ' + str(len(trajectory)) + ' steps logged'}")

    if guardrail_status.injection_detected:
        lines.append("\n  🚨 PROMPT INJECTION ATTACK BLOCKED")
        lines.append("  Meera Patel's resume contained manipulative instructions.")
        lines.append("  These instructions were detected and ignored.")

    if guardrail_status.fairness_validated:
        lines.append(f"\n  Fairness Note: {guardrail_status.fairness_note}")

    lines.append("\n" + "=" * 60)
    lines.append("END OF REPORT")
    lines.append("=" * 60)

    final_report = "\n".join(lines)

    log_trajectory_step(
        state,
        thought="Generate final report",
        tool="finalize",
        arguments={},
        observation="Recruitment process complete",
        decision="Pipeline finished",
    )

    return {
        "status": "completed",
        "final_report": final_report,
        "guardrail_status": guardrail_status,
        "verified_scores": state.get("scorecards", {}),
    }


def create_state_with_client(state: RecruitmentState, llm_client: LLMClient) -> RecruitmentState:
    """
    Stub kept for backwards compatibility. The graph no longer stores
    the LLM client in state — it's accessed via the graph instance.

    Args:
        state: The recruitment state.
        llm_client: The LLM client instance (ignored).

    Returns:
        State unchanged.
    """
    return state
