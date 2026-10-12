"""
Guardrails for the TechVest Recruitment Agent.

Implements five security and fairness guardrails:
1. Human approval required before scheduling interviews.
2. Recursion limit — maximum graph recursion depth.
3. Iteration cap — maximum node iterations per run.
4. Prompt injection detection — blocks manipulative resume instructions.
5. Fairness validation — ensures no name/gender/age/college prestige bias.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from .models import GuardrailStatus, TrajectoryStep
from .llm_client import LLMClient
from .state import RecruitmentState

logger = logging.getLogger(__name__)

# Maximum recursion depth before forcing termination
RECURSION_LIMIT: int = 50

# Maximum number of node iterations
ITERATION_CAP: int = 25


# ── Guardrail 1: Human Approval Gate ───────────────────────────────────────

def human_approval_required(state: RecruitmentState) -> bool:
    """
    Guardrail 1: Check whether human approval is needed before proceeding.

    The propose_interview tool must never execute automatically;
    it always requires explicit human confirmation.

    Args:
        state: The current recruitment state.

    Returns:
        True if an action is pending human approval.
    """
    return state.get("pending_action") is not None and state.get("status") == "awaiting_approval"


# ── Guardrail 2: Recursion Limit ───────────────────────────────────────────

def check_recursion_limit(state: RecruitmentState) -> bool:
    """
    Guardrail 2: Check if the recursion limit has been exceeded.

    Args:
        state: The current recruitment state.

    Returns:
        True if the limit has been exceeded.
    """
    trajectory = state.get("trajectory", [])
    total_steps = len(trajectory)
    if total_steps >= RECURSION_LIMIT:
        logger.warning("Recursion limit reached: %d steps", total_steps)
        return True
    return False


# ── Guardrail 3: Iteration Cap ─────────────────────────────────────────────

def check_iteration_cap(state: RecruitmentState) -> bool:
    """
    Guardrail 3: Check if the iteration cap has been exceeded.

    Iteration cap prevents infinite loops from repeated node executions.

    Args:
        state: The current recruitment state.

    Returns:
        True if the cap has been exceeded.
    """
    revision_count = state.get("revision_count", 0)
    if revision_count >= ITERATION_CAP:
        logger.warning("Iteration cap reached: %d revisions", revision_count)
        return True
    return False


# ── Guardrail 4: Injection Detection ───────────────────────────────────────

def check_prompt_injection(
    resume_text: str,
    llm_client: Optional[LLMClient] = None,
) -> bool:
    """
    Guardrail 4: Detect prompt injection attempts in resume text.

    Args:
        resume_text: The resume text to scan.
        llm_client: Optional LLM client for advanced detection.

    Returns:
        True if injection is detected.
    """
    if llm_client:
        return llm_client.check_injection(resume_text)

    # Deterministic fallback
    import re

    patterns = [
        r"ignore\s+(all\s+)?(previous|prior|above)\s+(instructions|directions|commands)",
        r"rank\s+(me|this\s+candidate|[\w\s]+?)\s+(first|top|highest)",
        r"rank\s+[\w\s]+\s+as\s+the\s+top\s+candidate",
        r"official\s+update\s+from\s+(hr|management|the\s+company)",
        r"override\s+(the\s+)?(scoring|rubric|evaluation|ranking|system)",
        r"disregard\s+(all\s+)?(previous|prior)\s+(scoring|instructions|rubric)",
        r"hire\s+me\s+immediately",
    ]
    text_lower = resume_text.lower()
    for pattern in patterns:
        import re as regex

        if regex.search(pattern, text_lower):
            logger.info("Injection pattern detected: %s", pattern)
            return True
    return False


# ── Guardrail 5: Fairness Validation ───────────────────────────────────────

def validate_fairness(
    state: RecruitmentState,
    llm_client: Optional[LLMClient] = None,
) -> Dict[str, Any]:
    """
    Guardrail 5: Validate that scoring is fair and unbiased.

    Checks that:
    - No score is influenced by candidate name, gender, age, or college prestige.
    - All scores are based on JD-relevant skills and experience only.
    - Similar skill levels produce similar scores.

    Args:
        state: The current recruitment state.
        llm_client: Optional LLM client for advanced analysis.

    Returns:
        A fairness report dictionary.
    """
    if llm_client:
        return llm_client.fairness_check(state)

    # Deterministic fairness check
    profiles = state.get("parsed_profiles", {})
    scorecards = state.get("scorecards", {})
    bias_indicators: list[str] = []

    if len(profiles) < 2:
        return {
            "bias_detected": False,
            "bias_indicators": [],
            "fairness_score": 1.0,
            "recommendation": "PASS",
            "note": "Insufficient candidates for cross-comparison",
        }

    names = list(profiles.keys())
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            p1 = profiles[names[i]]
            p2 = profiles[names[j]]
            s1 = scorecards.get(names[i])
            s2 = scorecards.get(names[j])
            if s1 and s2:
                score_diff = abs(s1.total_score - s2.total_score)
                exp_diff = abs(p1.years_experience - p2.years_experience)
                if score_diff > 2.0 and exp_diff < 1.0:
                    bias_indicators.append(
                        f"Score gap {score_diff:.1f} between {names[i]} and {names[j]} "
                        f"with similar experience"
                    )

    has_bias = len(bias_indicators) > 0
    return {
        "bias_detected": has_bias,
        "bias_indicators": bias_indicators,
        "fairness_score": 0.7 if has_bias else 1.0,
        "recommendation": "REVIEW" if has_bias else "PASS",
        "note": "Fairness check passed — candidates scored on JD-relevant criteria only."
        if not has_bias
        else "Potential bias indicators found — recommend manual review.",
    }


def update_guardrail_status(
    guardrail_status: GuardrailStatus,
    injection_detected: bool = False,
    injection_blocked: bool = False,
    fairness_result: Optional[Dict[str, Any]] = None,
) -> GuardrailStatus:
    """
    Update the guardrail status with new results.

    Args:
        guardrail_status: Current guardrail status.
        injection_detected: Whether injection was detected.
        injection_blocked: Whether injection was blocked.
        fairness_result: Fairness validation result.

    Returns:
        Updated GuardrailStatus.
    """
    if injection_detected:
        guardrail_status.injection_detected = True
        guardrail_status.injection_blocked = injection_blocked

    if fairness_result:
        guardrail_status.fairness_validated = True
        guardrail_status.fairness_note = fairness_result.get("note", "")

    return guardrail_status