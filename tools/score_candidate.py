"""
Tool: score_candidate

Scores a parsed candidate profile against the rubric and job description.
Produces a ScoreCard with per-criterion evidence and recommendation.
"""

from __future__ import annotations

import logging
from typing import Any, Dict

from ..models import CandidateProfile, ScoreCard
from ..llm_client import LLMClient
from ..prompts import DEFAULT_RUBRIC

logger = logging.getLogger(__name__)


def score_candidate_tool(
    profile: CandidateProfile,
    llm_client: LLMClient,
    rubric: str = DEFAULT_RUBRIC,
    jd: str = "",
) -> Dict[str, Any]:
    """
    Score a candidate profile against the rubric.

    Every score includes specific evidence from the profile.
    Uses weighted scoring on a 0-5 scale.

    Args:
        profile: The parsed candidate profile.
        llm_client: The LLM client for scoring.
        rubric: Scoring rubric text.
        jd: Job description text.

    Returns:
        A dictionary with:
            - "success": bool
            - "scorecard": ScoreCard (if successful)
            - "error": str (if failed)
    """
    logger.info("Scoring candidate: %s", profile.name)

    try:
        scorecard: ScoreCard = llm_client.score_candidate(
            profile=profile,
            rubric=rubric,
            jd=jd,
        )
        logger.info(
            "Score for %s: %.2f/5.0 → %s",
            scorecard.candidate_name,
            scorecard.total_score,
            scorecard.recommendation,
        )
        return {
            "success": True,
            "scorecard": scorecard,
            "error": None,
        }
    except Exception as exc:
        logger.error("Scoring failed for %s: %s", profile.name, exc)
        return {
            "success": False,
            "scorecard": None,
            "error": str(exc),
        }