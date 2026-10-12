"""
Tool: parse_resume

Extracts a structured CandidateProfile from raw resume text using the LLM client.
"""

from __future__ import annotations

import logging
from typing import Any, Dict

from ..models import CandidateProfile
from ..llm_client import LLMClient

logger = logging.getLogger(__name__)


def parse_resume_tool(
    resume_text: str,
    llm_client: LLMClient,
) -> Dict[str, Any]:
    """
    Parse a raw resume text into a structured CandidateProfile.

    Args:
        resume_text: The raw text content of the candidate's resume.
        llm_client: The LLM client used for extraction.

    Returns:
        A dictionary with:
            - "success": bool
            - "profile": CandidateProfile (if successful)
            - "error": str (if failed)
    """
    logger.info("Parsing resume (%d chars)", len(resume_text))

    try:
        profile: CandidateProfile = llm_client.parse_resume(resume_text)
        logger.info(
            "Parsed profile: %s | %d skills | %.1f years exp",
            profile.name,
            len(profile.skills),
            profile.years_experience,
        )
        return {
            "success": True,
            "profile": profile,
            "error": None,
        }
    except Exception as exc:
        logger.error("Resume parsing failed: %s", exc)
        return {
            "success": False,
            "profile": None,
            "error": str(exc),
        }