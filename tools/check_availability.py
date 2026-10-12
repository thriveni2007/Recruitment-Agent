"""
Tool: check_availability

Mock tool that returns interview slots for a candidate.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

from ..models import InterviewSlot
from ..llm_client import LLMClient

logger = logging.getLogger(__name__)


def check_availability_tool(
    candidate_name: str,
    llm_client: LLMClient,
) -> Dict[str, Any]:
    """
    Check interview availability for a candidate.

    This is a mock tool that generates available time slots
    in the upcoming week.

    Args:
        candidate_name: Name of the candidate.
        llm_client: The LLM client (provides check_availability).

    Returns:
        A dictionary with:
            - "success": bool
            - "candidate": str
            - "slots": List[InterviewSlot]
            - "error": str (if failed)
    """
    logger.info("Checking availability for candidate: %s", candidate_name)

    try:
        slots: List[InterviewSlot] = llm_client.check_availability(candidate_name)
        logger.info("Found %d slot(s) for %s", len(slots), candidate_name)

        return {
            "success": True,
            "candidate": candidate_name,
            "slots": slots,
            "error": None,
        }
    except Exception as exc:
        logger.error("Availability check failed for %s: %s", candidate_name, exc)
        return {
            "success": False,
            "candidate": candidate_name,
            "slots": [],
            "error": str(exc),
        }