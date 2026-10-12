"""
Tool: propose_interview

Proposes an interview for a shortlisted candidate.

IMPORTANT: This tool NEVER executes automatically.
It always requires human approval before proceeding.
"""

from __future__ import annotations

import logging
import uuid
from typing import Any, Dict

from ..models import InterviewSlot

logger = logging.getLogger(__name__)


def propose_interview_tool(
    candidate_name: str,
    slot: InterviewSlot,
) -> Dict[str, Any]:
    """
    Propose an interview for a candidate at a given slot.

    This tool generates a proposal that MUST be approved by a human
    before any interview is actually scheduled. It never executes
    automatically.

    Args:
        candidate_name: Name of the candidate.
        slot: The proposed interview slot.

    Returns:
        A dictionary with:
            - "success": bool
            - "candidate": str
            - "slot": InterviewSlot
            - "confirmation_id": str
            - "message": str
            - "status": Literal["pending_approval"]
            - "error": str (if failed)
    """
    logger.info("Proposing interview for %s on %s at %s", candidate_name, slot.date, slot.time)

    confirmation_id: str = uuid.uuid4().hex[:8].upper()

    result: Dict[str, Any] = {
        "success": True,
        "candidate": candidate_name,
        "slot": slot,
        "confirmation_id": confirmation_id,
        "message": (
            f"Interview proposed for {candidate_name} on {slot.date} at {slot.time} "
            f"with {slot.interviewer}. "
            f"Confirmation ID: {confirmation_id}. "
            f"AWAITING HUMAN APPROVAL — this action has been paused."
        ),
        "status": "pending_approval",
        "error": None,
    }

    logger.info("Interview proposal created: %s", confirmation_id)
    return result