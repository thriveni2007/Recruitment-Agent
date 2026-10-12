"""
Pydantic v2 models for the TechVest Recruitment Agent.

Defines all data structures: CandidateProfile, ScoreCard, InterviewSlot,
TrajectoryStep, GuardrailStatus, and the state model.
"""

from __future__ import annotations

import uuid
from typing import List, Dict, Any, Optional, Literal

from pydantic import BaseModel, Field


class CandidateProfile(BaseModel):
    """Structured profile extracted from a candidate's resume."""

    name: str = Field(description="Full name of the candidate")
    skills: List[str] = Field(default_factory=list, description="Technical and soft skills")
    years_experience: float = Field(default=0.0, description="Total years of professional experience")
    education: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Education history entries with degree, institution, year",
    )
    projects: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Project entries with name, description, technologies",
    )
    certifications: List[str] = Field(default_factory=list, description="Professional certifications")

    def model_dump_safe(self) -> Dict[str, Any]:
        """Serialize to dict without raw text snippets."""
        return {
            "name": self.name,
            "skills": self.skills,
            "years_experience": self.years_experience,
            "education": self.education,
            "projects": self.projects,
            "certifications": self.certifications,
        }


class CriterionScore(BaseModel):
    """Score for a single rubric criterion with supporting evidence."""

    criterion: str = Field(description="Name of the scoring criterion")
    weight: float = Field(ge=0.0, le=1.0, description="Weight of this criterion (0.0 to 1.0)")
    score: int = Field(ge=0, le=5, description="Score on a 0–5 scale")
    evidence: str = Field(description="Specific evidence from the resume supporting this score")
    justification: str = Field(description="Brief justification explaining the score")


class ScoreCard(BaseModel):
    """Complete scorecard for one candidate with per-criterion scores and recommendation."""

    candidate_name: str = Field(description="Name of the scored candidate")
    criterion_scores: List[CriterionScore] = Field(description="List of per-criterion scores")
    total_score: float = Field(ge=0.0, le=5.0, description="Weighted total score (0.0–5.0)")
    recommendation: Literal["INTERVIEW", "HOLD", "REJECT"] = Field(
        description="Final recommendation based on total score",
    )
    justification: str = Field(
        default="",
        description="Aggregated justification summarizing the scorecard",
    )


class InterviewSlot(BaseModel):
    """An available interview time slot."""

    date: str = Field(description="Date of the interview slot (YYYY-MM-DD)")
    time: str = Field(description="Time of the interview slot (HH:MM)")
    interviewer: str = Field(default="TechVest Hiring Team", description="Interviewer name")


class TrajectoryStep(BaseModel):
    """
    A single logged step in the agent's reasoning trajectory.

    Enhanced with stage, node, state_updated, and next_node fields
    so the trace reads like an execution graph from START to END.
    """

    # Stage in the workflow pipeline
    stage: str = Field(
        default="Processing",
        description="Current workflow stage (Resume Analysis, Candidate Evaluation, Verification, Shortlisting, Scheduling, Human Approval, Finished)",
    )
    # LangGraph node / agent name
    node: str = Field(
        default="Planner",
        description="Current LangGraph node or agent (Resume Analyst, Scorer, Verifier, Coordinator, Scheduler, Human Approval Gate, Finalizer)",
    )
    thought: str = Field(description="What the agent decided to do")
    tool: str = Field(description="Which tool was called")
    arguments: Dict[str, Any] = Field(default_factory=dict, description="Tool arguments")
    observation: str = Field(description="What the tool returned")
    # Which state keys were updated
    state_updated: str = Field(
        default="",
        description="State keys that were modified by this step (e.g. parsed_profiles, scorecards, shortlist)",
    )
    decision: str = Field(description="Next decision based on observation")
    # Next node / next stage in the graph
    next_node: str = Field(
        default="Planner",
        description="Next LangGraph node to execute (e.g. Scorer Agent, Verifier Agent, Coordinator, Scheduler, Human Approval, END)",
    )
    timestamp: str = Field(description="ISO 8601 timestamp of the step")


class GuardrailStatus(BaseModel):
    """Tracks the status of all active guardrails."""

    human_approval_required: bool = Field(default=True, description="Human-in-the-loop is active")
    recursion_limit: int = Field(default=50, description="Maximum graph recursion steps")
    iteration_cap: int = Field(default=25, description="Maximum node iterations")
    injection_detected: bool = Field(default=False, description="Prompt injection was detected")
    injection_blocked: bool = Field(default=False, description="Injection was successfully blocked")
    fairness_validated: bool = Field(default=False, description="Fairness check completed")
    fairness_note: str = Field(default="", description="Fairness validation result note")


def generate_interview_confirmation_id() -> str:
    """Generate a short unique confirmation ID for an interview."""
    return uuid.uuid4().hex[:8].upper()