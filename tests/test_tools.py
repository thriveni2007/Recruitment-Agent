"""
Tests for the recruitment agent tools: parse_resume, score_candidate, check_availability, propose_interview.
"""

from __future__ import annotations

import pytest
from ..tools.parse_resume import parse_resume_tool
from ..tools.score_candidate import score_candidate_tool
from ..tools.check_availability import check_availability_tool
from ..tools.propose_interview import propose_interview_tool
from ..llm_client import LLMClient
from ..models import CandidateProfile, InterviewSlot


@pytest.fixture
def llm_client() -> LLMClient:
    """Create an LLM client (no API key — uses fallback)."""
    return LLMClient(api_key="", model="test-model")


@pytest.fixture
def sample_resume() -> str:
    return """JOHN DOE
Email: john@email.com

SKILLS
Python, NumPy, Pandas, scikit-learn, LangChain

PROJECTS
1. ML Predictor (2024)
   Built a regression model for house prices.

CERTIFICATIONS
- AWS Certified
"""


class TestParseResumeTool:
    def test_parse_valid_resume(self, llm_client: LLMClient, sample_resume: str) -> None:
        result = parse_resume_tool(sample_resume, llm_client)
        assert result["success"] is True
        profile: CandidateProfile = result["profile"]
        assert profile.name != ""
        assert len(profile.skills) > 0
        assert profile.years_experience >= 0

    def test_parse_empty_resume(self, llm_client: LLMClient) -> None:
        result = parse_resume_tool("", llm_client)
        assert result["success"] is True
        assert result["profile"].name != ""

    def test_parse_returns_profile_object(self, llm_client: LLMClient, sample_resume: str) -> None:
        result = parse_resume_tool(sample_resume, llm_client)
        assert isinstance(result["profile"], CandidateProfile)


class TestScoreCandidateTool:
    @pytest.fixture
    def profile(self) -> CandidateProfile:
        return CandidateProfile(
            name="Test Candidate",
            skills=["Python", "NumPy", "Pandas", "scikit-learn", "LangChain"],
            years_experience=2.0,
            education=[{"degree": "B.Tech CS", "institution": "Test Univ", "year": 2023, "field": "CS"}],
            projects=[{"name": "RAG System", "description": "Built a RAG pipeline with LangChain", "technologies": ["LangChain", "ChromaDB"]}],
            certifications=["AWS Certified"],
        )

    def test_score_valid_profile(self, llm_client: LLMClient, profile: CandidateProfile) -> None:
        result = score_candidate_tool(profile, llm_client, jd="AI Engineer role")
        assert result["success"] is True
        sc = result["scorecard"]
        assert sc.candidate_name == "Test Candidate"
        assert len(sc.criterion_scores) == 5
        assert 0.0 <= sc.total_score <= 5.0
        assert sc.recommendation in ("INTERVIEW", "HOLD", "REJECT")

    def test_score_includes_evidence(self, llm_client: LLMClient, profile: CandidateProfile) -> None:
        result = score_candidate_tool(profile, llm_client)
        sc = result["scorecard"]
        for cs in sc.criterion_scores:
            assert cs.evidence, f"No evidence for criterion: {cs.criterion}"
            assert cs.justification, f"No justification for criterion: {cs.criterion}"

    def test_score_scale_0_to_5(self, llm_client: LLMClient, profile: CandidateProfile) -> None:
        result = score_candidate_tool(profile, llm_client)
        sc = result["scorecard"]
        for cs in sc.criterion_scores:
            assert 0 <= cs.score <= 5, f"Score {cs.score} out of range for {cs.criterion}"
            assert 0.0 <= cs.weight <= 1.0, f"Weight {cs.weight} out of range"


class TestCheckAvailabilityTool:
    def test_returns_slots(self, llm_client: LLMClient) -> None:
        result = check_availability_tool("Test Candidate", llm_client)
        assert result["success"] is True
        assert len(result["slots"]) > 0

    def test_slots_have_required_fields(self, llm_client: LLMClient) -> None:
        result = check_availability_tool("Test Candidate", llm_client)
        for slot in result["slots"]:
            assert isinstance(slot, InterviewSlot)
            assert slot.date
            assert slot.time
            assert slot.interviewer

    def test_deterministic_slots(self, llm_client: LLMClient) -> None:
        r1 = check_availability_tool("Alice", llm_client)
        # Generate deterministic by re-creating llm client each time
        llm2 = LLMClient(api_key="", model="test-model")
        r2 = check_availability_tool("Alice", llm2)
        assert r1["slots"] == r2["slots"]


class TestProposeInterviewTool:
    def test_proposal_pending_approval(self) -> None:
        slot = InterviewSlot(date="2025-06-15", time="10:00", interviewer="TechVest")
        result = propose_interview_tool("Test Candidate", slot)
        assert result["success"] is True
        assert result["status"] == "pending_approval"
        assert result["confirmation_id"]

    def test_never_executes_automatically(self) -> None:
        slot = InterviewSlot(date="2025-06-15", time="14:00", interviewer="TechVest")
        result = propose_interview_tool("Test Candidate", slot)
        assert result["status"] == "pending_approval"
        assert "AWAITING HUMAN APPROVAL" in result["message"]

    def test_contains_candidate_info(self) -> None:
        slot = InterviewSlot(date="2025-06-15", time="11:00", interviewer="TechVest")
        result = propose_interview_tool("Jane Doe", slot)
        assert result["candidate"] == "Jane Doe"
        assert result["slot"].date == "2025-06-15"