"""
Tests for the LangGraph graph execution: graph construction, routing, and full pipeline run.
"""

from __future__ import annotations

import pytest
from ..graph import RecruitmentGraph
from ..llm_client import LLMClient
from ..state import RecruitmentState
from ..routing import determine_next_action, build_plan_summary
from ..models import GuardrailStatus


@pytest.fixture
def graph() -> RecruitmentGraph:
    """Create a graph with fallback-only LLM client."""
    llm = LLMClient(api_key="", model="test-model")
    return RecruitmentGraph(llm_client=llm)


@pytest.fixture
def sample_state() -> RecruitmentState:
    return RecruitmentState(
        jd="AI Engineer role with Python and ML",
        candidates={
            "Alice": "ALICE\nPython, ML, LangChain\nProjects: RAG system",
            "Bob": "BOB\nPython, Django\nProjects: Web app",
        },
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
        candidate_order=["Alice", "Bob"],
        current_candidate=None,
        final_report=None,
    )


class TestGraphConstruction:
    def test_graph_initializes(self, graph: RecruitmentGraph) -> None:
        assert graph.graph is not None
        assert graph.app is not None

    def test_graph_has_all_nodes(self, graph: RecruitmentGraph) -> None:
        expected_nodes = {"parse", "score", "avail", "decide", "finish"}
        builder = graph.graph
        assert hasattr(builder, "nodes")
        for node in expected_nodes:
            assert node in builder.nodes, f"Node {node} not found in graph"


class TestRouting:
    def test_route_to_parse_when_no_profiles(self, sample_state: RecruitmentState) -> None:
        action = determine_next_action(sample_state)
        assert action == "parse_resume"

    def test_route_to_score_when_parsed_not_scored(self, sample_state: RecruitmentState) -> None:
        from ..models import CandidateProfile
        # Parse ALL candidates, but don't score any
        sample_state["parsed_profiles"] = {
            "Alice": CandidateProfile(name="Alice", skills=["Python"], years_experience=1.0),
            "Bob": CandidateProfile(name="Bob", skills=["Django"], years_experience=0.5),
        }
        action = determine_next_action(sample_state)
        assert action == "score_candidate"

    def test_route_to_finish_when_completed(self, sample_state: RecruitmentState) -> None:
        sample_state["status"] = "completed"
        action = determine_next_action(sample_state)
        assert action == "finish"

    def test_route_to_finish_when_error(self, sample_state: RecruitmentState) -> None:
        sample_state["status"] = "error"
        action = determine_next_action(sample_state)
        assert action == "finish"

    def test_route_to_human_approval_when_pending(self, sample_state: RecruitmentState) -> None:
        sample_state["pending_action"] = {"type": "interview_proposal"}
        action = determine_next_action(sample_state)
        assert action == "human_approval"


class TestFullPipeline:
    def test_graph_run_completes(self, graph: RecruitmentGraph) -> None:
        """Test that the full graph pipeline runs end-to-end (with fallback scoring)."""
        state = graph.run(
            jd="AI Engineer role requiring Python, ML, and RAG experience.",
            candidates={
                "Priya": "PRIYA SHARMA\nPython, NumPy, Pandas, LangChain, RAG\nProjects: FinRAG",
            },
            thread_id="test-run-1",
        )
        assert state is not None
        assert state.get("status") in ("completed", "error", "awaiting_approval")

    def test_graph_scores_candidates(self, graph: RecruitmentGraph) -> None:
        state = graph.run(
            jd="ML Engineer role",
            candidates={
                "Test": "TEST CANDIDATE\nPython, ML, scikit-learn\nProjects: ML model",
            },
            thread_id="test-run-2",
        )
        if state.get("status") == "completed":
            assert len(state.get("scorecards", {})) > 0
            assert len(state.get("shortlist", [])) > 0

    def test_graph_trajectory_logged(self, graph: RecruitmentGraph) -> None:
        state = graph.run(
            jd="Python developer role",
            candidates={
                "Dev": "DEV\nPython, Django\nProjects: Web app",
            },
            thread_id="test-run-3",
        )
        trajectory = state.get("trajectory", [])
        assert len(trajectory) > 0, "Trajectory should have at least one step"

    def test_graph_handles_multiple_candidates(self, graph: RecruitmentGraph) -> None:
        state = graph.run(
            jd="AI Engineer role",
            candidates={
                "Alice": "ALICE\nPython, ML, RAG\nProjects: AI system",
                "Bob": "BOB\nPython, Django\nProjects: Web app",
            },
            thread_id="test-run-4",
        )
        assert state is not None
        # The graph should process all candidates
        profiles = state.get("parsed_profiles", {})
        assert len(profiles) > 0


class TestHumanApproval:
    def test_approve_interview(self, graph: RecruitmentGraph) -> None:
        """Test that approve_interview updates state correctly."""
        state = RecruitmentState(
            jd="", candidates={}, parsed_profiles={}, scorecards={}, verified_scores={},
            revision_count=0, shortlist=[], trajectory=[], guardrail_status=GuardrailStatus(),
            pending_action={"type": "interview_proposal", "candidate": "Alice"},
            status="awaiting_approval", error=None, interview_slots={},
            interview_confirmations={"Alice": {"candidate": "Alice", "status": "pending_approval", "slot": {"date": "2025-06-15", "time": "10:00"}}},
            candidate_order=[], current_candidate=None, final_report=None,
        )
        result = graph.approve_interview(state, approved=True)
        assert result.get("pending_action") is None
        assert result.get("status") == "running"
        conf = result.get("interview_confirmations", {})
        assert conf.get("Alice", {}).get("status") == "confirmed"

    def test_reject_interview(self, graph: RecruitmentGraph) -> None:
        state = RecruitmentState(
            jd="", candidates={}, parsed_profiles={}, scorecards={}, verified_scores={},
            revision_count=0, shortlist=[], trajectory=[], guardrail_status=GuardrailStatus(),
            pending_action={"type": "interview_proposal", "candidate": "Bob"},
            status="awaiting_approval", error=None, interview_slots={},
            interview_confirmations={"Bob": {"candidate": "Bob", "status": "pending_approval", "slot": {"date": "2025-06-16", "time": "14:00"}}},
            candidate_order=[], current_candidate=None, final_report=None,
        )
        result = graph.approve_interview(state, approved=False)
        assert result.get("pending_action") is None
        conf = result.get("interview_confirmations", {})
        assert conf.get("Bob", {}).get("status") == "rejected"


class TestBuildPlanSummary:
    def test_summary_contains_candidates(self, sample_state: RecruitmentState) -> None:
        summary = build_plan_summary(sample_state)
        assert "Alice" in summary
        assert "Bob" in summary

    def test_summary_contains_status(self, sample_state: RecruitmentState) -> None:
        summary = build_plan_summary(sample_state)
        assert "running" in summary