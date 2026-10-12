"""
LangGraph graph builder for the TechVest Recruitment Agent.

Defines the state graph with nodes and conditional edges.
Uses MemorySaver for checkpointing and persistence.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Literal, Optional

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from .state import RecruitmentState, create_initial_state
from .llm_client import LLMClient
from .routing import determine_next_action, RouteName

logger = logging.getLogger(__name__)


class RecruitmentGraph:
    """The LangGraph-based recruitment agent graph."""

    def __init__(self, llm_client: LLMClient) -> None:
        self.llm_client: LLMClient = llm_client
        self.checkpointer: MemorySaver = MemorySaver()
        self.graph = self._build_graph()
        self.app = self.graph.compile(checkpointer=self.checkpointer)

    # ── Node implementations ──────────────────────────────────────────────

    def _resume_parser_node(self, state: RecruitmentState) -> Dict[str, Any]:
        from .trajectory import log_trajectory_step
        for name in state.get("candidates", {}):
            if name not in state.get("parsed_profiles", {}):
                from .guardrails import check_prompt_injection, update_guardrail_status
                from .models import GuardrailStatus
                from .tools import parse_resume_tool

                text = state["candidates"][name]
                injection = check_prompt_injection(text, self.llm_client)
                gs = state.get("guardrail_status", GuardrailStatus())
                if injection:
                    gs = update_guardrail_status(gs, injection_detected=True, injection_blocked=True)

                result = parse_resume_tool(text, self.llm_client)
                if result["success"]:
                    profiles = dict(state.get("parsed_profiles", {}))
                    profiles[name] = result["profile"]
                    traj = log_trajectory_step(
                        state, thought=f"Parse resume for {name}", tool="parse_resume",
                        arguments={"candidate": name},
                        observation=f"Parsed: {result['profile'].name}",
                        decision="Continue",
                        stage="Resume Analysis",
                        node="Resume Analyst",
                        state_updated=f"parsed_profiles[{name}]",
                        next_node="Scorer" if len(state.get("parsed_profiles", {})) + 1 >= len(state.get("candidates", {})) else "Resume Analyst",
                    )
                    return {"parsed_profiles": profiles, "guardrail_status": gs,
                            "current_candidate": name, "trajectory": traj}
                return {"error": result["error"]}
        return {"error": "No candidates to parse"}

    def _scorer_node(self, state: RecruitmentState) -> Dict[str, Any]:
        from .trajectory import log_trajectory_step
        for name in state.get("parsed_profiles", {}):
            if name not in state.get("scorecards", {}):
                from .tools import score_candidate_tool
                profile = state["parsed_profiles"][name]
                result = score_candidate_tool(profile, self.llm_client, jd=state.get("jd", ""))
                if result["success"]:
                    sc = result["scorecard"]
                    scorecards = dict(state.get("scorecards", {}))
                    scorecards[name] = sc
                    shortlist = list(state.get("shortlist", []))
                    shortlist.append({
                        "candidate": name, "scorecard": sc.model_dump(),
                        "total_score": sc.total_score, "recommendation": sc.recommendation,
                        "justification": sc.justification,
                    })
                    traj = log_trajectory_step(
                        state, thought=f"Score {name} against rubric", tool="score_candidate",
                        arguments={"candidate": name},
                        observation=f"Score: {sc.total_score:.2f} → {sc.recommendation}",
                        decision=sc.recommendation,
                        stage="Candidate Evaluation",
                        node="Scorer",
                        state_updated=f"scorecards[{name}], shortlist",
                        next_node="Coordinator",
                    )
                    return {"scorecards": scorecards, "shortlist": shortlist, "trajectory": traj}
                return {"error": result["error"]}
        return {"error": "No candidates to score"}

    def _availability_node(self, state: RecruitmentState) -> Dict[str, Any]:
        from .trajectory import log_trajectory_step
        for entry in state.get("shortlist", []):
            if entry["recommendation"] == "INTERVIEW" and entry["candidate"] not in state.get("interview_slots", {}):
                from .tools import check_availability_tool
                name = entry["candidate"]
                result = check_availability_tool(name, self.llm_client)
                if result["success"]:
                    interview_slots = dict(state.get("interview_slots", {}))
                    interview_slots[name] = result["slots"]
                    traj = log_trajectory_step(
                        state, thought=f"Check availability for {name}", tool="check_availability",
                        arguments={"candidate": name},
                        observation=f"Found {len(result['slots'])} slots",
                        decision="Proceed",
                        stage="Shortlisting",
                        node="Coordinator",
                        state_updated=f"interview_slots[{name}]",
                        next_node="Human Approval",
                    )
                    return {"interview_slots": interview_slots, "trajectory": traj}
                return {"error": result["error"]}
        return {"status": "completed"}

    def _decision_maker_node(self, state: RecruitmentState) -> Dict[str, Any]:
        from .trajectory import log_trajectory_step
        for entry in state.get("shortlist", []):
            if entry["recommendation"] == "INTERVIEW":
                name = entry["candidate"]
                slots = state.get("interview_slots", {}).get(name, [])
                if slots and name not in state.get("interview_confirmations", {}):
                    from .tools import propose_interview_tool
                    proposal = propose_interview_tool(name, slots[0])
                    if proposal["success"]:
                        confirmations = dict(state.get("interview_confirmations", {}))
                        confirmations[name] = {
                            "candidate": name, "slot": slots[0].model_dump(),
                            "status": "pending_approval", "confirmation_id": proposal["confirmation_id"],
                        }
                        traj = log_trajectory_step(
                            state, thought=f"Propose interview for {name}", tool="propose_interview",
                            arguments={"candidate": name, "slot": slots[0].model_dump()},
                            observation=proposal["message"],
                            decision="Awaiting human approval",
                            stage="Scheduling",
                            node="Scheduler",
                            state_updated=f"interview_confirmations[{name}], pending_action",
                            next_node="Human Approval",
                        )
                        return {
                            "pending_action": {"type": "interview_proposal", "candidate": name,
                                               "slot": slots[0].model_dump(), "confirmation_id": proposal["confirmation_id"]},
                            "interview_confirmations": confirmations,
                            "status": "awaiting_approval", "trajectory": traj,
                        }
        return {"status": "completed"}

    def _finish_node(self, state: RecruitmentState) -> Dict[str, Any]:
        from .guardrails import validate_fairness, update_guardrail_status
        from .models import GuardrailStatus
        from .trajectory import log_trajectory_step
        from datetime import datetime, timezone

        fairness = validate_fairness(state, self.llm_client)
        gs = state.get("guardrail_status", GuardrailStatus())
        gs = update_guardrail_status(gs, fairness_result=fairness)

        shortlist = state.get("shortlist", [])
        sorted_list = sorted(shortlist, key=lambda e: e.get("total_score", 0), reverse=True)
        confirmations = state.get("interview_confirmations", {})
        traj = log_trajectory_step(state, thought="Generate final report and fairness validation", tool="finalize",
                                   arguments={}, observation="Complete", decision="Finished",
                                   stage="Finished", node="Finalizer",
                                   state_updated="final_report, guardrail_status, verified_scores",
                                   next_node="END")

        lines = [
            "=" * 60, "TECHVEST RECRUITMENT AGENT — FINAL REPORT", "=" * 60,
            f"Generated: {datetime.now(timezone.utc).isoformat()}", "",
            "RANKED CANDIDATES:",
        ]
        for i, e in enumerate(sorted_list):
            lines.append(f"\n  #{i+1}: {e['candidate']} — {e['total_score']:.2f}/5.0 → {e['recommendation']}")
        lines.extend(["\nINTERVIEW STATUS:"])
        for e in sorted_list:
            c = confirmations.get(e['candidate'], {})
            s = c.get("status", "not_proposed") if isinstance(c, dict) else "not_proposed"
            lines.append(f"  {e['candidate']}: {s}")
        lines.extend(["\nGUARDRAILS:", f"  Injection: {'Blocked' if gs.injection_detected else 'No threats'}",
                      f"  Fairness: {'Passed' if gs.fairness_validated else 'Not run'}",
                      f"  Audit: {len(state.get('trajectory', [])) + 1} steps"])

        return {"status": "completed", "final_report": "\n".join(lines),
                "guardrail_status": gs, "trajectory": traj}

    # ── Router ────────────────────────────────────────────────────────────

    def _router(self, state: RecruitmentState) -> Literal["parse", "score", "avail", "decide", "finish", "__end__"]:
        status = state.get("status", "running")
        if status in ("completed", "error", "awaiting_approval"):
            return "__end__"

        action = determine_next_action(state)
        mapping: Dict[RouteName, str] = {
            "parse_resume": "parse", "score_candidate": "score",
            "check_availability": "avail", "human_approval": "decide",
            "schedule_interview": "decide", "finish": "finish", "continue": "parse",
        }
        return mapping.get(action, "finish")

    # ── Graph builder ────────────────────────────────────────────────────

    def _build_graph(self) -> StateGraph:
        builder = StateGraph(RecruitmentState)
        builder.add_node("parse", self._resume_parser_node)
        builder.add_node("score", self._scorer_node)
        builder.add_node("avail", self._availability_node)
        builder.add_node("decide", self._decision_maker_node)
        builder.add_node("finish", self._finish_node)
        builder.set_entry_point("parse")

        targets = {"parse": "parse", "score": "score", "avail": "avail",
                   "decide": "decide", "finish": "finish", "__end__": "__end__"}
        for node in ("parse", "score", "avail", "decide", "finish"):
            builder.add_conditional_edges(node, self._router, targets)

        return builder

    # ── Execution ─────────────────────────────────────────────────────────

    def run(self, jd: str, candidates: Dict[str, str],
            candidate_order: Optional[List[str]] = None,
            thread_id: str = "recruitment-1") -> RecruitmentState:
        state = create_initial_state(jd=jd, candidates=candidates, candidate_order=candidate_order)
        config = {"configurable": {"thread_id": thread_id}}
        try:
            return self.app.invoke(state, config)
        except Exception as exc:
            logger.error("Graph execution failed: %s", exc)
            state["status"] = "error"
            state["error"] = str(exc)
            return state

    def approve_interview(self, state: RecruitmentState, approved: bool = True) -> RecruitmentState:
        pending = state.get("pending_action")
        if not pending:
            return state
        name = pending.get("candidate", "")
        confs = dict(state.get("interview_confirmations", {}))
        if name in confs and isinstance(confs[name], dict):
            confs[name]["status"] = "confirmed" if approved else "rejected"
            state["interview_confirmations"] = confs
        state["pending_action"] = None
        state["status"] = "running"
        return state