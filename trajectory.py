"""
Trajectory logging for the TechVest Recruitment Agent.

Logs every reasoning step with stage, node, thought, tool, arguments,
observation, state_updated, decision, next_node, and timestamp.
All steps are stored inside the state for full auditability.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from .models import TrajectoryStep
from .state import RecruitmentState

logger = logging.getLogger(__name__)


def log_trajectory_step(
    state: RecruitmentState,
    thought: str,
    tool: str,
    arguments: Dict[str, Any],
    observation: str,
    decision: str,
    stage: str = "Processing",
    node: str = "Planner",
    state_updated: str = "",
    next_node: str = "Planner",
) -> List[Dict[str, Any]]:
    """
    Log a single trajectory step and return the updated trajectory list.

    Enhanced with stage, node, state_updated, and next_node fields so the
    trace reads like an execution graph from START to END.

    Args:
        state: Current recruitment state (used for context, not modified).
        thought: What the agent decided to do.
        tool: Which tool was called.
        arguments: Tool arguments as a dictionary.
        observation: What the tool returned.
        decision: Next decision or action based on the observation.
        stage: Current workflow stage (Resume Analysis, Candidate Evaluation,
               Verification, Shortlisting, Scheduling, Human Approval, Finished).
        node: Current LangGraph node or agent (Resume Analyst, Scorer,
              Verifier, Coordinator, Scheduler, Human Approval Gate, Finalizer).
        state_updated: State keys that were modified (e.g. parsed_profiles,
                       scorecards, shortlist).
        next_node: Next LangGraph node to execute (e.g. Scorer, Scheduler,
                   Human Approval, END).

    Returns:
        Updated trajectory list (append this to your node's return dict).
    """
    step = TrajectoryStep(
        stage=stage,
        node=node,
        thought=thought,
        tool=tool,
        arguments=arguments,
        observation=str(observation)[:1000],
        state_updated=state_updated,
        decision=decision,
        next_node=next_node,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )

    trajectory: List[Dict[str, Any]] = list(state.get("trajectory", []))
    trajectory.append(step.model_dump())

    logger.debug(
        "Trajectory step [%s]: %s | %s | %s -> %s -> %s",
        step.timestamp,
        step.stage,
        step.node,
        thought[:60],
        tool,
        next_node,
    )

    return trajectory


def build_trajectory_report(state: RecruitmentState) -> str:
    """
    Build a human-readable report from the full trajectory log.

    Args:
        state: The recruitment state with trajectory data.

    Returns:
        A formatted string showing the full reasoning trace.
    """
    trajectory = state.get("trajectory", [])
    if not trajectory:
        return "No trajectory steps recorded."

    lines: List[str] = []
    lines.append("=" * 80)
    lines.append("RECRUITMENT AGENT — FULL TRAJECTORY LOG")
    lines.append("=" * 80)
    lines.append(f"Total Steps: {len(trajectory)}")
    lines.append("")

    for i, step in enumerate(trajectory):
        if isinstance(step, dict):
            lines.append("─" * 80)
            lines.append(f"Step {i + 1}")
            lines.append(f"  Timestamp    : {step.get('timestamp', '')}")
            lines.append(f"  Stage        : {step.get('stage', 'Processing')}")
            lines.append(f"  Node         : {step.get('node', 'Planner')}")
            lines.append(f"  Thought      : {step.get('thought', '')}")
            lines.append(f"  Tool         : {step.get('tool', '')}")
            lines.append(f"  Arguments    : {step.get('arguments', {})}")
            lines.append(f"  Observation  : {str(step.get('observation', ''))[:300]}")
            lines.append(f"  State Updated: {step.get('state_updated', '')}")
            lines.append(f"  Decision     : {step.get('decision', '')}")
            lines.append(f"  Next Node    : {step.get('next_node', 'Planner')}")
            lines.append("")
        else:
            lines.append("─" * 80)
            lines.append(f"Step {i + 1}")
            lines.append(f"  Timestamp    : {step.timestamp}")
            lines.append(f"  Stage        : {step.stage}")
            lines.append(f"  Node         : {step.node}")
            lines.append(f"  Thought      : {step.thought}")
            lines.append(f"  Tool         : {step.tool}")
            lines.append(f"  Arguments    : {step.arguments}")
            lines.append(f"  Observation  : {step.observation[:300]}")
            lines.append(f"  State Updated: {step.state_updated}")
            lines.append(f"  Decision     : {step.decision}")
            lines.append(f"  Next Node    : {step.next_node}")
            lines.append("")

    lines.append("=" * 80)
    lines.append("END OF TRAJECTORY LOG — Workflow Complete — Reached END node")
    lines.append("=" * 80)

    return "\n".join(lines)


def summarize_trajectory(state: RecruitmentState) -> str:
    """
    Create a brief summary of the trajectory for display purposes.

    Args:
        state: The recruitment state with trajectory data.

    Returns:
        A short summary string.
    """
    trajectory = state.get("trajectory", [])
    if not trajectory:
        return "No steps recorded."

    tool_counts: Dict[str, int] = {}
    for step in trajectory:
        if isinstance(step, dict):
            tool_counts[step.get("tool", "unknown")] = tool_counts.get(step.get("tool", "unknown"), 0) + 1
        else:
            tool_counts[step.tool] = tool_counts.get(step.tool, 0) + 1

    summary_parts = [f"{len(trajectory)} steps total"]
    for tool, count in sorted(tool_counts.items()):
        summary_parts.append(f"{tool}: {count}x")

    return " | ".join(summary_parts)