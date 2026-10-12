"""
Entry point for running the recruitment agent as a module.

Usage:
    python -m recruitment_agent          # Run Streamlit UI
    python -m recruitment_agent --cli    # Run CLI mode (headless)
"""

from __future__ import annotations

import sys
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

if "--cli" in sys.argv:
    # Headless CLI mode
    from .graph import RecruitmentGraph
    from .llm_client import LLMClient
    from .trajectory import build_trajectory_report
    from .data import jd as jd_text
    from .data import candidates as candidate_data

    llm = LLMClient()
    graph = RecruitmentGraph(llm_client=llm)

    print("=" * 60)
    print("TechVest Recruitment Agent — CLI Mode")
    print("=" * 60)

    state = graph.run(
        jd=jd_text,
        candidates=candidate_data,
        thread_id="recruitment-cli-1",
    )

    print("\nFinal Report:")
    print(state.get("final_report", "No report generated."))

    print("\nTrajectory Summary:")
    print(build_trajectory_report(state))
else:
    # Default: launch Streamlit UI
    from .ui.streamlit_ui import render_main

    render_main()