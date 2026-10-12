
"""
Streamlit UI for the TechVest Recruitment Agent.

- Sidebar: Upload JD + resumes, scoring rubric, guardrail status
- Tab 1 (Shortlist): Ranked candidates with scorecards, evidence, approval
- Tab 2 (Trajectory): Step-by-step thought/action/observation/decision
- Tab 3 (Guardrails): Step cap, human gate, injection, fairness, audit log

Uploaded files override default sample data. Falls back to sample data when no uploads.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import streamlit as st

# Ensure imports work when run with `streamlit run`
_THIS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from recruitment_agent.graph import RecruitmentGraph
from recruitment_agent.llm_client import LLMClient
from recruitment_agent.state import RecruitmentState
from recruitment_agent.models import GuardrailStatus

logger = logging.getLogger(__name__)

# ── Data sources ────────────────────────────────────────────────────────────

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")


def _load_text(filename: str) -> str:
    """Load text content from a data file."""
    path = os.path.join(DATA_DIR, filename)
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as fh:
            return fh.read()
    return ""


def _load_sample_candidates() -> Dict[str, str]:
    """Load all candidate resumes from data files."""
    return {
        "Priya Sharma": _load_text("priya_resume.txt"),
        "Rahul Verma": _load_text("rahul_resume.txt"),
        "Meera Patel": _load_text("meera_resume.txt"),
    }


SAMPLE_CANDIDATES: Dict[str, str] = _load_sample_candidates()
SAMPLE_JD: str = _load_text("jd.txt") or "TechVest Junior AI Engineer position."


# ── File upload helpers ─────────────────────────────────────────────────────

def _extract_text_from_upload(uploaded_file) -> str:
    """
    Extract text from an uploaded file (PDF, DOCX, or TXT).

    Args:
        uploaded_file: A Streamlit UploadedFile object.

    Returns:
        Extracted text content.
    """
    if uploaded_file is None:
        return ""

    content = uploaded_file.read()
    fname = uploaded_file.name.lower()

    if fname.endswith(".pdf"):
        try:
            import io
            from PyPDF2 import PdfReader
            reader = PdfReader(io.BytesIO(content))
            text = "\n".join(page.extract_text() or "" for page in reader.pages)
            return text.strip()
        except Exception as exc:
            logger.warning("PDF extraction failed: %s", exc)
            return content.decode("utf-8", errors="replace")

    if fname.endswith(".docx"):
        try:
            import io
            from docx import Document
            doc = Document(io.BytesIO(content))
            text = "\n".join(p.text for p in doc.paragraphs)
            return text.strip()
        except Exception as exc:
            logger.warning("DOCX extraction failed: %s", exc)
            return content.decode("utf-8", errors="replace")

    # Plain text
    try:
        return content.decode("utf-8").strip()
    except Exception:
        return content.decode("latin-1").strip()


def _get_candidate_name_from_filename(filename: str) -> str:
    """
    Derive a candidate name from a filename.

    Args:
        filename: e.g. "john_doe_resume.pdf"

    Returns:
        A human-readable name, e.g. "John Doe"
    """
    base = filename.rsplit(".", 1)[0]
    # Remove common suffixes
    for suffix in ["_resume", "-resume", "_cv", "-cv", "_cv_", " resume", " cv"]:
        base = base.replace(suffix, "")
    # Title-case and replace separators
    base = base.replace("_", " ").replace("-", " ")
    return base.strip().title() or filename


# ── Session state management ───────────────────────────────────────────────

def _init_session() -> None:
    """Initialise Streamlit session state variables."""
    for key in ("graph", "state", "run_complete", "approval_processed"):
        if key not in st.session_state:
            st.session_state[key] = None if key != "approval_processed" else False


def _get_graph() -> RecruitmentGraph:
    """Get or create the recruitment graph singleton."""
    if st.session_state.graph is None:
        st.session_state.graph = RecruitmentGraph(llm_client=LLMClient())
    return st.session_state.graph


def _get_uploaded_data() -> Tuple[Optional[str], Dict[str, str]]:
    """
    Read uploaded files from session state with caching.

    Streamlit's file_uploader objects are consumed after the first read(),
    so we cache the extracted text in session state keyed by filename + size.

    Returns:
        (jd_text, candidates_dict) — candidates dict is {name: resume_text}.
    """
    jd_file = st.session_state.get("uploaded_jd")
    resume_files = st.session_state.get("uploaded_resumes", [])

    # Build a cache key from filenames and sizes
    cache_key_parts = []
    if jd_file:
        cache_key_parts.append(f"jd:{jd_file.name}:{jd_file.size}")
    for f in resume_files:
        cache_key_parts.append(f"res:{f.name}:{f.size}")
    cache_key = "|".join(cache_key_parts)

    # Return cached data if uploads haven't changed
    cached = st.session_state.get("_upload_cache", {})
    if cached.get("key") == cache_key:
        return cached["jd_text"], dict(cached["candidates"])

    # Extract fresh — read files only once
    jd_text = _extract_text_from_upload(jd_file) if jd_file else None
    candidates: Dict[str, str] = {}
    for f in resume_files:
        name = _get_candidate_name_from_filename(f.name)
        text = _extract_text_from_upload(f)
        if text:
            if name in candidates:
                name = f"{name} ({f.name})"
            candidates[name] = text

    # Cache the result
    st.session_state["_upload_cache"] = {
        "key": cache_key,
        "jd_text": jd_text,
        "candidates": dict(candidates),
    }

    return jd_text, candidates


# ── Sidebar ────────────────────────────────────────────────────────────────

def _render_sidebar() -> None:
    """Render the sidebar with uploads, config, and guardrail status."""
    with st.sidebar:
        st.title("🤖 TechVest Recruitment Agent")
        st.markdown("---")

        # ── Upload Section ────────────────────────────────────────────────
        st.subheader("📂 Upload Files")
        st.caption("Upload JD and resumes (PDF / DOCX / TXT). Falls back to sample data if empty.")

        uploaded_jd = st.file_uploader(
            "Upload Job Description",
            type=["pdf", "docx", "txt"],
            key="jd_uploader",
            help="Accepted: PDF, DOCX, TXT",
        )
        uploaded_resumes = st.file_uploader(
            "Upload Candidate Resumes",
            type=["pdf", "docx", "txt"],
            accept_multiple_files=True,
            key="resume_uploader",
            help="Upload one or more resumes. Each file becomes a candidate.",
        )

        # Store in session state so run button can access them
        st.session_state["uploaded_jd"] = uploaded_jd
        st.session_state["uploaded_resumes"] = uploaded_resumes

        # Determine active data source
        jd_text, uploaded_candidates = _get_uploaded_data()
        has_uploads = bool(jd_text and uploaded_candidates)
        using_sample = not has_uploads

        st.markdown(
            f"**Source:** {'📤 Uploaded files' if has_uploads else '📁 Sample data'}"
        )
        if has_uploads:
            st.info(f"JD: {uploaded_jd.name} ({len(jd_text)} chars) | {len(uploaded_candidates)} resume(s)")
        else:
            st.info(f"Using sample JD + 3 sample candidates")

        st.markdown("---")

        # ── Run Config ────────────────────────────────────────────────────
        st.subheader("⚙️ Run Config")
        st.info(f"**Framework:** LangGraph + LangChain")
        st.info(f"**Model:** GPT-4o Mini (OpenRouter)")

        # Scoring rubric
        with st.expander("📋 Scoring Rubric", expanded=False):
            st.markdown("""
            **Criteria & Weights:**
            | Criterion | Weight |
            |---|---|
            | Python & ML Fundamentals | **30%** |
            | LLM & RAG Experience | **25%** |
            | Relevant Projects & Tooling | **20%** |
            | Communication & Teamwork | **15%** |
            | Fintech / Domain Fit | **10%** |

            **Thresholds:**
            - 🟢 INTERVIEW: ≥ 3.5 / 5.0
            - 🟡 HOLD: 2.0 – 3.49
            - 🔴 REJECT: < 2.0
            """)

        # Guardrail status
        st.subheader("🛡️ Guardrail Status")
        state = st.session_state.state

        if state and isinstance(state, dict):
            gs: Optional[GuardrailStatus] = state.get("guardrail_status")
            if gs:
                st.markdown(f"""
                - **Human-in-loop:** {'✅ Active' if gs.human_approval_required else '❌ Inactive'}
                - **Recursion Limit:** {gs.recursion_limit} steps max
                - **Iteration Cap:** {gs.iteration_cap} cycles max
                """)
                st.markdown(
                    f"- **Injection:** {'🚨 Blocked' if gs.injection_detected else '✅ No threats'}"
                )
                st.markdown(
                    f"- **Fairness:** {'✅ Passed' if gs.fairness_validated else '⏳ Pending'}"
                )
                traj = state.get("trajectory", [])
                st.markdown(f"- **Audit Log:** ✅ {len(traj)} steps logged")
        else:
            st.markdown("_Run the agent to see guardrail status._")

        # Last run stats
        if st.session_state.run_complete and state:
            st.subheader("📊 Last Run Stats")
            shortlist = state.get("shortlist", [])
            st.markdown(f"""
            - **Total Steps:** {len(state.get('trajectory', []))}
            - **Scored:** {len(state.get('scorecards', {}))}
            - **Shortlisted:** {len([e for e in shortlist if e.get('recommendation') == 'INTERVIEW'])}
            - **Status:** {'✅ Completed' if state.get('status') == 'completed' else '⚠️ ' + str(state.get('status'))}
            """)

        st.markdown("---")
        st.caption("TechVest Recruitment Agent v1.0")


# ── Tab 1: Shortlist ───────────────────────────────────────────────────────

def _render_shortlist(state: RecruitmentState) -> None:
    """Render the ranked candidate shortlist."""
    st.header("🏆 Ranked Candidate Shortlist")

    shortlist: List[Dict[str, Any]] = state.get("shortlist", [])
    if not shortlist:
        st.info("No candidates scored yet. Click **Run Recruitment Agent** to start.")
        return

    sorted_list = sorted(shortlist, key=lambda e: e.get("total_score", 0), reverse=True)

    verdict_emojis = {"INTERVIEW": "🟢", "HOLD": "🟡", "REJECT": "🔴"}
    verdict_colors = {"INTERVIEW": "green", "HOLD": "orange", "REJECT": "red"}

    for i, entry in enumerate(sorted_list):
        candidate: str = entry.get("candidate", "Unknown")
        rec: str = entry.get("recommendation", "REJECT")
        score: float = entry.get("total_score", 0.0)
        emoji = verdict_emojis.get(rec, "⚪")

        with st.container(border=True):
            # Header row
            cols = st.columns([3, 1, 1])
            with cols[0]:
                st.subheader(f"{emoji} #{i + 1} — {candidate}")
            with cols[1]:
                st.markdown(f"### {score:.2f}")
                st.caption("Weighted Score")
            with cols[2]:
                color = verdict_colors.get(rec, "gray")
                st.markdown(f"### :{color}[{rec}]")

            # Justification / evidence summary
            st.markdown(f"**Resume Evidence:** {entry.get('justification', 'N/A')}")

            # Per-criterion scores (expandable)
            sc_data = entry.get("scorecard", {})
            criterion_scores = sc_data.get("criterion_scores", [])
            with st.expander(f"📊 Per-Criterion Scores for {candidate}", expanded=False):
                for cs in criterion_scores:
                    st.markdown(f"""
                    **{cs.get('criterion', '?')}** — {cs.get('score', 0)}/5 (weight: {cs.get('weight', 0):.2f})
                    - *Evidence:* {cs.get('evidence', 'N/A')}
                    - *Justification:* {cs.get('justification', 'N/A')}
                    """)

            # Interview proposal status
            interview_confirmations = state.get("interview_confirmations", {})
            conf = interview_confirmations.get(candidate, {})
            if isinstance(conf, dict):
                status = conf.get("status", "")
                slot = conf.get("slot", {})
                if status == "pending_approval":
                    st.warning(
                        f"⏳ Interview proposed: {slot.get('date', '?')} "
                        f"at {slot.get('time', '?')} — **Awaiting your approval**"
                    )
                elif status == "confirmed":
                    st.success(
                        f"✅ Interview confirmed: {slot.get('date', '?')} "
                        f"at {slot.get('time', '?')}"
                    )
                elif status == "scheduled":
                    st.success(
                        f"✅ Interview scheduled: {slot.get('date', '?')} "
                        f"at {slot.get('time', '?')}"
                    )
                elif status == "rejected":
                    st.error(
                        f"❌ Interview rejected: {slot.get('date', '?')} "
                        f"at {slot.get('time', '?')}"
                    )

    # ── Human Approval Panel (kept exactly as before) ─────────────────────
    if state.get("status") == "awaiting_approval":
        pending = state.get("pending_action")
        if pending:
            st.markdown("---")
            st.subheader("🛑 Human Approval Required")
            pa_candidate = pending.get("candidate", "Unknown")
            pa_slot = pending.get("slot", {})
            pa_id = pending.get("confirmation_id", "")

            st.warning(
                f"**Interview Proposal:** {pa_candidate} on "
                f"{pa_slot.get('date', '?')} at {pa_slot.get('time', '?')} "
                f"(ID: {pa_id})"
            )

            col_a, col_b = st.columns(2)
            with col_a:
                if st.button("✅ Approve Interview", type="primary", use_container_width=True):
                    graph = _get_graph()
                    state = graph.approve_interview(state, approved=True)
                    st.session_state.state = state
                    st.session_state.approval_processed = True
                    st.rerun()
            with col_b:
                if st.button("❌ Reject Interview", use_container_width=True):
                    graph = _get_graph()
                    state = graph.approve_interview(state, approved=False)
                    st.session_state.state = state
                    st.session_state.approval_processed = True
                    st.rerun()


# ── Tab 2: Trajectory ──────────────────────────────────────────────────────

def _render_trajectory(state: RecruitmentState) -> None:
    """Render the step-by-step trajectory with workflow stage, node, thought, action, observation, state updated, decision, and next node."""
    st.header("🔍 Agent Trajectory")

    trajectory: List[Dict[str, Any]] = state.get("trajectory", [])
    if not trajectory:
        st.info("No trajectory data available. Run the agent first.")
        return

    st.subheader("Execution Trace Through the LangGraph Workflow")
    st.caption(
        "Each step shows the workflow stage, current LangGraph node, the agent's thought, "
        "the tool called, what it observed, which state was updated, the decision made, "
        "and the next node to execute."
    )

    for i, step in enumerate(trajectory):
        if isinstance(step, dict):
            stage = step.get("stage", "Processing")
            node_name = step.get("node", "Planner")
            tool_name = step.get("tool", "?")
            next_node = step.get("next_node", "Planner")
            state_updated = step.get("state_updated", "")
            is_last = (i == len(trajectory) - 1)
            is_end = next_node == "END" or is_last

            with st.container(border=True):
                # Header with stage badge and step number
                stage_colors = {
                    "Resume Analysis": "blue", "Candidate Evaluation": "green",
                    "Shortlisting": "orange", "Scheduling": "red",
                    "Human Approval": "purple", "Finished": "gray", "Processing": "gray",
                }
                sc = stage_colors.get(stage, "gray")
                st.markdown(f"**Step {i + 1}** — :{sc}[{stage}]  |  Node: **{node_name}**")

                if is_end:
                    st.success("🚀 **Workflow Complete — Reached END node**" if next_node == "END" else "")

                col1, col2 = st.columns([1, 5])
                with col1:
                    st.caption(f"🕐 {step.get('timestamp', '').split('T')[0]}")
                    st.markdown(f"**⬇️ Next Node:** `{next_node}`")
                with col2:
                    st.markdown(f"**💭 Thought:** {step.get('thought', '')}")
                    st.markdown(f"**⚡ Tool Called:** `{tool_name}`")
                    if step.get("arguments"):
                        args = step["arguments"]
                        if isinstance(args, dict) and len(args) <= 3:
                            st.markdown(f"**📥 Arguments:** {args}")
                        else:
                            with st.expander("📥 Tool Arguments", expanded=False):
                                st.code(json.dumps(args, indent=2), language="json")
                    st.markdown(f"**👁️ Observation:** {step.get('observation', '')}")
                    if state_updated:
                        st.markdown(f"**📝 State Updated:** `{state_updated}`")
                    st.markdown(f"**➡️ Decision:** {step.get('decision', '')}")
                    if next_node:
                        st.markdown(f"**➡️ Next Node / Stage:** `{next_node}`")

            if is_end:
                st.markdown("---")
                st.success("🏁 **Workflow Complete** — Reached END node")


# ── Tab 3: Guardrails ──────────────────────────────────────────────────────

def _render_guardrails(state: RecruitmentState) -> None:
    """Render the guardrail results panel."""
    st.header("🛡️ Guardrails & Audit Log")

    trajectory: List[Dict[str, Any]] = state.get("trajectory", [])
    gs = state.get("guardrail_status")

    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Guardrail Results")

        # 1. Step Cap
        steps_used = len(trajectory)
        steps_max = gs.recursion_limit if gs else 50
        step_ok = steps_used < steps_max
        st.markdown(f"""
        **1. Iteration / Step Cap**
        - Steps used: {steps_used}
        - Maximum: {steps_max}
        - Status: {'✅ PASSED' if step_ok else '❌ EXCEEDED'}
        """)

        # 2. Human Gate
        human_active = gs.human_approval_required if gs else True
        pending = state.get("pending_action")
        st.markdown(f"""
        **2. Human-in-the-Loop Gate**
        - Status: {'✅ Active' if human_active else '❌ Inactive'}
        - Pending action: {'✅ Yes' if pending else '❌ No'}
        - Scheduling: {'⛔ NEVER automatic' if human_active else '⚠️ Manual override'}
        """)

        # 3. Prompt Injection
        if gs and gs.injection_detected:
            st.error("🚨 **3. Prompt Injection — DETECTED & BLOCKED**")
            st.markdown("""
            Meera Patel's resume contained manipulative instructions
            attempting to override the scoring rubric. The guardrail
            detected and blocked the attack. The candidate was scored
            based on factual JD-relevant skills only.
            """)
        else:
            st.markdown("""
            **3. Prompt Injection Defence**
            - Status: ✅ No threats detected
            - All resumes scanned for manipulative instructions
            """)

        # 4. Fairness
        if gs and gs.fairness_validated:
            st.markdown(f"""
            **4. Fairness Validation**
            - Status: ✅ PASSED
            - Note: {gs.fairness_note}
            """)
        else:
            st.markdown("""
            **4. Fairness Validation**
            - Status: ⏳ Pending or not applicable
            """)

    with col_right:
        st.subheader("Decision Audit Log")

        if not trajectory:
            st.info("No steps logged yet.")
        else:
            audit_data = []
            for i, step in enumerate(trajectory):
                if isinstance(step, dict):
                    audit_data.append({
                        "Step": i + 1,
                        "Tool": step.get("tool", ""),
                        "Decision": step.get("decision", ""),
                        "Timestamp": step.get("timestamp", ""),
                    })

            st.dataframe(
                audit_data,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Step": st.column_config.NumberColumn(format="%d"),
                    "Tool": "Tool Called",
                    "Decision": "Decision Made",
                    "Timestamp": st.column_config.TextColumn(width="medium"),
                },
            )

            st.caption(f"Total: {len(audit_data)} steps logged")

    # Final report
    st.divider()
    st.subheader("📄 Final Report")
    final_report = state.get("final_report")
    if final_report:
        st.text_area("Final Summary Report", final_report, height=400)
    else:
        st.info("No final report generated yet.")


# ── Run button and page layout ─────────────────────────────────────────────

def _render_run_button() -> None:
    """Render the run button and handle execution."""
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🚀 Run Recruitment Agent", type="primary", use_container_width=True):
            with st.spinner("Running recruitment agent... This may take a moment."):
                try:
                    # Determine data source: uploaded files or sample data
                    jd_text, uploaded_candidates = _get_uploaded_data()
                    has_uploads = bool(jd_text and uploaded_candidates)

                    if has_uploads:
                        jd = jd_text
                        candidates = uploaded_candidates
                    else:
                        jd = SAMPLE_JD
                        candidates = SAMPLE_CANDIDATES

                    if not jd:
                        st.error("No job description available. Upload a JD or check sample data.")
                        return
                    if not candidates:
                        st.error("No candidates available. Upload resumes or check sample data.")
                        return

                    graph = _get_graph()
                    state = graph.run(
                        jd=jd,
                        candidates=candidates,
                        thread_id="recruitment-streamlit-1",
                    )
                    st.session_state.state = state
                    st.session_state.run_complete = True
                    st.success("✅ Recruitment agent completed successfully!")
                    st.rerun()
                except Exception as exc:
                    st.error(f"❌ Error running agent: {str(exc)}")
                    logger.exception("Agent run failed")


def render_main() -> None:
    """
    Main entry point for the Streamlit UI.

    Call this from `streamlit run`.
    """
    st.set_page_config(
        page_title="TechVest Recruitment Agent",
        page_icon="🤖",
        layout="wide",
    )

    _init_session()
    _render_sidebar()

    # Title
    st.title("TechVest Recruitment Agent")
    st.caption("Autonomous AI-powered hiring assistant with guardrails (LangGraph)")

    # Tabs
    tab1, tab2, tab3 = st.tabs(["🏆 Shortlist", "🔍 Trajectory", "🛡️ Guardrails"])

    # Run button
    _render_run_button()

    # Content
    state = st.session_state.state
    if state and st.session_state.run_complete:
        with tab1:
            _render_shortlist(state)
        with tab2:
            _render_trajectory(state)
        with tab3:
            _render_guardrails(state)
    else:
        msg = "👈 Click **Run Recruitment Agent** to start the autonomous hiring process."
        with tab1:
            st.info(msg)
        with tab2:
            st.info("Trajectory data will appear here after the agent runs.")
        with tab3:
            st.info("Guardrail results will appear here after the agent runs.")


if __name__ == "__main__":
    render_main()