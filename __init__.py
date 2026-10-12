"""
TechVest Recruitment Agent — Autonomous AI-powered hiring assistant.

Built with LangGraph, LangChain, Pydantic v2, and Streamlit.
Uses OpenRouter API (GPT-4o Mini) for LLM-powered resume parsing,
candidate scoring, and decision-making.

Includes five guardrails:
1. Human approval required before scheduling.
2. Recursion limit (50 steps max).
3. Iteration cap (25 cycles max).
4. Prompt injection detection and blocking.
5. Fairness validation (no name/gender/age/college bias).
"""

from __future__ import annotations

__version__ = "1.0.0"
__author__ = "TechVest AI Team"