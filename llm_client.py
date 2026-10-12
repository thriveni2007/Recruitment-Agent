"""
LLM client for the Recruitment Agent using OpenRouter.

Provides structured output extraction for resume parsing, candidate scoring,
injection detection, and fairness validation. Includes deterministic fallback
when the API is unavailable.
"""

from __future__ import annotations

import json
import logging
import os
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Type

from openai import OpenAI
from pydantic import BaseModel

from .models import CandidateProfile, ScoreCard, CriterionScore, InterviewSlot
from .prompts import (
    SYSTEM_PROMPT_PARSE_RESUME,
    USER_PROMPT_PARSE_RESUME,
    SYSTEM_PROMPT_SCORE_CANDIDATE,
    USER_PROMPT_SCORE_CANDIDATE,
    SYSTEM_PROMPT_INJECTION_CHECK,
    SYSTEM_PROMPT_FAIRNESS_CHECK,
    SYSTEM_PROMPT_PLANNER,
    DEFAULT_RUBRIC,
)

logger = logging.getLogger(__name__)


def _try_load_dotenv() -> None:
    """Attempt to load .env files from standard locations."""
    try:
        from dotenv import load_dotenv

        search_paths = [
            os.path.join(os.path.dirname(__file__), ".env"),
            os.path.join(os.path.dirname(__file__), "..", ".env"),
            ".env",
        ]
        for path in search_paths:
            if os.path.isfile(path):
                load_dotenv(path)
                logger.debug("Loaded .env from %s", path)
                return
    except ImportError:
        logger.debug("python-dotenv not available, skipping .env load")


class LLMClient:
    """
    Client for interacting with LLMs via OpenRouter.

    Handles prompt construction, API calls, structured JSON extraction,
    and graceful fallback to deterministic logic.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "openai/gpt-4o-mini",
    ) -> None:
        _try_load_dotenv()

        self.api_key: str = api_key or os.getenv("OPENROUTER_API_KEY", "")
        self.model: str = model or os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini")
        self.base_url: str = "https://openrouter.ai/api/v1"
        self._api_available: bool = bool(self.api_key)

        if self.api_key:
            self.client: Optional[OpenAI] = OpenAI(
                base_url=self.base_url,
                api_key=self.api_key,
                default_headers={
                    "HTTP-Referer": "https://techvest-recruitment-agent.local",
                    "X-Title": "TechVest Recruitment Agent",
                },
            )
        else:
            self.client = None
            logger.warning("No OpenRouter API key found — using deterministic fallback logic")

    # ── Core LLM call ──────────────────────────────────────────────────────

    def _call_llm(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.1,
        max_tokens: int = 1500,
    ) -> str:
        """
        Make a raw LLM call via OpenRouter.

        Args:
            system_prompt: System-level instruction prompt.
            user_prompt: User message content.
            temperature: Sampling temperature (0.0–1.0).
            max_tokens: Maximum tokens in the response.

        Returns:
            The raw text response from the LLM.

        Raises:
            RuntimeError: If the API is unavailable or the call fails.
        """
        if not self.client or not self._api_available:
            raise RuntimeError("API not available")

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=min(max_tokens, 2000),
            )
            content: str = response.choices[0].message.content or ""
            return content
        except Exception as exc:
            error_str = str(exc)
            if "402" in error_str or "credits" in error_str.lower() or "quota" in error_str.lower():
                self._api_available = False
                logger.warning("API credits exhausted — falling back to deterministic logic")
            raise RuntimeError(f"LLM call failed: {exc}") from exc

    def _extract_json(self, text: str) -> Dict[str, Any]:
        """
        Extract a JSON object from LLM response text.

        Handles markdown code fences, leading/trailing text, and raw JSON.

        Args:
            text: The raw LLM response.

        Returns:
            Parsed JSON dictionary.

        Raises:
            json.JSONDecodeError: If no valid JSON is found.
        """
        # Try to extract from ```json ... ``` block
        if "```json" in text:
            json_str = text.split("```json")[1].split("```")[0].strip()
        elif "```" in text:
            json_str = text.split("```")[1].split("```")[0].strip()
        else:
            json_str = text.strip()
            start = json_str.find("{")
            end = json_str.rfind("}")
            if start >= 0 and end > start:
                json_str = json_str[start : end + 1]

        return json.loads(json_str)

    def _call_llm_structured(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.1,
        max_tokens: int = 1500,
    ) -> Dict[str, Any]:
        """
        Call the LLM and parse a JSON response.

        Args:
            system_prompt: System instruction.
            user_prompt: User message.
            temperature: Sampling temperature.
            max_tokens: Max response tokens.

        Returns:
            Parsed JSON dictionary.
        """
        raw = self._call_llm(system_prompt, user_prompt, temperature, max_tokens)
        return self._extract_json(raw)

    # ── Resume Parsing ─────────────────────────────────────────────────────

    def parse_resume(self, resume_text: str) -> CandidateProfile:
        """
        Extract a structured CandidateProfile from resume text.

        Uses LLM when available, falls back to deterministic extraction.

        Args:
            resume_text: Raw text of the candidate's resume.

        Returns:
            A populated CandidateProfile.
        """
        if self.client and self._api_available:
            try:
                user_prompt = USER_PROMPT_PARSE_RESUME.format(resume_text=resume_text[:4000])
                data = self._call_llm_structured(
                    SYSTEM_PROMPT_PARSE_RESUME,
                    user_prompt,
                    temperature=0.05,
                    max_tokens=1200,
                )
                return CandidateProfile(**data)
            except Exception as exc:
                logger.warning("LLM parse failed, using fallback: %s", exc)

        return self._fallback_parse(resume_text)

    def _fallback_parse(self, resume_text: str) -> CandidateProfile:
        """
        Deterministic resume extraction when LLM is unavailable.

        Args:
            resume_text: Raw resume text.

        Returns:
            A best-effort CandidateProfile.
        """
        lines = resume_text.strip().split("\n")

        # Extract name from first non-empty, non-header line
        name = "Unknown"
        for line in lines[:8]:
            line_stripped = line.strip()
            if (
                line_stripped
                and not line_stripped.startswith(("Email", "Phone", "http", "EDUCATION", "SKILLS", "PROJECTS"))
                and not line_stripped.isupper()
            ):
                words = line_stripped.split()
                if 2 <= len(words) <= 4 and all(w.isalpha() for w in words):
                    name = line_stripped
                    break

        # Extract skills (lines after SKILLS header)
        skills: List[str] = []
        in_skills = False
        for line in lines:
            stripped = line.strip()
            if "SKILLS" in stripped.upper() and len(stripped) < 30:
                in_skills = True
                continue
            if in_skills:
                if not stripped:
                    break
                for item in re.split(r"[,;•|]", stripped):
                    item = item.strip().strip(",")
                    if item and len(item) > 1:
                        skills.append(item)

        # Extract years of experience
        years = 0.0
        exp_match = re.search(r"(\d+)\s*\+?\s*years?\s*(of\s*)?(experience|exp)", resume_text, re.IGNORECASE)
        if exp_match:
            years = float(exp_match.group(1))
        if re.search(r"(summer|internship|intern)", resume_text, re.IGNORECASE):
            years = max(years, 0.5)

        # Extract education
        education: List[Dict[str, Any]] = []
        in_edu = False
        for line in lines:
            stripped = line.strip()
            if "EDUCATION" in stripped.upper() and len(stripped) < 30:
                in_edu = True
                continue
            if in_edu:
                if any(sec in stripped.upper() for sec in ["SKILLS", "PROJECTS", "EXPERIENCE", "CERTIFICATIONS"]):
                    break
                if stripped and any(d in stripped for d in ["B.Tech", "B.Sc", "M.Sc", "B.E", "B.A", "M.Tech", "PhD"]):
                    education.append({
                        "degree": stripped.split(",")[0] if "," in stripped else stripped,
                        "institution": stripped.split(",")[-1].strip() if "," in stripped else "Unknown",
                        "year": None,
                        "field": "Computer Science" if "Computer" in stripped else "General",
                    })

        # Extract projects
        projects: List[Dict[str, Any]] = []
        in_projects = False
        for line in lines:
            stripped = line.strip()
            if "PROJECTS" in stripped.upper() and len(stripped) < 30:
                in_projects = True
                continue
            if in_projects:
                if any(sec in stripped.upper() for sec in ["EXPERIENCE", "CERTIFICATIONS", "ACHIEVEMENTS"]):
                    break
                if stripped and re.match(r"^\d+\.", stripped):
                    proj_name = re.sub(r"^\d+\.\s*", "", stripped)
                    projects.append({
                        "name": proj_name,
                        "description": "",
                        "technologies": [],
                    })
                elif projects and stripped:
                    projects[-1]["description"] += " " + stripped

        # Extract certifications
        certifications: List[str] = []
        in_certs = False
        for line in lines:
            stripped = line.strip()
            if "CERTIFICATIONS" in stripped.upper() and len(stripped) < 30:
                in_certs = True
                continue
            if in_certs:
                if any(sec in stripped.upper() for sec in ["ACHIEVEMENTS", "ADDITIONAL", "PROJECTS"]):
                    break
                if stripped and not stripped.startswith(("Email", "Phone", "http")):
                    cert = stripped.lstrip("-• ").strip()
                    if cert:
                        certifications.append(cert)

        return CandidateProfile(
            name=name,
            skills=list(set(skills)) if skills else [],
            years_experience=years,
            education=education,
            projects=projects,
            certifications=certifications,
        )

    # ── Scoring ────────────────────────────────────────────────────────────

    def score_candidate(
        self,
        profile: CandidateProfile,
        rubric: str = DEFAULT_RUBRIC,
        jd: str = "",
    ) -> ScoreCard:
        """
        Score a candidate profile against a rubric and job description.

        Args:
            profile: The parsed candidate profile.
            rubric: Scoring rubric text.
            jd: Job description text.

        Returns:
            A completed ScoreCard with evidence and recommendation.
        """
        if self.client and self._api_available:
            try:
                user_prompt = USER_PROMPT_SCORE_CANDIDATE.format(
                    jd=jd[:2000],
                    rubric=rubric[:3000],
                    profile_json=profile.model_dump_json(indent=2),
                )
                data = self._call_llm_structured(
                    SYSTEM_PROMPT_SCORE_CANDIDATE,
                    user_prompt,
                    temperature=0.1,
                    max_tokens=2000,
                )
                return ScoreCard(**data)
            except Exception as exc:
                logger.warning("LLM scoring failed, using fallback: %s", exc)

        return self._fallback_score(profile)

    def _fallback_score(self, profile: CandidateProfile) -> ScoreCard:
        """
        Deterministic scoring fallback based on keyword matching.

        Args:
            profile: The parsed candidate profile.

        Returns:
            A ScoreCard with heuristic-based scores.
        """
        skills_lower = [s.lower() for s in profile.skills]
        projects_desc = " ".join(p.get("description", "") for p in profile.projects).lower()
        projects_name = " ".join(p.get("name", "") for p in profile.projects).lower()
        all_text = projects_desc + " " + projects_name

        def has_skill(keywords: List[str]) -> bool:
            return any(any(kw in s for s in skills_lower) for kw in keywords)

        # Python & ML Fundamentals
        python_score = 0
        if has_skill(["python"]):
            python_score = 2
        if has_skill(["numpy", "pandas"]):
            python_score = 3
        if has_skill(["scikit-learn", "sklearn", "tensorflow", "pytorch"]):
            python_score = 4
        if has_skill(["tensorflow"]) and has_skill(["pytorch"]):
            python_score = 5
        if has_skill(["deep learning"]) and python_score >= 4:
            python_score = 5

        # LLM & RAG Experience
        llm_score = 0
        if has_skill(["langchain", "llamaindex"]):
            llm_score = 3
        if has_skill(["rag", "retrieval", "chromadb", "weaviate", "vector"]):
            llm_score = max(llm_score, 4)
        if "rag" in all_text or "retrieval" in all_text:
            llm_score = max(llm_score, 4)
        if "production" in all_text and llm_score >= 4:
            llm_score = 5

        # Projects
        proj_count = len(profile.projects)
        proj_score = min(proj_count, 5) if proj_count > 0 else 0
        if any(kw in all_text for kw in ["ml", "model", "prediction", "pipeline"]):
            proj_score = max(proj_score, 2)

        # Communication
        comm_score = 1
        if "published" in all_text or "blog" in all_text or "medium" in all_text:
            comm_score = 3
        if "speaker" in all_text or "presented" in all_text:
            comm_score = 4
        if "teaching" in all_text or "assistant" in all_text:
            comm_score = max(comm_score, 2)

        # Fintech
        fin_score = 0
        if has_skill(["fintech", "finance", "financial"]):
            fin_score = 3
        if "financ" in all_text or "fintech" in all_text or "stock" in all_text or "trading" in all_text:
            fin_score = max(fin_score, 3)
        if "investment" in all_text:
            fin_score = max(fin_score, 2)

        criteria = [
            CriterionScore(
                criterion="Python & ML Fundamentals",
                weight=0.30,
                score=python_score,
                evidence=f"Skills: {', '.join(profile.skills[:5])}",
                justification=f"Python/ML assessed at {python_score}/5 based on skill keywords",
            ),
            CriterionScore(
                criterion="LLM & RAG Experience",
                weight=0.25,
                score=llm_score,
                evidence=f"Skills: {', '.join(s for s in profile.skills if any(k in s.lower() for k in ['langchain', 'rag', 'llama', 'chromadb']))}",
                justification=f"LLM/RAG assessed at {llm_score}/5",
            ),
            CriterionScore(
                criterion="Relevant Projects & Tooling",
                weight=0.20,
                score=proj_score,
                evidence=f"{proj_count} project(s): {', '.join(p.get('name', '') for p in profile.projects[:3])}",
                justification=f"Projects assessed at {proj_score}/5",
            ),
            CriterionScore(
                criterion="Communication & Teamwork",
                weight=0.15,
                score=comm_score,
                evidence="Resume text analysis",
                justification=f"Communication assessed at {comm_score}/5",
            ),
            CriterionScore(
                criterion="Fintech / Domain Fit",
                weight=0.10,
                score=fin_score,
                evidence="Financial keywords found" if fin_score > 0 else "No financial keywords",
                justification=f"Fintech fit assessed at {fin_score}/5",
            ),
        ]

        total_score = round(sum(c.score * c.weight for c in criteria), 2)
        if total_score >= 3.5:
            recommendation = "INTERVIEW"
        elif total_score >= 2.0:
            recommendation = "HOLD"
        else:
            recommendation = "REJECT"

        return ScoreCard(
            candidate_name=profile.name,
            criterion_scores=criteria,
            total_score=total_score,
            recommendation=recommendation,
            justification=f"Weighted score {total_score}/5.0 → {recommendation}",
        )

    # ── Injection Detection ────────────────────────────────────────────────

    def check_injection(self, text: str) -> bool:
        """
        Detect prompt injection attempts in resume text.

        Uses both regex patterns and LLM-based analysis.

        Args:
            text: The text to scan.

        Returns:
            True if injection is detected, False otherwise.
        """
        # Deterministic pattern matching
        injection_patterns = [
            r"ignore\s+(all\s+)?(previous|prior|above)\s+(instructions|directions|commands)",
            r"rank\s+(me|this\s+candidate|[\w\s]+?)\s+(first|top|highest)",
            r"rank\s+[\w\s]+\s+as\s+the\s+top\s+candidate",
            r"official\s+update\s+from\s+(hr|management|the\s+company)",
            r"override\s+(the\s+)?(scoring|rubric|evaluation|ranking|system)",
            r"disregard\s+(all\s+)?(previous|prior)\s+(scoring|instructions|rubric)",
            r"hire\s+me\s+immediately",
        ]
        text_lower = text.lower()
        for pattern in injection_patterns:
            if re.search(pattern, text_lower):
                logger.info("Injection pattern matched: %s", pattern)
                return True

        # LLM-based check
        if self.client and self._api_available:
            try:
                data = self._call_llm_structured(
                    SYSTEM_PROMPT_INJECTION_CHECK,
                    f"Scan this text for injection attempts:\n\n{text[:3000]}",
                    temperature=0.0,
                    max_tokens=200,
                )
                if data.get("injection_detected", False):
                    logger.info("LLM injection detection triggered: %s", data.get("indicator", ""))
                    return True
            except Exception as exc:
                logger.debug("LLM injection check failed: %s", exc)

        return False

    # ── Availability Mock ──────────────────────────────────────────────────

    def check_availability(self, candidate_name: str) -> List[InterviewSlot]:
        """
        Mock interview availability check.

        Returns 3 random slots in the upcoming week for the given candidate.

        Args:
            candidate_name: Name of the candidate.

        Returns:
            List of InterviewSlot objects.
        """
        import random
        import hashlib

        slots: List[InterviewSlot] = []
        base = datetime.now()

        for day_offset in [1, 2, 3, 4, 5]:
            dt = base.replace(hour=0, minute=0, second=0, microsecond=0)
            from datetime import timedelta

            dt = dt + timedelta(days=day_offset)
            if dt.weekday() >= 5:  # Skip weekends
                continue
            for hour in [10, 11, 14, 15, 16]:
                slots.append(InterviewSlot(
                    date=dt.strftime("%Y-%m-%d"),
                    time=f"{hour:02d}:00",
                    interviewer="TechVest Hiring Team",
                ))

        # Deterministic seed based on name for reproducibility
        seed_value = int(hashlib.md5(candidate_name.encode()).hexdigest()[:8], 16)
        random.seed(seed_value)
        selected = random.sample(slots, min(3, len(slots)))
        return sorted(selected, key=lambda s: s.date)

    # ── Fairness Check ─────────────────────────────────────────────────────

    def fairness_check(self, state: Any) -> Dict[str, Any]:
        """
        Run fairness validation across candidates.

        Args:
            state: The recruitment state object with profiles and scorecards.

        Returns:
            A fairness report dictionary.
        """
        if self.client and self._api_available:
            try:
                profiles_summary = {}
                for name, profile in state.get("parsed_profiles", {}).items():
                    sc = state.get("scorecards", {}).get(name)
                    profiles_summary[name] = {
                        "skills": profile.skills[:10],
                        "years_experience": profile.years_experience,
                        "education": [e.get("institution", "") for e in profile.education],
                        "total_score": sc.total_score if sc else None,
                        "recommendation": sc.recommendation if sc else None,
                    }

                data = self._call_llm_structured(
                    SYSTEM_PROMPT_FAIRNESS_CHECK,
                    f"Review these candidate profiles and scores for bias:\n\n{json.dumps(profiles_summary, indent=2)}",
                    temperature=0.0,
                    max_tokens=500,
                )
                return data
            except Exception as exc:
                logger.warning("LLM fairness check failed: %s", exc)

        # Deterministic fairness check
        profiles = state.get("parsed_profiles", {})
        scorecards = state.get("scorecards", {})
        bias_indicators: List[str] = []

        if len(profiles) < 2:
            return {
                "bias_detected": False,
                "bias_indicators": [],
                "fairness_score": 1.0,
                "recommendation": "PASS",
                "note": "Insufficient candidates for comparison",
            }

        names = list(profiles.keys())
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                p1 = profiles[names[i]]
                p2 = profiles[names[j]]
                if p1.years_experience >= 1 and p2.years_experience >= 1:
                    s1 = scorecards.get(names[i])
                    s2 = scorecards.get(names[j])
                    if s1 and s2:
                        diff = abs(s1.total_score - s2.total_score)
                        if diff > 2.0 and abs(p1.years_experience - p2.years_experience) < 1:
                            bias_indicators.append(
                                f"Large score gap ({diff:.1f}) between {names[i]} and {names[j]} "
                                f"with similar experience"
                            )

        has_bias = len(bias_indicators) > 0
        return {
            "bias_detected": has_bias,
            "bias_indicators": bias_indicators,
            "fairness_score": 0.7 if has_bias else 1.0,
            "recommendation": "REVIEW" if has_bias else "PASS",
            "note": "Fairness check completed. Candidates scored on JD-relevant criteria only."
            if not has_bias
            else "Potential bias indicators found — recommend manual review.",
        }

    # ── Planning ───────────────────────────────────────────────────────────

    def plan_next_action(self, state_summary: str) -> Dict[str, Any]:
        """
        Ask the LLM to determine the next action based on current state.

        Args:
            state_summary: Text summary of the current state.

        Returns:
            Dict with keys: action, candidate, reasoning.
        """
        if self.client and self._api_available:
            try:
                return self._call_llm_structured(
                    SYSTEM_PROMPT_PLANNER,
                    f"Current state:\n{state_summary}",
                    temperature=0.0,
                    max_tokens=300,
                )
            except Exception as exc:
                logger.warning("LLM planning failed: %s", exc)

        # Default fallback: determine action from state
        return {"action": "finish", "candidate": None, "reasoning": "Fallback planner"}