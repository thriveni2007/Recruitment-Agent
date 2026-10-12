"""
Data loading utilities for candidate resumes and job description.
Provides Python access to the text files in data/ directory.
"""

from __future__ import annotations

import os
from typing import Dict

_DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


def _load(filename: str) -> str:
    """Load a text file from the data directory."""
    path = os.path.join(_DATA_DIR, filename)
    if os.path.isfile(path):
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    return ""


# Job description
jd: str = _load("jd.txt")

# Candidate resumes
candidates: Dict[str, str] = {
    "Priya Sharma": _load("priya_resume.txt"),
    "Rahul Verma": _load("rahul_resume.txt"),
    "Meera Patel": _load("meera_resume.txt"),
}