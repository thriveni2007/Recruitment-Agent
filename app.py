"""
Entry point for the TechVest Recruitment Agent.

Run with: streamlit run recruitment_agent/app.py
or:      python -m recruitment_agent
"""

from __future__ import annotations

import logging
import os
import sys

# Ensure the parent directory is on the path for imports
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PARENT_DIR = os.path.dirname(_THIS_DIR)
if _PARENT_DIR not in sys.path:
    sys.path.insert(0, _PARENT_DIR)

# Set up basic logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stdout,
)

# Load .env file if available
try:
    from dotenv import load_dotenv

    env_path = os.path.join(_THIS_DIR, ".env")
    if os.path.isfile(env_path):
        load_dotenv(env_path)
except ImportError:
    pass

# Import the UI
from recruitment_agent.ui.streamlit_ui import render_main

if __name__ == "__main__":
    render_main()