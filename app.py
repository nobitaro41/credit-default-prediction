"""Entry point for Hugging Face Spaces and Streamlit deployment."""

from __future__ import annotations

import sys
from pathlib import Path

root_dir = Path(__file__).resolve().parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))
if str(root_dir / "app") not in sys.path:
    sys.path.insert(0, str(root_dir / "app"))

import streamlit_app  # noqa: E402

if __name__ == "__main__":
    streamlit_app.main()
