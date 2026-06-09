"""Core package: embedding, LLM client, storage, metrics, and engine helpers."""
from __future__ import annotations

import os

# Repository root — computed once here so sub-modules don't duplicate the path logic.
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
