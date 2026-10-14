"""
Gemma-2 Architectural Ablation Framework (Question 2)
-----------------------------------------------------
Modular package for performing GQA head surgery, position-aware
layer ablation, and spectral embedding masking on Google Gemma-2.
Built on top of the unified common_gemma foundation engine.
"""

import os
import sys

# Ensure repository root is on sys.path so common_gemma is importable
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

__version__ = "0.2.0"
