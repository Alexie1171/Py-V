"""
model_loader.py — PY-V (inference/engine/)
Thin inference-layer wrapper around the shared model loader.
Keeps inference/engine independent while avoiding duplication.
"""

from model.utils.model_loader import load_model

__all__ = ["load_model"]