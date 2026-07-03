"""Reusable Streamlit interface package for the learning-support platform."""

from .navigation import PAGES, render_navigation
from .theme import apply_theme

__all__ = ["PAGES", "apply_theme", "render_navigation"]

