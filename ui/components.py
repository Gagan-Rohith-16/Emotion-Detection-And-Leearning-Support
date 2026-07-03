"""Reusable, safe HTML and native Streamlit interface components."""

from __future__ import annotations

from html import escape

import streamlit as st


def render_brand(compact: bool = False) -> None:
    """Render the product mark and name in full or compact form."""

    name = "" if compact else "<strong>EmotiLearn AI</strong>"
    st.markdown(
        f'<div style="display:flex;align-items:center;gap:.7rem;margin-bottom:1.4rem">'
        f'<span class="brand-mark">✦</span>{name}</div>',
        unsafe_allow_html=True,
    )


def render_page_header(eyebrow: str, title: str, description: str) -> None:
    """Render a consistent heading block with escaped user-safe content."""

    st.markdown(
        f'<div class="eyebrow">{escape(eyebrow)}</div>'
        f'<h1 style="margin:.1rem 0 .55rem">{escape(title)}</h1>'
        f'<p class="hero-copy">{escape(description)}</p>',
        unsafe_allow_html=True,
    )


def render_card(icon: str, title: str, body: str) -> None:
    """Render an animated informational card with HTML escaping."""

    st.markdown(
        f'<div class="glass-card"><div class="card-icon">{escape(icon)}</div>'
        f'<div class="card-title">{escape(title)}</div>'
        f'<div class="card-copy">{escape(body)}</div></div>',
        unsafe_allow_html=True,
    )


def render_metric(value: str, label: str) -> None:
    """Render a compact dashboard metric tile."""

    st.markdown(
        f'<div class="glass-card"><div class="metric-value">{escape(value)}</div>'
        f'<div class="metric-label">{escape(label)}</div></div>',
        unsafe_allow_html=True,
    )


def render_module_status(module_name: str, message: str) -> None:
    """Display a polished readiness state for a forthcoming module."""

    st.markdown(
        '<div class="glass-card" style="margin-top:1.25rem">'
        '<span class="status-pill"><span class="dot"></span>Interface ready</span>'
        f'<div class="card-title" style="margin-top:1rem">{escape(module_name)}</div>'
        f'<div class="card-copy">{escape(message)}</div></div>',
        unsafe_allow_html=True,
    )

