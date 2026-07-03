"""Sidebar navigation and session-aware product controls."""

from __future__ import annotations

import streamlit as st

from database import DatabaseManager

from .components import render_brand


PAGES: tuple[tuple[str, str], ...] = (
    ("Home", "🏠"),
    ("Dashboard", "✨"),
    ("Prediction", "🧠"),
    ("Analytics", "📊"),
    ("History", "🕘"),
    ("Settings", "⚙️"),
    ("About", "💡"),
)


def render_navigation(database: DatabaseManager) -> str:
    """Render responsive sidebar navigation and return the selected page."""

    with st.sidebar:
        render_brand()
        labels = [f"{icon}  {name}" for name, icon in PAGES]
        current_page = st.session_state.get("current_page", "Home")
        current_index = next(
            (index for index, (name, _) in enumerate(PAGES) if name == current_page), 0
        )
        selected_label = st.radio(
            "Main navigation", labels, index=current_index, label_visibility="collapsed"
        )
        selected_page = selected_label.split("  ", maxsplit=1)[-1]
        st.session_state.current_page = selected_page
        st.markdown("---")
        user = st.session_state.get("user")

        if user:
            
            st.markdown(
                f"""
                <div style="text-align:center;">
                    <h4 style="margin-bottom:0;">{user.name}</h4>
                    <p style="color:gray;margin-top:0;">{user.email}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            if st.button("Sign out", use_container_width=True):
                from ui.auth import clear_auth_token

                clear_auth_token(database)
                st.session_state.user = None
                st.rerun()

        else:
            st.caption("Your progress stays private and secure.")
        return selected_page

