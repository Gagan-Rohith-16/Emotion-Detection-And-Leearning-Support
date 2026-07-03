"""Streamlit entry point for the Emotion Detection Learning Support platform."""

from __future__ import annotations

import streamlit as st

from database import DatabaseManager
from ui.auth import initialize_auth_state
from ui.navigation import render_navigation
from ui.pages import render_page
from ui.theme import apply_theme


@st.cache_resource
def get_database() -> DatabaseManager:
    """Create one reusable database manager per Streamlit server process."""

    return DatabaseManager()


def main() -> None:
    """Configure the application and render the currently selected page."""

    st.set_page_config(
        page_title="EmotiLearn AI",
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    initialize_auth_state(get_database())
    apply_theme(dark_mode=False)
    database = get_database()
    selected_page = render_navigation(database)
    render_page(selected_page, database)


if __name__ == "__main__":
    main()
